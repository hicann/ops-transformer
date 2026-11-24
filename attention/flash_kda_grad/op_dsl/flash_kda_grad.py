# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host API, workspace lifecycle, compilation and explicit template selection."""

from __future__ import annotations
import atexit
import cannbotdsl
import threading
import torch
from cannbotdsl import Constexpr, aot, dtypes, get_platform_info
from cannbotdsl.lang.host import host
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from queue import Queue
from typing import NamedTuple, Optional

if __package__:
    from .flash_kda_grad_check import (
        BackwardInputs,
        _validate_input_structure,
        _validate_saved_cache_contract,
        _validate_chunk_heads,
        _validate_forward_outputs,
        _validate_scale,
        _validate_state_backward_inputs,
    )
    from .flash_kda_grad_kernel import FlashKDAGradKernel
    from .flash_kda_grad_buffer import (
        CHUNK_SIZE,
        GATE_SCALAR_TRANSFER_WIDTH,
        SUPPORTED_HEAD_DIM,
    )
    from .flash_kda_grad_metadata import (
        HEADER_FLAGS,
        wait_metadata_ready,
        HEADER_WORDS,
        MAX_AIC_CORES,
        TASK_WORDS,
    )
    from . import flash_kda_grad_metadata as _metadata
else:
    from flash_kda_grad_check import (
        BackwardInputs,
        _validate_input_structure,
        _validate_saved_cache_contract,
        _validate_chunk_heads,
        _validate_forward_outputs,
        _validate_scale,
        _validate_state_backward_inputs,
    )
    from flash_kda_grad_kernel import FlashKDAGradKernel
    from flash_kda_grad_buffer import (
        CHUNK_SIZE,
        GATE_SCALAR_TRANSFER_WIDTH,
        SUPPORTED_HEAD_DIM,
    )
    from flash_kda_grad_metadata import (
        HEADER_FLAGS,
        wait_metadata_ready,
        HEADER_WORDS,
        MAX_AIC_CORES,
        TASK_WORDS,
    )
    import flash_kda_grad_metadata as _metadata


class BackwardConfig(NamedTuple):
    """Named Host fields, flattened in this order only at the launch boundary."""

    scale: float
    fused_beta: int
    inverse_factor: float
    heads_per_key_head: int
    fused_gate: int
    has_bias: int
    use_l2norm: int
    softplus: int
    lower_bound: float
    beta_fp32: bool
    padded_layout: int
    use_full_chunks: bool
    needs_masked_chunks: bool
    staged: bool
    cube_causal: bool

    @classmethod
    def from_abi(cls, value):
        return value if isinstance(value, cls) else cls(*value)

    def scalar_specs(self, dtypes):
        compile_fields = {
            "beta_fp32",
            "use_full_chunks",
            "needs_masked_chunks",
            "staged",
            "cube_causal",
        }
        float_fields = {"scale", "inverse_factor", "lower_bound"}
        return tuple(
            getattr(self, name)
            if name in compile_fields
            else dtypes.float32
            if name in float_fields
            else dtypes.int64
            for name in self._fields
        )

    def runtime_values(self):
        compile_fields = {
            "beta_fp32",
            "use_full_chunks",
            "needs_masked_chunks",
            "staged",
            "cube_causal",
        }
        float_fields = {"scale", "inverse_factor", "lower_bound"}
        return tuple(
            float(getattr(self, name))
            if name in float_fields
            else int(getattr(self, name))
            for name in self._fields
            if name not in compile_fields
        )


@dataclass(frozen=True)
class BackwardPlan:
    core_num: int
    use_full_chunks: bool
    needs_masked_chunks: bool
    staged: bool
    cube_causal: bool
    gradient_slots: int

    @property
    def template(self):
        return "staged" if self.staged else "persistent"


def _resolve_chunk_template(full, masked, staged, *, recompute, round_u_pre=False):
    """Resolve legal cache/schedule combinations before workspace allocation."""
    if recompute and not staged and (masked or round_u_pre):
        return False, True, True
    return full, masked, staged


def make_backward_plan(
    metadata, query_heads, value_heads, core_num, *, use_gate, lower_bound, recompute
):
    staged = (
        _metadata.backward_schedule(
            metadata, query_heads, value_heads, core_num
        ).strategy
        == "staged"
    )
    block_size = CHUNK_SIZE if use_gate and lower_bound is not None else 1
    full, masked = _metadata.backward_chunk_paths(
        metadata, query_heads, value_heads, block_size, staged
    )
    full, masked, staged = _resolve_chunk_template(
        full,
        masked,
        staged,
        recompute=recompute,
        round_u_pre=getattr(metadata, "_flash_kda_round_recomputed_u_pre", False),
    )
    cube_causal = bool(
        full
        and use_gate
        and lower_bound is not None
        and abs(lower_bound) * (CHUNK_SIZE - 1) <= 70.0
        and not staged
    )
    return BackwardPlan(
        core_num,
        full,
        masked,
        staged,
        cube_causal,
        _metadata.backward_gradient_slots(metadata, value_heads, core_num, staged),
    )


@dataclass
class _WorkspaceEntry:
    device: object
    tensors: dict = field(default_factory=dict)
    event: object = None


class _WorkspaceLease:
    def __init__(self, pool, entry, stream):
        self.pool, self.entry, self.stream = pool, entry, stream
        self.position = 0

    def empty(self, shape, dtype):
        shape = tuple(shape)
        key = (self.position, shape, dtype)
        self.position += 1
        tensor = self.entry.tensors.get(key)
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=self.entry.device)
            size = tensor.numel() * tensor.element_size()
            occupied = sum(
                value.numel() * value.element_size()
                for entry in self.pool.entries.values()
                if entry.device == self.entry.device
                for value in entry.tensors.values()
            )
            if size <= self.pool.item_limit and occupied + size <= self.pool.byte_limit:
                self.entry.tensors[key] = tensor
        # Raw DSL launches do not tell the allocator about their pointer uses.
        tensor.record_stream(self.stream)
        return tensor


class WorkspacePool:
    """Small per-device scratch pool with stream isolation and bounded retention.

    The lock covers preparation and launch, so another host thread cannot take
    an entry before its completion event is recorded. Each caller stream owns
    its scratch, so independent streams never wait for another compute call.
    """

    def __init__(self, byte_limit=32 * 1024 * 1024, item_limit=1024 * 1024, limit=2):
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.byte_limit, self.item_limit, self.limit = byte_limit, item_limit, limit
        self.pending_events = []

    @contextmanager
    def acquire(self, ref, key):
        with torch.npu.device(ref.device), self.lock:
            stream = torch.npu.current_stream(torch.npu.current_device())
            key = (str(ref.device), int(stream.npu_stream), *key)
            entry = self.entries.get(key)
            if entry is None:
                # Evicted tensors have recorded their consumer stream; freeing
                # the host owners cannot recycle storage before that use ends.
                matching = [
                    k for k, value in self.entries.items() if value.device == ref.device
                ]
                if len(matching) >= self.limit:
                    del self.entries[matching[0]]
                entry = _WorkspaceEntry(ref.device)
                self.entries[key] = entry
            self.entries.move_to_end(key)
            if entry.event is not None:
                stream.wait_event(entry.event)
            finished = torch.npu.Event()
            try:
                yield _WorkspaceLease(self, entry, stream)
                finished.record(stream)
                entry.event = finished
                self.pending_events[:] = [
                    (device, event)
                    for device, event in self.pending_events
                    if not event.query()
                ]
                self.pending_events.append((ref.device, finished))
            except BaseException:
                # Retire possibly submitted raw-pointer work before reusing or
                # releasing storage after a failed launch.
                stream.synchronize()
                self.entries.pop(key, None)
                raise

    def clear(self):
        with self.lock:
            # Include evicted entries: their code may still be executing.
            for device, event in self.pending_events:
                with torch.npu.device(device):
                    event.synchronize()
            self.pending_events.clear()
            self.entries.clear()


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """Use the resident Cube core count, capped for whole-grid barriers."""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return MAX_AIC_CORES
    npu = getattr(torch, "npu", None)
    if npu is None:
        return MAX_AIC_CORES
    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            return MAX_AIC_CORES
    platform_info = get_platform_info(stream=npu.current_stream(device_index))
    try:
        cube_core_num = int(platform_info.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU {device_index} has no valid cube_core_num") from exc
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports cube_core_num={cube_core_num}")
    return min(cube_core_num, MAX_AIC_CORES)


@dataclass(frozen=True)
class StateBackwardResult:
    d_v_new: torch.Tensor
    d_Aqk: torch.Tensor
    d_qg: torch.Tensor
    d_kg: torch.Tensor
    d_w: torch.Tensor
    d_gate_last: torch.Tensor
    dh0: torch.Tensor


def _backward_tensor_spec(tensor, dtype):
    return cannbotdsl.TensorSpec(
        tuple(tensor.shape), dtype, stride=tuple(tensor.stride())
    )


def _prepare_state_backward(
    *,
    aqk: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    v_new: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_states: torch.Tensor,
    output_gradient: torch.Tensor,
    final_state_gradient: torch.Tensor,
    layout: str,
    metadata: torch.Tensor,
    scale: float,
    core_num: int = 1,
    _workspace=None,
    _needs_masked_chunks: bool = True,
    _staged: bool = False,
    _gradient_slots: int | None = None,
):
    """Prepare inputs and per-core or global-history backward workspaces."""
    batch, heads = _validate_state_backward_inputs(
        aqk=aqk,
        qg=qg,
        kg=kg,
        w=w,
        v_new=v_new,
        g_cumsum=g_cumsum,
        chunk_states=chunk_states,
        layout=layout,
        metadata=metadata,
    )
    is_tnd = layout == "TND"
    device = qg.device

    def logical_tokens(tensor):
        return tensor.unsqueeze(0).transpose(1, 2)

    def input_view(tensor):
        return (
            logical_tokens(tensor)
            if layout == "TND"
            else tensor
            if layout == "BNSD"
            else tensor.transpose(1, 2)
        )

    qg_view = input_view(qg)
    kg_view = input_view(kg)
    w_view = input_view(w)
    v_new_view = input_view(v_new)
    gate_view = input_view(g_cumsum)
    matrix_view = aqk.unsqueeze(0) if is_tnd else aqk
    state_view = chunk_states.unsqueeze(0).transpose(1, 2) if is_tnd else chunk_states
    output_view = input_view(output_gradient)
    terminal_gradient = final_state_gradient

    # Staged owners exchange all chunks through GM. One additional slot per
    # task keeps empty sequences and TASK_FLAT_CHUNK_START + task collision-free.
    gradient_slots = (
        _gradient_slots
        if _gradient_slots is not None
        else _metadata.backward_gradient_slots(metadata, heads, core_num, _staged)
    )
    gradient_shape = (gradient_slots * CHUNK_SIZE, 1, SUPPORTED_HEAD_DIM)
    d_v_new = torch.empty(gradient_shape, dtype=torch.float32, device=device)
    d_qg = torch.empty_like(d_v_new)
    d_kg = torch.empty_like(d_v_new)
    d_w = torch.empty_like(d_v_new)
    d_Aqk = torch.empty(
        (gradient_slots, 1, CHUNK_SIZE, CHUNK_SIZE), dtype=torch.float32, device=device
    )
    d_gate_last = torch.empty(
        (gradient_slots, 1, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=device
    )
    dh0 = torch.empty(
        (batch, heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )

    # Persistent full chunks keep state in UB. Masked chunks use two FP32
    # slots per core; staged execution keeps both slots for every chunk.
    masked_cores = core_num if _needs_masked_chunks or _staged else 0
    state_slots = 2 * gradient_slots if _staged else 2 * masked_cores
    state_gradients = torch.empty(
        (state_slots * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )

    def scratch(shape, dtype):
        if _workspace is not None:
            return _workspace.empty(shape, dtype)
        return torch.empty(shape, dtype=dtype, device=device)

    masked_aqk = scratch((masked_cores * CHUNK_SIZE, CHUNK_SIZE), torch.bfloat16)
    masked_qg, masked_v_new, masked_d_output = (
        scratch((masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.bfloat16)
        for _ in range(3)
    )
    d_v_new_output, d_kg_scratch = (
        scratch((masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32)
        for _ in range(2)
    )
    d_v_new_scratch = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32
    )
    d_v_new_low = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.bfloat16
    )
    d_qg_scratch = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32
    )
    d_w_scratch = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32
    )
    state_high = scratch(
        (masked_cores * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), torch.bfloat16
    )
    state_residual = scratch(
        (masked_cores * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), torch.bfloat16
    )
    saved_state_high = scratch(
        (masked_cores * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), torch.bfloat16
    )

    arguments = (
        matrix_view,
        qg_view,
        kg_view,
        w_view,
        v_new_view,
        gate_view,
        state_view,
        output_view,
        terminal_gradient,
        metadata,
        logical_tokens(d_v_new),
        d_Aqk.view(-1, CHUNK_SIZE),
        logical_tokens(d_qg),
        logical_tokens(d_kg),
        logical_tokens(d_w),
        logical_tokens(d_gate_last),
        dh0,
        state_gradients,
        masked_aqk,
        masked_qg,
        masked_v_new,
        masked_d_output,
        d_v_new_output,
        d_kg_scratch,
        d_v_new_scratch,
        d_v_new_low,
        d_qg_scratch,
        d_w_scratch,
        state_high,
        state_residual,
        saved_state_high,
    )
    dtype_map = {
        torch.bfloat16: dtypes.bfloat16,
        torch.float32: dtypes.float32,
        torch.int32: dtypes.int32,
    }
    specs = tuple(
        _backward_tensor_spec(tensor, dtype_map[tensor.dtype]) for tensor in arguments
    )
    result = StateBackwardResult(d_v_new, d_Aqk, d_qg, d_kg, d_w, d_gate_last, dh0)
    return result, arguments, specs


class FlashKDAGrad:
    @host
    def run(
        self,
        stage_two_gm_Aqk,
        stage_two_gm_qg,
        stage_two_gm_kg,
        stage_two_gm_w,
        stage_two_gm_v_new,
        stage_two_gm_g_cumsum,
        stage_two_gm_chunk_states,
        stage_two_gm_output_gradient,
        stage_two_gm_final_state_gradient,
        stage_two_gm_task_metadata,
        stage_two_gm_d_v_new,
        stage_two_gm_d_Aqk,
        stage_two_gm_d_qg,
        stage_two_gm_d_kg,
        stage_two_gm_d_w,
        stage_two_gm_d_gate_last,
        stage_two_gm_dh0,
        stage_two_gm_state_gradients,
        stage_two_gm_masked_aqk,
        stage_two_gm_masked_qg,
        stage_two_gm_masked_v_new,
        stage_two_gm_masked_output_gradient,
        stage_two_gm_d_v_new_output,
        stage_two_gm_d_kg_scratch,
        stage_two_gm_d_v_new_scratch,
        stage_two_gm_d_v_new_low,
        stage_two_gm_d_qg_scratch,
        stage_two_gm_d_w_scratch,
        stage_two_gm_state_high,
        stage_two_gm_state_residual,
        stage_two_gm_saved_state_high,
        stage_one_gm_effective_query,
        stage_one_gm_effective_key,
        stage_one_gm_value,
        stage_one_gm_activated_beta,
        stage_one_gm_Akk,
        stage_one_gm_saved_qg,
        stage_one_gm_saved_kg,
        stage_one_gm_g_cumsum,
        stage_one_gm_stage_two_d_v_new,
        stage_one_gm_stage_two_d_w,
        stage_one_gm_stage_two_d_qg,
        stage_one_gm_stage_two_d_kg,
        stage_one_gm_stage_two_d_Aqk,
        stage_one_gm_stage_two_d_gate_last,
        stage_one_gm_task_metadata,
        stage_one_gm_public_d_value,
        stage_one_gm_public_d_beta,
        stage_one_gm_raw_gate,
        stage_one_gm_A_log,
        stage_one_gm_dt_bias,
        stage_one_gm_q_rstd,
        stage_one_gm_k_rstd,
        stage_one_gm_public_d_query,
        stage_one_gm_public_d_key,
        stage_one_gm_public_d_gate,
        stage_one_gm_public_dA,
        stage_one_gm_public_dBias,
        stage_one_gm_inverse,
        stage_one_gm_values,
        stage_one_gm_keys,
        stage_one_gm_gamma,
        stage_one_gm_beta,
        stage_one_gm_d_v_new_low,
        stage_one_gm_d_w_low,
        stage_one_gm_weighted_v,
        stage_one_gm_weighted_k,
        stage_one_gm_d_beta_v,
        stage_one_gm_d_beta_k,
        stage_one_gm_d_beta_wy,
        stage_one_gm_d_lower_unmasked,
        stage_one_gm_qg,
        stage_one_gm_kg,
        stage_one_gm_d_qg,
        stage_one_gm_d_kg,
        stage_one_gm_d_gate_last,
        stage_one_gm_d_query_base,
        stage_one_gm_d_key_base,
        stage_one_gm_d_gate_base,
        stage_one_gm_query,
        stage_one_gm_d_Aqk,
        stage_one_gm_d_beta_matrix,
        stage_one_gm_d_query_pair,
        stage_one_gm_d_key_pair,
        stage_one_gm_d_gate_pair,
        stage_one_gm_d_beta_pair,
        stage_one_gm_d_value,
        stage_one_gm_d_beta,
        gm_owner_schedule,
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_fused_beta: dtypes.int64,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_heads_per_key_head: dtypes.int64,
        stage_one_fused_gate: dtypes.int64,
        stage_one_has_bias: dtypes.int64,
        stage_one_use_l2norm: dtypes.int64,
        stage_one_softplus: dtypes.int64,
        stage_one_lower_bound: dtypes.float32,
        stage_one_beta_fp32: Constexpr[bool],
        padded_layout: dtypes.int64,
        use_full_chunks: Constexpr[bool],
        needs_masked_chunks: Constexpr[bool],
        staged: Constexpr[bool],
        cube_causal: Constexpr[bool],
        has_dht: dtypes.int64,
        core_num: int,
    ):
        FlashKDAGradKernel()[core_num](
            stage_two_gm_Aqk,
            stage_two_gm_qg,
            stage_two_gm_kg,
            stage_two_gm_w,
            stage_two_gm_v_new,
            stage_two_gm_g_cumsum,
            stage_two_gm_chunk_states,
            stage_two_gm_output_gradient,
            stage_two_gm_final_state_gradient,
            stage_two_gm_task_metadata,
            stage_two_gm_d_v_new,
            stage_two_gm_d_Aqk,
            stage_two_gm_d_qg,
            stage_two_gm_d_kg,
            stage_two_gm_d_w,
            stage_two_gm_d_gate_last,
            stage_two_gm_dh0,
            stage_two_gm_state_gradients,
            stage_two_gm_masked_aqk,
            stage_two_gm_masked_qg,
            stage_two_gm_masked_v_new,
            stage_two_gm_masked_output_gradient,
            stage_two_gm_d_v_new_output,
            stage_two_gm_d_kg_scratch,
            stage_two_gm_d_v_new_scratch,
            stage_two_gm_d_v_new_low,
            stage_two_gm_d_qg_scratch,
            stage_two_gm_d_w_scratch,
            stage_two_gm_state_high,
            stage_two_gm_state_residual,
            stage_two_gm_saved_state_high,
            stage_one_gm_effective_query,
            stage_one_gm_effective_key,
            stage_one_gm_value,
            stage_one_gm_activated_beta,
            stage_one_gm_Akk,
            stage_one_gm_saved_qg,
            stage_one_gm_saved_kg,
            stage_one_gm_g_cumsum,
            stage_one_gm_stage_two_d_v_new,
            stage_one_gm_stage_two_d_w,
            stage_one_gm_stage_two_d_qg,
            stage_one_gm_stage_two_d_kg,
            stage_one_gm_stage_two_d_Aqk,
            stage_one_gm_stage_two_d_gate_last,
            stage_one_gm_task_metadata,
            stage_one_gm_public_d_value,
            stage_one_gm_public_d_beta,
            stage_one_gm_raw_gate,
            stage_one_gm_A_log,
            stage_one_gm_dt_bias,
            stage_one_gm_q_rstd,
            stage_one_gm_k_rstd,
            stage_one_gm_public_d_query,
            stage_one_gm_public_d_key,
            stage_one_gm_public_d_gate,
            stage_one_gm_public_dA,
            stage_one_gm_public_dBias,
            stage_one_gm_inverse,
            stage_one_gm_values,
            stage_one_gm_keys,
            stage_one_gm_gamma,
            stage_one_gm_beta,
            stage_one_gm_d_v_new_low,
            stage_one_gm_d_w_low,
            stage_one_gm_weighted_v,
            stage_one_gm_weighted_k,
            stage_one_gm_d_beta_v,
            stage_one_gm_d_beta_k,
            stage_one_gm_d_beta_wy,
            stage_one_gm_d_lower_unmasked,
            stage_one_gm_qg,
            stage_one_gm_kg,
            stage_one_gm_d_qg,
            stage_one_gm_d_kg,
            stage_one_gm_d_gate_last,
            stage_one_gm_d_query_base,
            stage_one_gm_d_key_base,
            stage_one_gm_d_gate_base,
            stage_one_gm_query,
            stage_one_gm_d_Aqk,
            stage_one_gm_d_beta_matrix,
            stage_one_gm_d_query_pair,
            stage_one_gm_d_key_pair,
            stage_one_gm_d_gate_pair,
            stage_one_gm_d_beta_pair,
            stage_one_gm_d_value,
            stage_one_gm_d_beta,
            gm_owner_schedule,
            stage_two_gm_final_state_gradient,
            stage_two_gm_w,
            stage_one_fused_beta,
            stage_one_heads_per_key_head,
            stage_one_fused_gate,
            stage_one_has_bias,
            stage_one_use_l2norm,
            stage_one_softplus,
            stage_one_beta_fp32,
            padded_layout,
            use_full_chunks,
            needs_masked_chunks,
            cube_causal,
            0,
            0,
            stage_two_scale,
            stage_one_scale,
            stage_one_inverse_factor,
            stage_one_lower_bound,
            has_dht,
            staged,
            False,
        )


def _prepare_chunk_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    values: torch.Tensor,
    raw_gate: torch.Tensor,
    activated_beta: torch.Tensor,
    saved_inverse: torch.Tensor,
    saved_qg: torch.Tensor,
    saved_kg: torch.Tensor,
    saved_gamma: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    query_rstd: torch.Tensor | None,
    key_rstd: torch.Tensor | None,
    stage_two: StateBackwardResult,
    layout: str,
    metadata: torch.Tensor,
    scale: float,
    fused_beta: bool,
    fused_gate: bool,
    use_l2norm: bool,
    inverse_factor: float,
    lower_bound: float | None,
    beta_dtype: torch.dtype = torch.bfloat16,
    _workspace=None,
):
    """Prepare local chunk scratch and public gradient outputs without launching."""
    device = values.device
    value_heads = (
        values.shape[1]
        if layout == "TND"
        else (values.shape[1] if layout == "BNSD" else values.shape[2])
    )
    qk_heads = query.shape[1] if query.ndim == 3 or layout == "BNSD" else query.shape[2]
    _validate_chunk_heads(qk_heads, value_heads)
    heads_per_key_head = value_heads // qk_heads
    has_tokens = saved_qg.numel() > 0
    # GQA retains only the current chunk's Q/K contributions for this core.
    # All other intermediates are also reused when this owner advances.
    scratch_chunks = _device_block_num(values) * heads_per_key_head if has_tokens else 0
    rows = scratch_chunks * CHUNK_SIZE

    def as_bnsd_view(tensor, scalar=False):
        if scalar:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0).transpose(1, 2)
        if layout == "BSND":
            return tensor.transpose(1, 2)
        return tensor

    d_value = torch.empty_like(values)
    # Allocate the gradient in the public BSN/TN GM layout. Kernel views
    # preserve those strides and select the strided scalar store as needed.
    if layout == "TND":
        d_beta = torch.empty(
            (values.shape[0], value_heads), dtype=beta_dtype, device=device
        )
    else:
        length = values.shape[2 if layout == "BNSD" else 1]
        d_beta = torch.empty(
            (values.shape[0], length, value_heads), dtype=beta_dtype, device=device
        )
        if layout == "BNSD":
            d_beta = d_beta.transpose(1, 2)
    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_gate = torch.empty_like(raw_gate)
    # The first rows hold the public outputs. Each AIV has its own FP32 rows.
    parameter_rows = value_heads * (1 + 2 * _device_block_num(values))
    alog_width = 2 * GATE_SCALAR_TRANSFER_WIDTH
    dA_padded = torch.empty(
        (parameter_rows, alog_width), dtype=torch.float32, device=device
    )
    dBias = torch.empty(
        (parameter_rows, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=device
    )
    has_bias = dt_bias is not None
    if A_log is None:
        A_log = torch.empty((value_heads,), dtype=torch.float32, device=device)
    if dt_bias is None:
        dt_bias = torch.empty(
            (value_heads, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=device
        )
    if query_rstd is None:
        query_rstd = torch.empty(query.shape[:-1], dtype=torch.float32, device=device)
    if key_rstd is None:
        key_rstd = torch.empty(key.shape[:-1], dtype=torch.float32, device=device)

    def tensor(shape, dtype):
        if _workspace is not None:
            return _workspace.empty(shape, dtype)
        return torch.empty(shape, dtype=dtype, device=device)

    def f32_rows():
        return tensor((rows, SUPPORTED_HEAD_DIM), torch.float32)

    def b16_rows():
        return tensor((rows, SUPPORTED_HEAD_DIM), torch.bfloat16)

    def f32_matrix():
        return tensor((rows, CHUNK_SIZE), torch.float32)

    def b16_matrix():
        return tensor((rows, CHUNK_SIZE), torch.bfloat16)

    def f32_scalar():
        return tensor((rows,), torch.float32)

    stage_one_scratch = {
        "inverse": b16_matrix(),
        "values": b16_rows(),
        "keys": b16_rows(),
        "gamma": f32_rows(),
        "beta": f32_scalar(),
        "d_v_new_low": b16_rows(),
        "d_w_low": b16_rows(),
        "weighted_v": b16_rows(),
        "weighted_k": b16_rows(),
        "d_beta_v": f32_rows(),
        "d_beta_k": f32_rows(),
        "d_beta_wy": f32_scalar(),
        "d_lower_unmasked": f32_matrix(),
        "qg": b16_rows(),
        "kg": b16_rows(),
        "d_qg": f32_rows(),
        "d_kg": f32_rows(),
        "d_gate_last": tensor((scratch_chunks, SUPPORTED_HEAD_DIM), torch.float32),
        "d_query_base": f32_rows(),
        "d_key_base": f32_rows(),
        "d_gate_base": f32_rows(),
        "query": b16_rows(),
        "d_Aqk": f32_matrix(),
        "d_beta_matrix": f32_rows(),
        "d_query_pair": f32_rows(),
        "d_key_pair": f32_rows(),
        "d_gate_pair": f32_rows(),
        "d_beta_pair": f32_scalar(),
        "d_value": b16_rows(),
        "d_beta": tensor((rows,), beta_dtype),
    }
    source_arguments = (
        as_bnsd_view(query),
        as_bnsd_view(key),
        as_bnsd_view(values),
        as_bnsd_view(activated_beta, scalar=True),
        (saved_inverse.unsqueeze(0) if layout == "TND" else saved_inverse),
        as_bnsd_view(saved_qg),
        as_bnsd_view(saved_kg),
        as_bnsd_view(saved_gamma),
        as_bnsd_view(stage_two.d_v_new),
        as_bnsd_view(stage_two.d_w),
        as_bnsd_view(stage_two.d_qg),
        as_bnsd_view(stage_two.d_kg),
        stage_two.d_Aqk,
        stage_two.d_gate_last.unsqueeze(0).transpose(1, 2),
        metadata,
        as_bnsd_view(d_value),
        as_bnsd_view(d_beta, scalar=True),
        as_bnsd_view(raw_gate),
        A_log,
        dt_bias,
        as_bnsd_view(query_rstd, scalar=True),
        as_bnsd_view(key_rstd, scalar=True),
        as_bnsd_view(d_query),
        as_bnsd_view(d_key),
        as_bnsd_view(d_gate),
        dA_padded,
        dBias,
    )
    tensor_arguments = source_arguments + tuple(stage_one_scratch.values())
    dtype_map = {
        torch.bfloat16: dtypes.bfloat16,
        torch.float32: dtypes.float32,
        torch.int32: dtypes.int32,
    }
    specs = tuple(
        _backward_tensor_spec(item, dtype_map[item.dtype]) for item in tensor_arguments
    )
    config = (
        float(scale),
        fused_beta,
        float(inverse_factor),
        heads_per_key_head,
        fused_gate,
        has_bias,
        use_l2norm,
        lower_bound is None,
        0.0 if lower_bound is None else float(lower_bound),
        activated_beta.dtype == torch.float32,
        layout != "TND",
    )
    result = (
        d_query,
        d_key,
        d_value,
        d_gate,
        d_beta,
        dA_padded[:value_heads, 0] if fused_gate else None,
        dBias[:value_heads] if fused_gate and has_bias else None,
    )
    return result, tensor_arguments, specs, config


@dataclass
class _BackwardProgramEntry:
    launcher: object
    program: object
    pending_events: list


@dataclass(eq=False)
class _BackwardWorkspaceLease:
    event: object
    device: object
    arguments: Optional[tuple]
    stream: object = None
    event_recorded: bool = True
    error: Optional[Exception] = None


_BACKWARD_PROGRAMS = OrderedDict()


_BACKWARD_SCRATCH = WorkspacePool()


_BACKWARD_PROGRAM_LOCK = threading.RLock()


_BACKWARD_PROGRAM_LIMIT = 16


_BACKWARD_WORKSPACE_LEASES = OrderedDict()


_BACKWARD_RETIREMENT_QUEUE = None


_BACKWARD_RETIREMENT_THREAD = None


def _release_backward_workspace_lease(lease):
    # Queue entries keep this mutable owner, never another copy of its Tensor
    # tuple. Detach under the lock, then destroy on the owning device. Tensor
    # destruction can call the native allocator and must not hold the worker's
    # cache lock or fall back to a new thread's uninitialized device 0.
    with torch.npu.device(lease.device):
        with _BACKWARD_PROGRAM_LOCK:
            arguments = lease.arguments
            lease.arguments = None
            lease.error = None
            _BACKWARD_WORKSPACE_LEASES.pop(id(lease), None)
        del arguments


def _synchronize_backward_workspace_lease(lease):
    with torch.npu.device(lease.device):
        if lease.event_recorded:
            lease.event.synchronize()
        else:
            # A failed launch/event record has no reliable completion marker.
            # Explicit cache cleanup can retry its original stream boundary.
            lease.stream.synchronize()


def _retire_backward_workspace_lease(lease):
    try:
        _synchronize_backward_workspace_lease(lease)
    except Exception as error:
        # A failed wait does not prove the allocation is idle. Keep every
        # owner until explicit cleanup successfully synchronizes the call.
        with _BACKWARD_PROGRAM_LOCK:
            if lease.arguments is not None:
                lease.error = error
    else:
        _release_backward_workspace_lease(lease)


def _backward_workspace_retirement_worker(retirement_queue):
    while True:
        lease = retirement_queue.get()
        try:
            if lease is None:
                return
            # Device/stream state is thread local. Keep the worker bound after
            # the temporary event guard exits, including final lease/event
            # destruction below. A rank's device need not be logical device 0.
            torch.npu.set_device(lease.device)
            torch.npu.current_stream(lease.device)
            _retire_backward_workspace_lease(lease)
        except Exception as error:
            # Initialization failures must retain owners and reach the caller,
            # rather than silently terminating the retirement worker.
            with _BACKWARD_PROGRAM_LOCK:
                if lease.arguments is not None:
                    lease.error = error
        finally:
            retirement_queue.task_done()
            del lease


def _queue_backward_workspace_retirement(lease):
    global _BACKWARD_RETIREMENT_QUEUE, _BACKWARD_RETIREMENT_THREAD
    # Called under the cache lock after recording the completion event. No
    # thread or queue is created by importing this module or compiling code.
    if _BACKWARD_RETIREMENT_QUEUE is None:
        _BACKWARD_RETIREMENT_QUEUE = Queue()
    if (
        _BACKWARD_RETIREMENT_THREAD is None
        or not _BACKWARD_RETIREMENT_THREAD.is_alive()
    ):
        _BACKWARD_RETIREMENT_THREAD = threading.Thread(
            target=_backward_workspace_retirement_worker,
            args=(_BACKWARD_RETIREMENT_QUEUE,),
            name="kda_bwd_workspace_retire",
            daemon=True,
        )
        _BACKWARD_RETIREMENT_THREAD.start()
    _BACKWARD_RETIREMENT_QUEUE.put(lease)


def _stop_backward_workspace_retirement():
    global _BACKWARD_RETIREMENT_QUEUE, _BACKWARD_RETIREMENT_THREAD
    with _BACKWARD_PROGRAM_LOCK:
        worker = _BACKWARD_RETIREMENT_THREAD
        retirement_queue = _BACKWARD_RETIREMENT_QUEUE
        _BACKWARD_RETIREMENT_THREAD = None
        _BACKWARD_RETIREMENT_QUEUE = None
        if worker is not None:
            retirement_queue.put(None)
    # The worker needs the same lock to retire leases; never join under it.
    if worker is not None and worker is not threading.current_thread():
        worker.join()


def _reap_backward_workspaces():
    """Release completed owners before allocating another backward workspace."""
    with _BACKWARD_PROGRAM_LOCK:
        leases = tuple(_BACKWARD_WORKSPACE_LEASES.values())
    for lease in leases:
        if lease.arguments is None:
            continue
        if lease.error is not None:
            raise RuntimeError(
                "KDA backward workspace retirement failed; "
                "clear the backward program cache after resolving the NPU error"
            ) from lease.error
        if lease.event_recorded:
            with torch.npu.device(lease.device):
                completed = lease.event.query()
            if completed:
                _release_backward_workspace_lease(lease)


_BACKWARD_RAW_GATE_INDEX = 48


_BACKWARD_BETA_GRAD_INDEX = 47


def _backward_scalar_specs(config):
    return BackwardConfig.from_abi(config).scalar_specs(dtypes)


def _backward_runtime_config(config):
    return BackwardConfig.from_abi(config).runtime_values()


def backward_dynamic_compile_request(
    beta_fp32=True,
    use_full_chunks=False,
    needs_masked_chunks=True,
    gate_fp32=False,
    staged=False,
    beta_grad_fp32=False,
    cube_causal=False,
):
    """Shared AOT/runtime ABI, independent of concrete shapes and gate settings.

    Host preparation normalizes all public layouts to head-major views. The
    input, token-cache, matrix-cache and state-cache storage can have different
    capacities; keep those dimensions independent. Q/K inverse norms use the
    input layout after host preparation. Public/cache strides are runtime
    values, while internal scratch is contiguous.
    """
    dim = cannbotdsl.Dim
    fake = cannbotdsl.TensorSpec
    p, length = dim("P"), dim("L", min=0)
    batch, nq, nv = dim("B"), dim("NQK"), dim("NV")
    cache_p, cache_l = dim("CACHE_P"), dim("CACHE_L", min=0)
    beta_p, beta_l = dim("BETA_P"), dim("BETA_L", min=0)
    matrix_p, matrix_l = dim("MATRIX_P", min=0), dim("MATRIX_L", min=0)
    inverse_p, inverse_l = dim("INVERSE_P", min=0), dim("INVERSE_L", min=0)
    state_p, state_c = dim("STATE_P"), dim("STATE_C", min=0)
    output_p, output_l = dim("DO_P"), dim("DO_L", min=0)
    meta = dim("META", min=HEADER_WORDS)
    cores = dim("CORES", min=1, max=MAX_AIC_CORES)
    gradient_slots = dim("STAGED_SLOTS", min=1) if staged else cores
    core_rows = gradient_slots * CHUNK_SIZE
    masked_rows = dim(
        "MASKED_CORE_ROWS",
        min=0,
        max=MAX_AIC_CORES * CHUNK_SIZE,
        multiple_of=CHUNK_SIZE,
    )
    state_gradient_rows = (
        2 * gradient_slots * SUPPORTED_HEAD_DIM
        if staged
        else masked_rows * (2 * SUPPORTED_HEAD_DIM // CHUNK_SIZE)
    )
    scratch_chunks = dim("STAGE_ONE_SCRATCH_CHUNKS", min=0)
    rows = scratch_chunks * CHUNK_SIZE
    bf16, fp32 = dtypes.bfloat16, dtypes.float32

    def sequence(
        dtype, heads=nv, physical=cache_p, size=cache_l, width=SUPPORTED_HEAD_DIM
    ):
        return fake((physical, heads, size, width), dtype)

    cache = sequence(bf16)
    cache_f32 = sequence(fp32)
    compact_f32 = sequence(fp32, heads=1, physical=1, size=core_rows)
    gate_last = sequence(fp32, heads=1, physical=1, size=gradient_slots)
    raw_value = sequence(bf16, physical=p, size=length)
    qk = sequence(bf16, heads=nq, physical=p, size=length)
    rstd = sequence(fp32, heads=nq, physical=p, size=length, width=1)
    state = fake((batch, nv, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), fp32)
    metadata = fake((meta,), dtypes.int32)
    parameter_rows = dim("PARAMETER_ROWS")
    alog_width = 2 * GATE_SCALAR_TRANSFER_WIDTH
    chunk_matrix = fake((gradient_slots, 1, CHUNK_SIZE, CHUNK_SIZE), fp32)
    per_core_bf16 = fake((masked_rows, SUPPORTED_HEAD_DIM), bf16)
    per_core_f32 = fake((masked_rows, SUPPORTED_HEAD_DIM), fp32)
    per_core_state = fake(
        (masked_rows * (SUPPORTED_HEAD_DIM // CHUNK_SIZE), SUPPORTED_HEAD_DIM), bf16
    )
    specs = [
        sequence(bf16, physical=matrix_p, size=matrix_l, width=CHUNK_SIZE),
        cache,
        cache,
        cache,
        cache,
        cache_f32,
        fake((state_p, nv, state_c, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), bf16),
        sequence(bf16, physical=output_p, size=output_l),
        state,
        metadata,
        compact_f32,
        fake((core_rows, CHUNK_SIZE), fp32),
        compact_f32,
        compact_f32,
        compact_f32,
        gate_last,
        state,
        fake((state_gradient_rows, SUPPORTED_HEAD_DIM), fp32),
        fake((masked_rows, CHUNK_SIZE), bf16),
        per_core_bf16,
        per_core_bf16,
        per_core_bf16,
        per_core_f32,
        per_core_f32,
        per_core_f32,
        per_core_bf16,
        per_core_f32,
        per_core_f32,
        per_core_state,
        per_core_state,
        per_core_state,
        qk,
        qk,
        raw_value,
        sequence(fp32 if beta_fp32 else bf16, physical=beta_p, size=beta_l, width=1),
        sequence(bf16, physical=inverse_p, size=inverse_l, width=CHUNK_SIZE),
        cache,
        cache,
        cache_f32,
        compact_f32,
        compact_f32,
        compact_f32,
        compact_f32,
        chunk_matrix,
        gate_last,
        metadata,
        raw_value,
        sequence(fp32 if beta_grad_fp32 else bf16, physical=p, size=length, width=1),
        sequence(fp32 if gate_fp32 else bf16, physical=p, size=length),
        fake((nv,), fp32),
        fake((nv, SUPPORTED_HEAD_DIM), fp32),
        rstd,
        rstd,
        qk,
        qk,
        sequence(fp32 if gate_fp32 else bf16, physical=p, size=length),
        fake((parameter_rows, alog_width), fp32),
        fake((parameter_rows, SUPPORTED_HEAD_DIM), fp32),
    ]
    # The order mirrors _prepare_chunk_backward's stage_one_scratch dictionary.
    scratch = (
        (bf16, CHUNK_SIZE),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, None),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, None),
        (fp32, CHUNK_SIZE),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, "gate_last"),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, CHUNK_SIZE),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, None),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32 if beta_grad_fp32 else bf16, None),
    )
    for dtype, width in scratch:
        shape = (
            (scratch_chunks, SUPPORTED_HEAD_DIM)
            if width == "gate_last"
            else (rows,)
            if width is None
            else (rows, width)
        )
        specs.append(fake(shape, dtype))
    specs.append(fake((dim("OWNER_SCHEDULE_WORDS", min=1),), dtypes.int32))
    # Per-argument strides accept BNSD, BSND, TND and independently strided
    # forward caches, including the padded beta-gradient allocation.
    for index, spec in enumerate(specs):
        if index < 58:
            if len(spec.shape) > 2:
                stride = tuple(
                    dim(f"ARG{index}_S{axis}", min=0)
                    for axis in range(len(spec.shape) - 1)
                ) + (1,)
                specs[index] = fake(spec.shape, spec.dtype, stride=stride)
    if cube_causal and (not use_full_chunks or staged):
        raise ValueError("Cube causal requires persistent full chunks")
    if not use_full_chunks and not needs_masked_chunks:
        raise ValueError("at least one backward chunk path must be enabled")
    config = (0,) * 9 + (
        bool(beta_fp32),
        0,
        bool(use_full_chunks),
        bool(needs_masked_chunks),
        bool(staged),
        bool(cube_causal),
    )
    return tuple(specs), (
        dtypes.float32,
        *_backward_scalar_specs(config),
        dtypes.int64,
        dtypes.int64,
    )


def _cache_backward_program_abi(program):
    """Cache immutable ABI descriptions only on this backward program."""
    try:
        from cannbotdsl.core.compiler.function import ProviderCallable
        from cannbotdsl.core.compiler.parameter_lowering import ParameterLoweringPlan
    except ImportError:
        return program
    # This adaptation targets the inspected SDK plan layout. Other providers,
    # collection stubs and changed SDK layouts retain their original behavior.
    if type(program) is not ProviderCallable or program.closed or program._noop:
        return program
    plan = program._parameter_lowering
    fields = ("contract", "validation", "layout", "_host_ctypes")
    if (
        type(plan) is not ParameterLoweringPlan
        or tuple(getattr(ParameterLoweringPlan, "__slots__", ())) != fields
    ):
        return program

    class _BackwardCachedLowering(ParameterLoweringPlan):
        __slots__ = ("_cached_abi_arguments",)

        def __init__(self, original):
            for name in fields:
                object.__setattr__(self, name, getattr(original, name))
            object.__setattr__(self, "_cached_abi_arguments", original.abi_arguments)

        @property
        def abi_arguments(self):
            return self._cached_abi_arguments

    program._parameter_lowering = _BackwardCachedLowering(plan)
    return program


def _compile_backward_program(
    beta_fp32,
    use_full_chunks,
    needs_masked_chunks,
    gate_fp32=False,
    staged=False,
    beta_grad_fp32=False,
    cube_causal=False,
):
    """Use one compile factory for Native collection and runtime caching."""
    launcher = FlashKDAGrad()
    specs, scalar_specs = backward_dynamic_compile_request(
        beta_fp32,
        use_full_chunks,
        needs_masked_chunks,
        gate_fp32,
        staged,
        beta_grad_fp32,
        cube_causal,
    )
    program = cannbotdsl.compile(launcher.run, *specs, *scalar_specs)
    return _BackwardProgramEntry(launcher, _cache_backward_program_abi(program), [])


def _close_backward_program(entry):
    for event in entry.pending_events:
        event.synchronize()
    for lease in tuple(_BACKWARD_WORKSPACE_LEASES.values()):
        if any(lease.event is event for event in entry.pending_events):
            _release_backward_workspace_lease(lease)
    entry.pending_events.clear()
    entry.program.close()


def _clear_backward_program_cache():
    """Release compiled code only after its outstanding device calls finish."""
    with _BACKWARD_SCRATCH.lock:
        _BACKWARD_SCRATCH.clear()
        with _BACKWARD_PROGRAM_LOCK:
            for lease in tuple(_BACKWARD_WORKSPACE_LEASES.values()):
                _synchronize_backward_workspace_lease(lease)
                _release_backward_workspace_lease(lease)
            while _BACKWARD_PROGRAMS:
                key, entry = next(iter(_BACKWARD_PROGRAMS.items()))
                _close_backward_program(entry)
                del _BACKWARD_PROGRAMS[key]
    _stop_backward_workspace_retirement()


def _launch_backward_program(
    device,
    arguments,
    specs,
    config,
    scale,
    core_num,
    *,
    has_dht=True,
):
    # Cache code by configuration. Tensor owners belong only to the pending
    # completion lease and are retired as soon as that call's event finishes.
    config = BackwardConfig.from_abi(config)
    key = (
        cannbotdsl.compile,
        str(device),
        config.beta_fp32,
        bool(config.use_full_chunks),
        bool(config.needs_masked_chunks),
        bool(config.staged),
        specs[_BACKWARD_RAW_GATE_INDEX].dtype == dtypes.float32,
        specs[_BACKWARD_BETA_GRAD_INDEX].dtype == dtypes.float32,
        bool(config.cube_causal),
    )
    with torch.npu.device(device), _BACKWARD_PROGRAM_LOCK:
        entry = _BACKWARD_PROGRAMS.get(key)
        if entry is None:
            entry = _compile_backward_program(
                config.beta_fp32,
                config.use_full_chunks,
                config.needs_masked_chunks,
                key[-3],
                config.staged,
                key[-2],
                key[-1],
            )
            _BACKWARD_PROGRAMS[key] = entry
        _BACKWARD_PROGRAMS.move_to_end(key)
        stream = torch.npu.current_stream(torch.npu.current_device())
        for tensor in arguments:
            tensor.record_stream(stream)
        finished = torch.npu.Event()
        entry.pending_events[:] = [
            event for event in entry.pending_events if not event.query()
        ]
        lease = _BackwardWorkspaceLease(
            finished, device, arguments, stream, event_recorded=False
        )
        _BACKWARD_WORKSPACE_LEASES[id(lease)] = lease
        try:
            entry.program(
                *arguments,
                scale,
                *_backward_runtime_config(config),
                int(has_dht),
                core_num,
            )
            finished.record(stream)
            lease.event_recorded = True
            entry.pending_events.append(finished)
            _queue_backward_workspace_retirement(lease)
        except Exception:
            try:
                stream.synchronize()
            except Exception as error:
                lease.error = error
                raise
            _release_backward_workspace_lease(lease)
            raise
        while len(_BACKWARD_PROGRAMS) > _BACKWARD_PROGRAM_LIMIT:
            _, oldest = _BACKWARD_PROGRAMS.popitem(last=False)
            _close_backward_program(oldest)


atexit.register(_clear_backward_program_cache)


def _resolve_backward_metadata(q, v, layout, metadata, cu_seqlens, seqused):
    """Require caller-supplied metadata and check any explicit lengths."""
    if metadata is None:
        raise ValueError(
            "metadata is required: passing None is not supported yet; "
            "call flash_kda_grad_metadata first"
        )
    if cu_seqlens is None and seqused is None:
        return metadata
    if cu_seqlens is not None and layout != "TND":
        raise ValueError(
            "bwd cu_seqlens is only supported for TND; BNSD/BSND use seqused"
        )
    if seqused is not None and layout not in ("BNSD", "BSND"):
        raise ValueError(
            "bwd seqused is only supported for BNSD/BSND; TND uses cu_seqlens"
        )
    if not isinstance(q, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("q and v must be Tensors")
    if q.ndim != (3 if layout == "TND" else 4) or v.ndim != q.ndim:
        raise ValueError("q/v rank does not match layout")
    head_axis = 2 if layout == "BSND" else 1
    length_axis = 0 if layout == "TND" else 1 if layout == "BSND" else 2
    if (
        q.device.type != "npu"
        or v.device != q.device
        or any(
            t.device != q.device
            for t in (cu_seqlens, seqused)
            if isinstance(t, torch.Tensor)
        )
    ):
        raise ValueError("q/v and sequence lengths must be on the same NPU")
    if q.shape[length_axis] != v.shape[length_axis] or (
        layout != "TND" and q.shape[0] != v.shape[0]
    ):
        raise ValueError("q/v sequence dimensions do not match")
    if layout == "TND" and not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("cu_seqlens must be a Tensor")
    generated = _metadata.flash_kda_grad_metadata(
        cu_seqlens,
        seqused,
        batch_size=cu_seqlens.numel() - 1 if layout == "TND" else q.shape[0],
        n_heads_qk=q.shape[head_axis],
        n_heads_v=v.shape[head_axis],
        seqlens=q.shape[length_axis],
        headdim_qk=q.shape[-1],
        headdim_v=v.shape[-1],
        layout=layout,
    )
    # Reading the generated snapshot also checks device-side length validation.
    expected = _metadata.metadata_host_info(generated)
    if metadata is generated:
        return metadata
    if (
        not isinstance(metadata, torch.Tensor)
        or metadata.dtype != torch.int32
        or metadata.device != q.device
        or metadata.shape != generated.shape
        or not metadata.is_contiguous()
    ):
        raise ValueError("metadata must match the explicit sequence-length task ABI")
    wait_metadata_ready(metadata)
    actual = _metadata.metadata_host_info(metadata)
    # An all-full padded snapshot may use flags=0 instead of flags=1 while
    # describing exactly the same token ranges. Packed layout must still agree.
    if (
        actual.header[:HEADER_FLAGS] != expected.header[:HEADER_FLAGS]
        or (actual.header[HEADER_FLAGS] & 2) != (expected.header[HEADER_FLAGS] & 2)
        or actual.sequences != expected.sequences
    ):
        raise ValueError("metadata does not match cu_seqlens/seqused")
    return metadata


def _validate_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    chunk_states: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    v_new: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    g_cumsum: torch.Tensor,
    q_rstd: Optional[torch.Tensor],
    k_rstd: Optional[torch.Tensor],
    beta_sigmoid: Optional[torch.Tensor],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
    disable_recompute: bool = False,
    initial_state: Optional[torch.Tensor] = None,
):
    """Coordinate pure checks and metadata readiness in their original order."""
    inputs = BackwardInputs(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        chunk_states,
        Aqk,
        Akk,
        w,
        u,
        v_new,
        qg,
        kg,
        g_cumsum,
        q_rstd,
        k_rstd,
        beta_sigmoid,
        A_log,
        dt_bias,
        metadata,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        lower_bound,
        layout,
        disable_recompute,
        initial_state,
    )
    shape = _validate_input_structure(inputs)
    wait_metadata_ready(metadata)
    host_info = _metadata.metadata_host_info(metadata)
    _validate_saved_cache_contract(inputs, shape, host_info)
    return shape.batch, shape.value_heads


def _persistent_owner_plan(q, v, layout, metadata, core_num):
    axis = 2 if layout == "BSND" else 1
    return _metadata.backward_schedule(metadata, q.shape[axis], v.shape[axis], core_num)


def _owner_schedule_for_launch(q, v, layout, metadata, core_num, staged):
    axis = 2 if layout == "BSND" else 1
    return _metadata.owner_schedule_for_launch(
        metadata, q.shape[axis], v.shape[axis], core_num, staged
    )


def _select_backward_template(q, v, layout, metadata, core_num):
    return _persistent_owner_plan(q, v, layout, metadata, core_num).strategy


def _track_chunk_plan(q, v, layout, metadata, block_size, staged=False):
    axis = 2 if layout == "BSND" else 1
    return _metadata.backward_chunk_paths(
        metadata, q.shape[axis], v.shape[axis], block_size, staged
    )


def _forward_token_view(cache, layout):
    """Forward padded caches are head-major even for BSND inputs."""
    return cache.transpose(1, 2) if layout == "BSND" and cache is not None else cache


class _RecomputeBackward:
    @host
    def run(
        self,
        stage_two_gm_Aqk,
        stage_two_gm_qg,
        stage_two_gm_kg,
        stage_two_gm_w,
        stage_two_gm_v_new,
        stage_two_gm_g_cumsum,
        stage_two_gm_chunk_states,
        stage_two_gm_output_gradient,
        stage_two_gm_final_state_gradient,
        stage_two_gm_task_metadata,
        stage_two_gm_d_v_new,
        stage_two_gm_d_Aqk,
        stage_two_gm_d_qg,
        stage_two_gm_d_kg,
        stage_two_gm_d_w,
        stage_two_gm_d_gate_last,
        stage_two_gm_dh0,
        stage_two_gm_state_gradients,
        stage_two_gm_masked_aqk,
        stage_two_gm_masked_qg,
        stage_two_gm_masked_v_new,
        stage_two_gm_masked_output_gradient,
        stage_two_gm_d_v_new_output,
        stage_two_gm_d_kg_scratch,
        stage_two_gm_d_v_new_scratch,
        stage_two_gm_d_v_new_low,
        stage_two_gm_d_qg_scratch,
        stage_two_gm_d_w_scratch,
        stage_two_gm_state_high,
        stage_two_gm_state_residual,
        stage_two_gm_saved_state_high,
        stage_one_gm_effective_query,
        stage_one_gm_effective_key,
        stage_one_gm_value,
        stage_one_gm_activated_beta,
        stage_one_gm_Akk,
        stage_one_gm_saved_qg,
        stage_one_gm_saved_kg,
        stage_one_gm_g_cumsum,
        stage_one_gm_stage_two_d_v_new,
        stage_one_gm_stage_two_d_w,
        stage_one_gm_stage_two_d_qg,
        stage_one_gm_stage_two_d_kg,
        stage_one_gm_stage_two_d_Aqk,
        stage_one_gm_stage_two_d_gate_last,
        stage_one_gm_task_metadata,
        stage_one_gm_public_d_value,
        stage_one_gm_public_d_beta,
        stage_one_gm_raw_gate,
        stage_one_gm_A_log,
        stage_one_gm_dt_bias,
        stage_one_gm_q_rstd,
        stage_one_gm_k_rstd,
        stage_one_gm_public_d_query,
        stage_one_gm_public_d_key,
        stage_one_gm_public_d_gate,
        stage_one_gm_public_dA,
        stage_one_gm_public_dBias,
        stage_one_gm_inverse,
        stage_one_gm_values,
        stage_one_gm_keys,
        stage_one_gm_gamma,
        stage_one_gm_beta,
        stage_one_gm_d_v_new_low,
        stage_one_gm_d_w_low,
        stage_one_gm_weighted_v,
        stage_one_gm_weighted_k,
        stage_one_gm_d_beta_v,
        stage_one_gm_d_beta_k,
        stage_one_gm_d_beta_wy,
        stage_one_gm_d_lower_unmasked,
        stage_one_gm_qg,
        stage_one_gm_kg,
        stage_one_gm_d_qg,
        stage_one_gm_d_kg,
        stage_one_gm_d_gate_last,
        stage_one_gm_d_query_base,
        stage_one_gm_d_key_base,
        stage_one_gm_d_gate_base,
        stage_one_gm_query,
        stage_one_gm_d_Aqk,
        stage_one_gm_d_beta_matrix,
        stage_one_gm_d_query_pair,
        stage_one_gm_d_key_pair,
        stage_one_gm_d_gate_pair,
        stage_one_gm_d_beta_pair,
        stage_one_gm_d_value,
        stage_one_gm_d_beta,
        gm_owner_schedule,
        initial_state,
        recomputed_u,
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_fused_beta: dtypes.int64,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_heads_per_key_head: dtypes.int64,
        stage_one_fused_gate: dtypes.int64,
        stage_one_has_bias: dtypes.int64,
        stage_one_use_l2norm: dtypes.int64,
        stage_one_softplus: dtypes.int64,
        stage_one_lower_bound: dtypes.float32,
        stage_one_beta_fp32: Constexpr[bool],
        padded_layout: dtypes.int64,
        use_full_chunks: Constexpr[bool],
        needs_masked_chunks: Constexpr[bool],
        staged: Constexpr[bool],
        cube_causal: Constexpr[bool],
        has_initial_state: dtypes.int64,
        round_u_pre: dtypes.int64,
        has_dht: dtypes.int64,
        core_num: int,
    ):
        FlashKDAGradKernel()[core_num](
            stage_two_gm_Aqk,
            stage_two_gm_qg,
            stage_two_gm_kg,
            stage_two_gm_w,
            stage_two_gm_v_new,
            stage_two_gm_g_cumsum,
            stage_two_gm_chunk_states,
            stage_two_gm_output_gradient,
            stage_two_gm_final_state_gradient,
            stage_two_gm_task_metadata,
            stage_two_gm_d_v_new,
            stage_two_gm_d_Aqk,
            stage_two_gm_d_qg,
            stage_two_gm_d_kg,
            stage_two_gm_d_w,
            stage_two_gm_d_gate_last,
            stage_two_gm_dh0,
            stage_two_gm_state_gradients,
            stage_two_gm_masked_aqk,
            stage_two_gm_masked_qg,
            stage_two_gm_masked_v_new,
            stage_two_gm_masked_output_gradient,
            stage_two_gm_d_v_new_output,
            stage_two_gm_d_kg_scratch,
            stage_two_gm_d_v_new_scratch,
            stage_two_gm_d_v_new_low,
            stage_two_gm_d_qg_scratch,
            stage_two_gm_d_w_scratch,
            stage_two_gm_state_high,
            stage_two_gm_state_residual,
            stage_two_gm_saved_state_high,
            stage_one_gm_effective_query,
            stage_one_gm_effective_key,
            stage_one_gm_value,
            stage_one_gm_activated_beta,
            stage_one_gm_Akk,
            stage_one_gm_saved_qg,
            stage_one_gm_saved_kg,
            stage_one_gm_g_cumsum,
            stage_one_gm_stage_two_d_v_new,
            stage_one_gm_stage_two_d_w,
            stage_one_gm_stage_two_d_qg,
            stage_one_gm_stage_two_d_kg,
            stage_one_gm_stage_two_d_Aqk,
            stage_one_gm_stage_two_d_gate_last,
            stage_one_gm_task_metadata,
            stage_one_gm_public_d_value,
            stage_one_gm_public_d_beta,
            stage_one_gm_raw_gate,
            stage_one_gm_A_log,
            stage_one_gm_dt_bias,
            stage_one_gm_q_rstd,
            stage_one_gm_k_rstd,
            stage_one_gm_public_d_query,
            stage_one_gm_public_d_key,
            stage_one_gm_public_d_gate,
            stage_one_gm_public_dA,
            stage_one_gm_public_dBias,
            stage_one_gm_inverse,
            stage_one_gm_values,
            stage_one_gm_keys,
            stage_one_gm_gamma,
            stage_one_gm_beta,
            stage_one_gm_d_v_new_low,
            stage_one_gm_d_w_low,
            stage_one_gm_weighted_v,
            stage_one_gm_weighted_k,
            stage_one_gm_d_beta_v,
            stage_one_gm_d_beta_k,
            stage_one_gm_d_beta_wy,
            stage_one_gm_d_lower_unmasked,
            stage_one_gm_qg,
            stage_one_gm_kg,
            stage_one_gm_d_qg,
            stage_one_gm_d_kg,
            stage_one_gm_d_gate_last,
            stage_one_gm_d_query_base,
            stage_one_gm_d_key_base,
            stage_one_gm_d_gate_base,
            stage_one_gm_query,
            stage_one_gm_d_Aqk,
            stage_one_gm_d_beta_matrix,
            stage_one_gm_d_query_pair,
            stage_one_gm_d_key_pair,
            stage_one_gm_d_gate_pair,
            stage_one_gm_d_beta_pair,
            stage_one_gm_d_value,
            stage_one_gm_d_beta,
            gm_owner_schedule,
            initial_state,
            recomputed_u,
            stage_one_fused_beta,
            stage_one_heads_per_key_head,
            stage_one_fused_gate,
            stage_one_has_bias,
            stage_one_use_l2norm,
            stage_one_softplus,
            stage_one_beta_fp32,
            padded_layout,
            use_full_chunks,
            needs_masked_chunks,
            cube_causal,
            has_initial_state,
            round_u_pre,
            stage_two_scale,
            stage_one_scale,
            stage_one_inverse_factor,
            stage_one_lower_bound,
            has_dht,
            staged,
            True,
        )


def _recompute_dynamic_compile_request(
    beta_fp32,
    use_full_chunks=False,
    needs_masked_chunks=True,
    gate_fp32=False,
    staged=False,
    beta_grad_fp32=False,
    cube_causal=False,
):
    """Fused two-phase ABI; the saved-cache backward ABI stays separate."""
    specs, scalars = backward_dynamic_compile_request(
        beta_fp32,
        use_full_chunks,
        needs_masked_chunks,
        gate_fp32,
        staged,
        beta_grad_fp32,
        cube_causal,
    )
    # Recompute keeps a private FP32 state cache for recurrence.
    saved_state = specs[6]
    stride = tuple(cannbotdsl.Dim(f"ARG6_S{axis}", min=0) for axis in range(4)) + (1,)
    state_spec = cannbotdsl.TensorSpec(saved_state.shape, dtypes.float32, stride=stride)
    specs = (*specs[:6], state_spec, *specs[7:])
    # Initial state follows dht; the U output shares W's cache layout.
    u_spec = cannbotdsl.TensorSpec(
        specs[3].shape, dtypes.float32, stride=specs[3].stride
    )
    return (*specs, specs[8], u_spec), (
        *scalars[:-2],
        dtypes.int64,
        dtypes.int64,
        *scalars[-2:],
    )


_RECOMPUTE_PROGRAMS = OrderedDict()


_RECOMPUTE_LOCK = threading.RLock()


def _clear_recompute_program_cache():
    with _RECOMPUTE_LOCK, _BACKWARD_PROGRAM_LOCK:
        while _RECOMPUTE_PROGRAMS:
            key, entry = next(iter(_RECOMPUTE_PROGRAMS.items()))
            for lease in tuple(_BACKWARD_WORKSPACE_LEASES.values()):
                if any(
                    lease.event is pending or lease.stream is pending
                    for pending in entry.pending_events
                ):
                    _synchronize_backward_workspace_lease(lease)
                    _release_backward_workspace_lease(lease)
            _close_backward_program(entry)
            del _RECOMPUTE_PROGRAMS[key]


atexit.register(_clear_recompute_program_cache)


def _launch_recompute_program(
    device,
    arguments,
    config,
    scale,
    core_num,
    has_initial_state,
    round_u_pre=False,
    *,
    has_dht=True,
):
    config = BackwardConfig.from_abi(config)
    beta_fp32, full, masked, staged = (
        config.beta_fp32,
        config.use_full_chunks,
        config.needs_masked_chunks,
        config.staged,
    )
    cube_causal = bool(config.cube_causal)
    gate_fp32 = arguments[_BACKWARD_RAW_GATE_INDEX].dtype == torch.float32
    beta_grad_fp32 = arguments[_BACKWARD_BETA_GRAD_INDEX].dtype == torch.float32
    key = (
        cannbotdsl.compile,
        str(device),
        beta_fp32,
        bool(full),
        bool(masked),
        gate_fp32,
        bool(staged),
        beta_grad_fp32,
        cube_causal,
    )
    with torch.npu.device(device), _RECOMPUTE_LOCK, _BACKWARD_PROGRAM_LOCK:
        entry = _RECOMPUTE_PROGRAMS.get(key)
        if entry is None:
            launcher = _RecomputeBackward()
            specs, scalars = _recompute_dynamic_compile_request(
                beta_fp32,
                full,
                masked,
                gate_fp32,
                staged,
                beta_grad_fp32,
                cube_causal,
            )
            program = cannbotdsl.compile(launcher.run, *specs, *scalars)
            # 保留 Cube 编译变体，并缓存不可变 ABI 描述，避免重复下发时重新构造。
            entry = _BackwardProgramEntry(
                launcher, _cache_backward_program_abi(program), []
            )
            _RECOMPUTE_PROGRAMS[key] = entry
        stream = torch.npu.current_stream(device)
        for tensor in arguments:
            tensor.record_stream(stream)
        event = torch.npu.Event()
        lease = _BackwardWorkspaceLease(
            event, device, arguments, stream, event_recorded=False
        )
        _BACKWARD_WORKSPACE_LEASES[id(lease)] = lease
        try:
            entry.program(
                *arguments,
                float(scale),
                *_backward_runtime_config(config),
                int(has_initial_state),
                int(round_u_pre),
                int(has_dht),
                core_num,
            )
            event.record(stream)
            lease.event_recorded = True
            entry.pending_events[:] = [
                pending for pending in entry.pending_events if not pending.query()
            ]
            entry.pending_events.append(event)
            _queue_backward_workspace_retirement(lease)
        except Exception:
            try:
                stream.synchronize()
            except Exception as error:
                lease.error = error
                # No completion event can be trusted. Retain the program too;
                # cleanup must retry this stream before releasing device code.
                entry.pending_events.append(stream)
                raise
            _release_backward_workspace_lease(lease)
            raise


def _dma_gradient_view(tensor):
    """Use strided GM rows directly when their starts satisfy DMA alignment."""
    if tensor.is_contiguous():
        return tensor
    alignment = 32 // tensor.element_size()
    row_width = tensor.shape[-1]
    if (
        tensor.stride(-1) == 1
        and tensor.storage_offset() % alignment == 0
        and all(
            size <= 1 or (stride >= row_width and stride % alignment == 0)
            for size, stride in zip(tensor.shape[:-1], tensor.stride()[:-1])
        )
    ):
        return tensor
    return tensor.contiguous()


def _allocate_recompute_caches(values, final_state_gradient, layout):
    """Allocate recompute token caches in head-major GM storage."""
    batch, heads = final_state_gradient.shape[:2]
    length = (
        values.shape[0]
        if layout == "TND"
        else values.shape[1 if layout == "BSND" else 2]
    )
    width = values.shape[-1]
    storage_shape = (
        (heads, length, width) if layout == "TND" else (batch, heads, length, width)
    )
    caches = []
    for dtype in (torch.bfloat16,) * 5 + (torch.float32,):
        storage = values.new_empty(storage_shape, dtype=dtype)
        # Existing preparation expects token views in the input layout.
        if layout == "TND":
            storage = storage.transpose(0, 1)
        elif layout == "BSND":
            storage = storage.transpose(1, 2)
        caches.append(storage)
    u_cache = caches[1]
    caches[1] = torch.empty_strided(
        u_cache.shape, u_cache.stride(), dtype=torch.float32, device=u_cache.device
    )
    chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    state_prefix = (
        (chunks + batch, heads) if layout == "TND" else (batch, heads, chunks)
    )
    states = values.new_empty(
        (*state_prefix, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtype=torch.float32
    )
    return (*caches, states)


def _backward_variants():
    for staged in (False, True):
        paths = (
            ((False, True),) if staged else ((False, True), (True, False), (True, True))
        )
        # Fused beta is always FP32, independently of the original beta dtype.
        for beta_fp32, beta_grad_fp32 in ((False, False), (True, False), (True, True)):
            for gate_fp32 in (False, True):
                for full, masked in paths:
                    for cube_causal in (False, True) if full else (False,):
                        yield (
                            beta_fp32,
                            full,
                            masked,
                            gate_fp32,
                            staged,
                            beta_grad_fp32,
                            cube_causal,
                        )


def _recompute_variants():
    """Export only configurations the public recompute path can launch."""
    for variant in _backward_variants():
        _, full, masked, _, staged, _, _ = variant
        if (full, masked, staged) == _resolve_chunk_template(
            full, masked, staged, recompute=True
        ):
            yield variant


def _all_export_variants():
    return tuple(
        (recompute, *variant)
        for recompute in (False, True)
        for variant in (_recompute_variants() if recompute else _backward_variants())
    )


def _backward_export_shard():
    """Select an isolated Native build worker; ordinary exports remain complete."""
    import os

    value = os.environ.get("CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD", "0/1")
    try:
        index, count = map(int, value.split("/"))
    except ValueError as error:
        raise ValueError("backward export shard must be INDEX/COUNT") from error
    variants = len(_all_export_variants())
    if not 1 <= count <= variants or not 0 <= index < count:
        raise ValueError(
            f"backward export shard requires 0 <= INDEX < COUNT <= {variants}"
        )
    return index, count


def _disable_grad_at_call(fn):
    """Run ``fn`` with grad recording disabled, entering ``no_grad`` per call.

    The build-time torch stub used by host layout discovery only exposes
    symbolic declarations and has no ``no_grad`` attribute, so the context
    manager must not be created while this module is imported.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


@_disable_grad_at_call
def _flash_kda_grad_kernel_entry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rstd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
    *,
    cu_seqlens_cpu: Optional[list[int]] = None,
    dbeta_dtype: Optional[torch.dtype] = None,
):
    """Return (dq, dk, dv, dh0, dg, dbeta, dA, dBias).

    Positional arguments follow the backward-operator sheet in KDA.xls.
    Optional inputs and attributes also accept keyword arguments.
    disable_recompute=False rebuilds w/u/v_new/qg/kg/g_cumsum and
    FP32 chunk_states, ignoring those seven supplied caches. initial_state
    is the forward's FP32 [B,Nv,128,128] state; None means a zero state.
    False (default) rebuilds the caches to save device memory; True uses the
    separate saved-cache kernel and requires all seven caches.
    metadata is a required input today: it is declared optional in the schema so
    callers can omit it once metadata derivation lands, but passing None raises.
    Aqk/Akk are required Tensor arguments in both modes; conditional
    normalization/beta caches remain required when their fusion is enabled.
    g/beta independently accept BF16 or FP32. Recompute uses forward's BF16
    gate values in the activation and its derivative, and reads FP32 beta
    activations directly.
    Backward vector epilogues write dg/dbeta
    directly in the corresponding original g/beta dtype, rounding only for BF16.
    With Q/K L2Norm enabled, q/k are the saved normalized forward values;
    q_rstd/k_rstd are required FP32 inverse norms. The returned dq/dk are
    gradients of the original pre-normalization inputs.
    Host checks supplied sequence metadata, validates the cache ABI, allocates outputs
    and workspace, prepares forward-cache views, and
    launches one compute kernel. The host automatically selects persistent
    or staged scheduling. Recompute mode first rebuilds the forward
    tracks in that same kernel. Persistent keeps the same owners and uses a
    block-local handoff; staged waits for all cores before its four backward
    phases, giving five phases with recompute enabled.
    ``u`` remains in the forward-cache ABI; its gradient is represented by
    the Stage 2 ``d_v_new`` path and does not require the saved ``u`` values.
    Caches must use the current forward output layout. TND chunk storage
    starts at floor(token_start / 64) + sequence; padded caches are head-major
    and matrix caches retain their explicit chunk axis.
    Layout defaults to TND, fusion flags default to True, and omitted/None
    scale uses 1/sqrt(Dqk). With gate fusion enabled, lower_bound=None (default)
    uses -exp(A_log) * softplus(dt_bias + g); an explicit bound in [-5, 0)
    uses lower_bound * sigmoid(exp(A_log) * (dt_bias + g)).
    TND accepts int32 [B+1] ``cu_seqlens``; BNSD/BSND accept
    int32 [B] ``seqused``. metadata is a required int32 snapshot built by
    flash_kda_grad_metadata; any explicit lengths must agree with it.
    Absent ``do`` and ``dht`` are zero-initialized by the backward kernel.
    """
    if cu_seqlens_cpu is not None:
        raise ValueError("cu_seqlens_cpu must be None")
    scale = _validate_scale(scale)
    metadata = _resolve_backward_metadata(q, v, layout, metadata, cu_seqlens, seqused)
    state_dimensions = _validate_backward_inputs(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        chunk_states,
        Aqk,
        Akk,
        w,
        u,
        v_new,
        qg,
        kg,
        g_cumsum,
        q_rstd,
        k_rstd,
        beta_sigmoid,
        A_log,
        dt_bias,
        metadata,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        lower_bound,
        layout,
        disable_recompute,
        initial_state,
    )
    if dbeta_dtype not in (None, beta.dtype, torch.bfloat16):
        raise ValueError("dbeta_dtype must match beta.dtype or be BF16")
    has_dht = dht is not None
    do = _dma_gradient_view(do)
    if not has_dht:
        batch, value_heads = state_dimensions
        # The NPU initializes this backing storage in its existing prologue.
        dht = v.new_empty(
            (batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
            dtype=torch.float32,
        )
    else:
        dht = _dma_gradient_view(dht)
    if use_beta_sigmoid_in_kernel:
        beta_sigmoid = _forward_token_view(beta_sigmoid, layout)
    if use_qk_l2norm_in_kernel:
        q_rstd = _forward_token_view(q_rstd, layout)
        k_rstd = _forward_token_view(k_rstd, layout)
    if not disable_recompute:
        w, u, v_new, qg, kg, g_cumsum, chunk_states = _allocate_recompute_caches(
            v, dht, layout
        )
    else:
        w, u, v_new, qg, kg, g_cumsum = (
            _forward_token_view(tensor, layout)
            for tensor in (w, u, v_new, qg, kg, g_cumsum)
        )
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    _reap_backward_workspaces()
    core_num = _device_block_num(q)
    round_u_pre = not disable_recompute and getattr(
        metadata, "_flash_kda_round_recomputed_u_pre", False
    )
    head_axis = 2 if layout == "BSND" else 1
    plan = make_backward_plan(
        metadata,
        q.shape[head_axis],
        v.shape[head_axis],
        core_num,
        use_gate=use_gate_in_kernel,
        lower_bound=lower_bound,
        recompute=not disable_recompute,
    )
    staged = plan.staged
    needs_masked_chunks = plan.needs_masked_chunks
    selected_template = plan.template
    with _BACKWARD_SCRATCH.acquire(
        q, (core_num, layout, tuple(q.shape), selected_template)
    ) as workspace:
        stage_two, stage_two_arguments, stage_two_specs = _prepare_state_backward(
            aqk=Aqk,
            qg=qg,
            kg=kg,
            w=w,
            v_new=v_new,
            g_cumsum=g_cumsum,
            chunk_states=chunk_states,
            output_gradient=do,
            final_state_gradient=dht,
            layout=layout,
            metadata=metadata,
            scale=scale,
            core_num=core_num,
            _workspace=workspace,
            _needs_masked_chunks=needs_masked_chunks,
            _staged=staged,
            _gradient_slots=plan.gradient_slots,
        )
        stage_one_result, stage_one_arguments, stage_one_specs, stage_one_config = (
            _prepare_chunk_backward(
                query=q,
                key=k,
                values=v,
                raw_gate=g,
                activated_beta=beta_sigmoid if use_beta_sigmoid_in_kernel else beta,
                saved_inverse=Akk,
                saved_qg=qg,
                saved_kg=kg,
                saved_gamma=g_cumsum,
                A_log=A_log,
                dt_bias=dt_bias,
                query_rstd=q_rstd if use_qk_l2norm_in_kernel else None,
                key_rstd=k_rstd if use_qk_l2norm_in_kernel else None,
                stage_two=stage_two,
                layout=layout,
                metadata=metadata,
                scale=scale,
                fused_beta=use_beta_sigmoid_in_kernel,
                fused_gate=use_gate_in_kernel,
                use_l2norm=use_qk_l2norm_in_kernel,
                inverse_factor=0.5 if allow_neg_eigval else 1.0,
                lower_bound=lower_bound,
                beta_dtype=beta.dtype if dbeta_dtype is None else dbeta_dtype,
                _workspace=workspace,
            )
        )
        owner_schedule = _owner_schedule_for_launch(
            q, v, layout, metadata, core_num, staged
        )
        arguments = (*stage_two_arguments, *stage_one_arguments, owner_schedule)
        config = BackwardConfig(
            *stage_one_config,
            plan.use_full_chunks,
            plan.needs_masked_chunks,
            plan.staged,
            plan.cube_causal,
        )
        if disable_recompute:
            _launch_backward_program(
                q.device,
                arguments,
                (
                    *stage_two_specs,
                    *stage_one_specs,
                    _backward_tensor_spec(owner_schedule, dtypes.int32),
                ),
                config,
                float(scale),
                core_num,
                has_dht=has_dht,
            )
        else:
            u_view = (
                u.unsqueeze(0).transpose(1, 2)
                if layout == "TND"
                else _forward_token_view(u, layout)
            )
            _launch_recompute_program(
                q.device,
                (*arguments, dht if initial_state is None else initial_state, u_view),
                config,
                float(scale),
                core_num,
                initial_state is not None,
                round_u_pre,
                has_dht=has_dht,
            )
        dq, dk, dv, dg, dbeta, dA, dBias = stage_one_result
        return dq, dk, dv, stage_two.dh0, dg, dbeta, dA, dBias


@_disable_grad_at_call
def flash_kda_grad_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rstd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
    *,
    cu_seqlens_cpu: Optional[list[int]] = None,
    dbeta_dtype: Optional[torch.dtype] = None,
):
    """Run backward using the public FlashKDA gradient tensor layouts."""
    if cu_seqlens_cpu is not None:
        raise ValueError("cu_seqlens_cpu must be None")
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError("layout must be TND, BNSD, or BSND")
    gate_dtype = torch.bfloat16 if use_gate_in_kernel else torch.float32
    beta_dtype = torch.bfloat16 if use_beta_sigmoid_in_kernel else torch.float32
    if g.dtype != gate_dtype:
        raise ValueError(
            f"g must have dtype {gate_dtype} for use_gate_in_kernel={use_gate_in_kernel}"
        )
    if beta.dtype != beta_dtype:
        raise ValueError(
            f"beta must have dtype {beta_dtype} for use_beta_sigmoid_in_kernel={use_beta_sigmoid_in_kernel}"
        )

    if q.ndim != (3 if layout == "TND" else 4) or v.ndim != q.ndim:
        raise ValueError("q and v rank must match layout")
    if layout == "TND":
        length, heads, dim = v.shape
        gate_shape = (length, heads, dim)
        cache_shape = (heads, length, dim)
        norm_shape = (q.shape[1], length)
    else:
        batch = v.shape[0]
        heads, length = v.shape[1:3] if layout == "BNSD" else v.shape[1:3][::-1]
        dim = v.shape[-1]
        gate_shape = (batch, length, heads, dim)
        cache_shape = (batch, heads, length, dim)
        norm_shape = (batch, q.shape[1 if layout == "BNSD" else 2], length)
    matrix_shape = (*cache_shape[:-1], CHUNK_SIZE)
    if Aqk.shape != matrix_shape or Akk.shape != matrix_shape:
        raise ValueError(f"Aqk/Akk must have shape {matrix_shape} for {layout}")
    if chunk_states is not None:
        if layout == "TND":
            metadata = _resolve_backward_metadata(
                q, v, layout, metadata, cu_seqlens, seqused
            )
            batch = (metadata.numel() - HEADER_WORDS) // (TASK_WORDS * heads)
            state_prefix = (
                (length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch,
                heads,
            )
        else:
            state_prefix = (batch, (length + CHUNK_SIZE - 1) // CHUNK_SIZE, heads)
        if chunk_states.dtype != torch.bfloat16 or chunk_states.shape != (
            *state_prefix,
            dim,
            dim,
        ):
            raise ValueError("chunk_states must use the public BF16 chunk layout")
    for name, tensor, shape in (
        ("g", g, gate_shape),
        ("beta", beta, gate_shape[:-1]),
        ("w", w, cache_shape),
        ("u", u, cache_shape),
        ("qg", qg, cache_shape),
        ("kg", kg, cache_shape),
        ("v_new", v_new, cache_shape),
        ("g_cumsum", g_cumsum, cache_shape),
        ("q_rstd", q_rstd, norm_shape),
        ("k_rstd", k_rstd, norm_shape),
        ("beta_sigmoid", beta_sigmoid, cache_shape[:-1]),
    ):
        if tensor is not None and tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape} for {layout}")

    if not isinstance(do, torch.Tensor):
        raise ValueError("do must be a Tensor")
    if do.shape != gate_shape:
        raise ValueError(f"do must have shape {gate_shape} for {layout}")
    bias_is_flat = dt_bias is not None and dt_bias.ndim == 1
    if bias_is_flat:
        if dt_bias.numel() != heads * dim:
            raise ValueError("flat dt_bias must have Nv * Dk elements")
        dt_bias = dt_bias.view(heads, dim)
    if layout == "BNSD":
        do = do.transpose(1, 2)
        g, beta = g.transpose(1, 2), beta.transpose(1, 2)

    def kernel_cache(tensor):
        if tensor is None:
            return None
        return tensor.transpose(0, 1) if layout == "TND" else tensor

    def kernel_chunks(tensor):
        if tensor is None:
            return None
        return tensor if layout == "TND" else tensor.transpose(1, 2)

    dq, dk, dv, dh0, dg, dbeta, da, dbias = _flash_kda_grad_kernel_entry(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk,
        Akk,
        metadata,
        initial_state=initial_state,
        chunk_states=kernel_chunks(chunk_states),
        w=kernel_cache(w),
        u=kernel_cache(u),
        qg=kernel_cache(qg),
        kg=kernel_cache(kg),
        v_new=kernel_cache(v_new),
        g_cumsum=kernel_cache(g_cumsum),
        q_rstd=kernel_cache(q_rstd),
        k_rstd=kernel_cache(k_rstd),
        beta_sigmoid=kernel_cache(beta_sigmoid),
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
        dbeta_dtype=dbeta_dtype,
    )
    if layout == "BNSD":
        dg, dbeta = dg.transpose(1, 2), dbeta.transpose(1, 2)
    if bias_is_flat and dbias is not None:
        dbias = dbias.reshape(-1)
    return dq, dk, dv, dh0, dg, dbeta, da, dbias


def flash_kda_grad_from_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    forward_outputs: tuple[torch.Tensor | None, ...],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    metadata: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    disable_recompute: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states_fp32: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
):
    """Run backward with the public 16-slot training-forward result.

    A legacy FP32 state override is rounded to the BF16 saved-cache ABI.
    """
    if not isinstance(do, torch.Tensor):
        raise ValueError("do must be a Tensor")
    _validate_forward_outputs(forward_outputs)
    metadata = _resolve_backward_metadata(q, v, layout, metadata, cu_seqlens, seqused)
    (
        _,
        _,
        chunk_states,
        Aqk,
        Akk,
        w,
        u,
        qg,
        kg,
        v_new,
        g_cumsum,
        q_norm,
        q_rstd,
        k_norm,
        k_rstd,
        beta_activated,
    ) = forward_outputs
    if chunk_states_fp32 is not None:
        chunk_states = chunk_states_fp32.bfloat16()
    if use_qk_l2norm_in_kernel:
        if (
            q_norm is None
            or k_norm is None
            or q_norm.shape != q.shape
            or k_norm.shape != k.shape
        ):
            raise ValueError("normalized Q/K must match the input shapes")
        q, k = q_norm, k_norm
    return flash_kda_grad_npu(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk,
        Akk,
        metadata,
        initial_state=initial_state,
        chunk_states=chunk_states,
        w=w,
        u=u,
        v_new=v_new,
        qg=qg,
        kg=kg,
        g_cumsum=g_cumsum,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_activated,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
        layout=layout,
        disable_recompute=disable_recompute,
    )


@aot.compile_cache
def _compile_export_variant(
    recompute,
    beta_fp32,
    use_full_chunks,
    needs_masked_chunks,
    gate_fp32,
    staged,
    beta_grad_fp32,
    cube_causal,
):
    request = (
        _recompute_dynamic_compile_request
        if recompute
        else backward_dynamic_compile_request
    )
    launcher = _RecomputeBackward() if recompute else FlashKDAGrad()
    specs, scalars = request(
        beta_fp32,
        use_full_chunks,
        needs_masked_chunks,
        gate_fp32,
        staged,
        beta_grad_fp32,
        cube_causal,
    )
    return cannbotdsl.compile(launcher.run, *specs, *scalars)


def _export_variant_calls():
    """Enumerate both cache modes in the original shard order."""
    shard_index, shard_count = _backward_export_shard()
    return _all_export_variants()[shard_index::shard_count]


@aot.export("flash_kda_grad")
def export_flash_kda_grad():
    """Export deterministic kernels for both templates and cache modes."""
    for variant in _export_variant_calls():
        _compile_export_variant(*variant)


__all__ = ["flash_kda_grad_npu", "flash_kda_grad_from_forward"]
