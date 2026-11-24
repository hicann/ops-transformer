# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Pure Host input and saved-cache contract checks; consumes metadata snapshots."""

from __future__ import annotations
import math
import torch
from typing import NamedTuple, Optional

if __package__:
    from .flash_kda_grad_metadata import (
        ABI_VERSION,
        CHUNK_SIZE,
        HEADER_ABI_VERSION,
        HEADER_BATCH,
        HEADER_CHUNK_COUNT,
        HEADER_FLAGS,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_STORAGE_LENGTH,
        HEADER_VALUE_HEADS,
        HEADER_WORDS,
        MAGIC,
        STATUS_OK,
        SUPPORTED_HEAD_DIM,
        TASK_WORDS,
        metadata_capacity,
    )
else:
    from flash_kda_grad_metadata import (
        ABI_VERSION,
        CHUNK_SIZE,
        HEADER_ABI_VERSION,
        HEADER_BATCH,
        HEADER_CHUNK_COUNT,
        HEADER_FLAGS,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_STORAGE_LENGTH,
        HEADER_VALUE_HEADS,
        HEADER_WORDS,
        MAGIC,
        STATUS_OK,
        SUPPORTED_HEAD_DIM,
        TASK_WORDS,
        metadata_capacity,
    )


def _validate_scale(scale):
    """Normalize an optional scale before any metadata or device work."""
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("scale must be a finite number or None") from error
        if not math.isfinite(scale):
            raise ValueError("scale must be a finite number or None")
    return scale


class BackwardInputs(NamedTuple):
    """Public tensors and options, without device execution or synchronization."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    do: torch.Tensor
    dht: Optional[torch.Tensor]
    chunk_states: torch.Tensor
    Aqk: torch.Tensor
    Akk: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    v_new: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    g_cumsum: torch.Tensor
    q_rstd: Optional[torch.Tensor]
    k_rstd: Optional[torch.Tensor]
    beta_sigmoid: Optional[torch.Tensor]
    A_log: Optional[torch.Tensor]
    dt_bias: Optional[torch.Tensor]
    metadata: Optional[torch.Tensor] = None
    use_qk_l2norm_in_kernel: bool = True
    use_gate_in_kernel: bool = True
    use_beta_sigmoid_in_kernel: bool = True
    allow_neg_eigval: bool = False
    lower_bound: Optional[float] = None
    layout: str = "TND"
    disable_recompute: bool = False
    initial_state: Optional[torch.Tensor] = None


class BackwardInputShape(NamedTuple):
    batch: int
    value_heads: int
    qk_heads: int
    storage_length: int


def _validate_input_structure(inputs):
    if not isinstance(inputs.do, torch.Tensor):
        raise ValueError("do must be a Tensor")
    required = (
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.Aqk,
        inputs.Akk,
    )
    if inputs.disable_recompute:
        required += (
            inputs.chunk_states,
            inputs.w,
            inputs.u,
            inputs.v_new,
            inputs.qg,
            inputs.kg,
            inputs.g_cumsum,
        )
    if any((t is None for t in required)):
        raise ValueError("backward requires all saved forward caches")
    if any((not isinstance(t, torch.Tensor) for t in required)):
        raise ValueError("backward inputs and required saved caches must be Tensors")
    if inputs.q.device.type != "npu" or any(
        (t.device != inputs.q.device for t in required)
    ):
        raise ValueError("inputs and caches must be on one NPU")
    if any(
        (
            t.dtype != torch.bfloat16
            for t in (
                (
                    inputs.q,
                    inputs.k,
                    inputs.v,
                    inputs.Aqk,
                    inputs.Akk,
                    inputs.w,
                    inputs.u,
                    inputs.v_new,
                    inputs.qg,
                    inputs.kg,
                )
                if inputs.disable_recompute
                else (inputs.q, inputs.k, inputs.v, inputs.Aqk, inputs.Akk)
            )
        )
    ):
        raise ValueError(
            "the first specialization requires BF16 token and matrix planes"
        )
    if any(
        (
            t.dtype not in (torch.bfloat16, torch.float32)
            for t in (inputs.g, inputs.beta)
        )
    ):
        raise ValueError("g and beta must be BF16 or FP32")
    if inputs.disable_recompute and (
        inputs.chunk_states.dtype != torch.bfloat16
        or inputs.g_cumsum.dtype != torch.float32
    ):
        raise ValueError("chunk_states must be BF16 and g_cumsum must be FP32")
    if inputs.layout not in ("TND", "BNSD", "BSND"):
        raise ValueError("layout must be TND, BNSD, or BSND")
    if (
        inputs.q.ndim != (3 if inputs.layout == "TND" else 4)
        or inputs.q.shape != inputs.k.shape
        or inputs.q.shape[-1] != SUPPORTED_HEAD_DIM
    ):
        raise ValueError("Q/K layout or width is invalid")
    if inputs.v.ndim != inputs.q.ndim:
        raise ValueError("V rank must match Q")
    value_heads = (
        inputs.v.shape[1] if inputs.layout in ("TND", "BNSD") else inputs.v.shape[2]
    )
    qk_heads = (
        inputs.q.shape[1] if inputs.layout in ("TND", "BNSD") else inputs.q.shape[2]
    )
    if value_heads < 1 or qk_heads < 1 or value_heads % qk_heads:
        raise ValueError("value heads must be divisible by QK heads")
    if inputs.layout == "TND":
        value_shape = (inputs.q.shape[0], value_heads, SUPPORTED_HEAD_DIM)
        storage_length = inputs.q.shape[0]
    elif inputs.layout == "BNSD":
        value_shape = (
            inputs.q.shape[0],
            value_heads,
            inputs.q.shape[2],
            SUPPORTED_HEAD_DIM,
        )
        storage_length = inputs.q.shape[2]
    else:
        value_shape = (
            inputs.q.shape[0],
            inputs.q.shape[1],
            value_heads,
            SUPPORTED_HEAD_DIM,
        )
        storage_length = inputs.q.shape[1]
    if (
        inputs.v.shape != value_shape
        or inputs.g.shape != value_shape
        or inputs.beta.shape != value_shape[:-1]
    ):
        raise ValueError("V/g/beta physical shapes do not match Q")
    if (
        inputs.do.shape != inputs.v.shape
        or inputs.do.dtype != torch.bfloat16
        or inputs.do.device != inputs.q.device
    ):
        raise ValueError("do must be a BF16 Tensor matching v")
    if (
        not isinstance(inputs.metadata, torch.Tensor)
        or inputs.metadata.ndim != 1
        or inputs.metadata.dtype != torch.int32
        or (inputs.metadata.device != inputs.q.device)
        or (not inputs.metadata.is_contiguous())
        or (inputs.metadata.numel() - HEADER_WORDS) % (TASK_WORDS * value_heads)
    ):
        raise ValueError("metadata must use the backward int32 task ABI")
    batch = (inputs.metadata.numel() - HEADER_WORDS) // (TASK_WORDS * value_heads)
    if batch < 1 or inputs.metadata.numel() != metadata_capacity(batch, value_heads):
        raise ValueError("metadata task count is invalid")
    if inputs.layout != "TND" and batch != inputs.q.shape[0]:
        raise ValueError("metadata batch does not match padded inputs")
    return BackwardInputShape(batch, value_heads, qk_heads, storage_length)


def _validate_saved_cache_contract(inputs, shape, host_info):
    batch, value_heads, qk_heads, storage_length = shape
    metadata_header = host_info.header
    if (
        metadata_header[HEADER_MAGIC] != MAGIC
        or metadata_header[HEADER_ABI_VERSION] != ABI_VERSION
    ):
        raise ValueError("backward metadata ABI does not match")
    if metadata_header[HEADER_STATUS] != STATUS_OK:
        raise ValueError(f"backward metadata status {metadata_header[HEADER_STATUS]}")
    metadata_flags = metadata_header[HEADER_FLAGS]
    if (
        metadata_header[HEADER_BATCH] != batch
        or metadata_header[HEADER_VALUE_HEADS] != value_heads
        or metadata_header[HEADER_STORAGE_LENGTH] != storage_length
        or (metadata_flags not in (0, 1, 3))
        or (bool(metadata_flags & 2) != (inputs.layout == "TND"))
    ):
        raise ValueError("backward metadata input shape or layout does not match")
    chunk_count = metadata_header[HEADER_CHUNK_COUNT]
    if chunk_count < 0 or chunk_count > batch * (
        (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    ):
        raise ValueError("backward metadata chunk count is invalid")
    if inputs.layout == "TND":
        cache_shape = (storage_length, value_heads, SUPPORTED_HEAD_DIM)
        qk_scalar_shape = (storage_length, qk_heads)
        capacity = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch
        matrix_shape = (value_heads, storage_length, CHUNK_SIZE)
        state_shape = (capacity, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
    else:
        cache_shape = (batch, value_heads, storage_length, SUPPORTED_HEAD_DIM)
        qk_scalar_shape = (batch, qk_heads, storage_length)
        chunks = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
        matrix_shape = (batch, value_heads, storage_length, CHUNK_SIZE)
        state_shape = (
            batch,
            value_heads,
            chunks,
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
    if inputs.disable_recompute:
        if inputs.qg.shape != cache_shape:
            raise ValueError("qg must use the current forward cache shape")
        if any(
            (
                t.shape != cache_shape
                for t in (inputs.kg, inputs.w, inputs.u, inputs.v_new, inputs.g_cumsum)
            )
        ):
            raise ValueError("saved token caches must use the current forward layout")
    if inputs.Aqk.shape != matrix_shape or inputs.Akk.shape != matrix_shape:
        raise ValueError("Aqk/Akk must use the token-row matrix cache shape")
    if inputs.disable_recompute and inputs.chunk_states.shape != state_shape:
        raise ValueError("BF16 chunk_states shape must match the current forward cache")
    if inputs.dht is not None and (
        not isinstance(inputs.dht, torch.Tensor)
        or inputs.dht.dtype != torch.float32
        or inputs.dht.device != inputs.q.device
        or inputs.dht.shape
        != (batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
    ):
        raise ValueError(
            f"dht must be FP32 [sequence, value_head, {SUPPORTED_HEAD_DIM}, {SUPPORTED_HEAD_DIM}]"
        )
    if not inputs.disable_recompute and inputs.initial_state is not None:
        if (
            not isinstance(inputs.initial_state, torch.Tensor)
            or inputs.initial_state.dtype != torch.float32
            or inputs.initial_state.device != inputs.q.device
            or (
                inputs.initial_state.shape
                != (batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
            )
            or (inputs.initial_state.stride(-1) != 1)
        ):
            raise ValueError(
                "initial_state must be FP32 [sequence, value_head, 128, 128] on the input device with contiguous rows"
            )
    if inputs.use_qk_l2norm_in_kernel:
        if (
            not isinstance(inputs.q_rstd, torch.Tensor)
            or not isinstance(inputs.k_rstd, torch.Tensor)
            or inputs.q_rstd.shape != qk_scalar_shape
            or (inputs.k_rstd.shape != qk_scalar_shape)
            or (inputs.q_rstd.dtype != torch.float32)
            or (inputs.k_rstd.dtype != torch.float32)
            or any(
                (t.device != inputs.q.device for t in (inputs.q_rstd, inputs.k_rstd))
            )
        ):
            raise ValueError(
                "use_qk_l2norm_in_kernel requires valid FP32 q_rstd and k_rstd caches"
            )
    if inputs.use_beta_sigmoid_in_kernel:
        if (
            not isinstance(inputs.beta_sigmoid, torch.Tensor)
            or inputs.beta_sigmoid.shape != cache_shape[:-1]
            or inputs.beta_sigmoid.dtype != torch.float32
            or (inputs.beta_sigmoid.device != inputs.q.device)
        ):
            raise ValueError("activated beta cache must be FP32")
    if inputs.use_gate_in_kernel:
        if (
            not isinstance(inputs.A_log, torch.Tensor)
            or inputs.A_log.shape != (value_heads,)
            or inputs.A_log.dtype != torch.float32
            or (inputs.A_log.device != inputs.q.device)
        ):
            raise ValueError("A_log must be FP32 [value_heads]")
    if inputs.dt_bias is not None and (
        not isinstance(inputs.dt_bias, torch.Tensor)
        or inputs.dt_bias.shape != (value_heads, SUPPORTED_HEAD_DIM)
        or inputs.dt_bias.dtype != torch.float32
        or (inputs.dt_bias.device != inputs.q.device)
    ):
        raise ValueError(f"dt_bias must be FP32 [value_heads, {SUPPORTED_HEAD_DIM}]")
    if inputs.allow_neg_eigval and (not inputs.use_beta_sigmoid_in_kernel):
        raise ValueError("allow_neg_eigval requires activated beta")
    if inputs.lower_bound is not None and (not -5.0 <= inputs.lower_bound < 0.0):
        raise ValueError("lower_bound must be None or in [-5, 0)")


def _validate_state_backward_inputs(
    *, aqk, qg, kg, w, v_new, g_cumsum, chunk_states, layout, metadata
):
    """Check physical cache views before allocating state workspaces."""
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError("layout must be TND, BNSD, or BSND")
    is_tnd = layout == "TND"
    if is_tnd:
        if qg.shape[2] != SUPPORTED_HEAD_DIM:
            raise ValueError("TND qg width is invalid")
        heads = qg.shape[1]
    else:
        heads = qg.shape[1] if layout == "BNSD" else qg.shape[2]
        if qg.shape[-1] != SUPPORTED_HEAD_DIM or qg.ndim != 4:
            raise ValueError("qg physical cache shape is invalid")
    if (
        metadata.ndim != 1
        or metadata.dtype != torch.int32
        or metadata.device != qg.device
        or not metadata.is_contiguous()
        or (metadata.numel() - HEADER_WORDS) % (TASK_WORDS * heads)
    ):
        raise ValueError("backward metadata ABI is invalid")
    batch = (metadata.numel() - HEADER_WORDS) // (TASK_WORDS * heads)
    if batch < 1 or metadata.numel() != metadata_capacity(batch, heads):
        raise ValueError("backward metadata task count is invalid")
    if not is_tnd and qg.shape[0] != batch:
        raise ValueError("physical qg batch does not match metadata")
    physical_matrix_shape = (
        batch,
        heads,
        qg.shape[2 if layout == "BNSD" else 1],
        CHUNK_SIZE,
    )
    if not is_tnd and tuple(aqk.shape) != physical_matrix_shape:
        raise ValueError("padded Aqk view must match the token rows")
    if not is_tnd and not (
        chunk_states.ndim == 5
        and chunk_states.shape[0] == batch
        and chunk_states.shape[1] == heads
        and chunk_states.shape[3:] == (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
    ):
        raise ValueError("padded chunk_states must match the batch and heads")
    if any(t.shape != qg.shape for t in (kg, w, v_new, g_cumsum)):
        raise ValueError("token caches do not match qg")
    return batch, heads


def _validate_forward_qk_cache(cache, reference, layout):
    expected = (
        (reference.shape[0], reference.shape[2], reference.shape[1], reference.shape[3])
        if layout == "BSND"
        else reference.shape
    )
    if (
        not isinstance(cache, torch.Tensor)
        or cache.dtype != torch.bfloat16
        or cache.device != reference.device
        or cache.shape != expected
    ):
        raise ValueError("saved Q/K cache must use the current forward layout")


def _validate_chunk_heads(qk_heads, value_heads):
    if qk_heads < 1 or value_heads % qk_heads:
        raise ValueError("effective Q/K heads do not match value heads")


def _validate_forward_outputs(forward_outputs):
    if len(forward_outputs) != 16:
        raise ValueError("training forward must return 16 tensors/cache slots")
