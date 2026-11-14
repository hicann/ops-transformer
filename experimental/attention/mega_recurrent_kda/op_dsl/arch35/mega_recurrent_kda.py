# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mega KDA 公开入口、TP 调度、内核下发与导出。"""

from __future__ import annotations

import cannbotdsl
import ctypes
import functools
import math
import sys
import threading
import torch
from cannbotdsl import MemLoc, PIPE, Tensor, TensorSpec, dtypes, make_layout
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    global_sync_all,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import ceil_div, from_tensor, tile_slice
from types import ModuleType
from typing import NamedTuple

if __package__:
    from .mega_recurrent_kda_backend import (
        KDA_STATE_KEY_UNROLL,
        KDA_STATE_UPDATE_UNROLL,
        GridScheduleBackendOperations,
        OutputGateStage,
        OutputProjectionStage,
        RecurrentKDAStage,
        RecurrentStage,
        S4PipelinedBackendSchedule,
        SequenceRecurrentStage,
        TP16S4BackendSchedule,
        TP4BackendSchedule,
        TP4HeadPanelBackend,
        TP4S4BackendSchedule,
        TP4TPScheduleBackendOperations,
        TP8BackendSchedule,
        TP8HeadPanelBackend,
        TP8TPScheduleBackendOperations,
        _DelayedGateProjection,
        _HeadStreamOutput,
        _RawGateContext,
        _RawRecurrentContext,
        _TiledOutputProjectionStage,
        _head_stream_persistent_raw_round,
        _run_raw_recurrent_item,
        _store_raw_gated_item,
        publish_batch_group_vector_grid_to_cube,
        publish_vector_grid_to_cube,
        run_recurrent_wave,
        run_s4_raw_item,
    )
    from .mega_recurrent_kda_common import (
        BETA_TILE,
        BATCH_TILE,
        CONV_KERNEL,
        D,
        HIDDEN_SIZE,
        MODEL_HEAD_COUNT,
        OPROJ_KC,
        OPROJ_KS,
        QKV_PANEL_COUNT,
        ReplayPool,
        S4_SEQUENCE_LENGTH,
        S8_SEQUENCE_LENGTH,
        Schedule,
        TILE,
        TP16_HEADS,
        TP16_TP_DEGREE,
        TP4_HEADS,
        TP4_TP_DEGREE,
        TP8_HEADS,
        TP8_TP_DEGREE,
        _CheckpointPool,
        _ConvPool,
        _checkpoint_batches,
        _checkpoint_head,
        _mega_workspace_views,
        _pack_conv_weights,
        _physical_span,
        replay_batches,
    )
    from .mega_recurrent_kda_frontend import (
        Core32ProjectionSchedule,
        MultiCoreProjectionSchedule,
        ProjectionScheduleOperationsFrontendOperations,
        S4ProjectionScheduleFrontendOperations,
        S4ProjectionStage,
        SequenceConvStage,
        _accurate_auxiliary_projection,
        _auxiliary_twofold_projection,
        _twofold_qkv_projection,
        prepare_optimized_precision,
        SequenceProjectionStage,
        TP16GridScheduleFrontendOperations,
        TP16S4AuxiliarySchedule,
        TP4AuxiliaryScheduleFrontendOperations,
        TP4GridScheduleFrontendOperations,
        TP4S4AuxiliarySchedule,
        TP8AuxiliaryScheduleFrontendOperations,
        TP8S4AuxiliarySchedule,
        TPScheduleFrontendOperations,
        TailMaskedCore32ProjectionSchedule,
    )
else:
    from mega_recurrent_kda_backend import (
        KDA_STATE_KEY_UNROLL,
        KDA_STATE_UPDATE_UNROLL,
        GridScheduleBackendOperations,
        OutputGateStage,
        OutputProjectionStage,
        RecurrentKDAStage,
        RecurrentStage,
        S4PipelinedBackendSchedule,
        SequenceRecurrentStage,
        TP16S4BackendSchedule,
        TP4BackendSchedule,
        TP4HeadPanelBackend,
        TP4S4BackendSchedule,
        TP4TPScheduleBackendOperations,
        TP8BackendSchedule,
        TP8HeadPanelBackend,
        TP8TPScheduleBackendOperations,
        _DelayedGateProjection,
        _HeadStreamOutput,
        _RawGateContext,
        _RawRecurrentContext,
        _TiledOutputProjectionStage,
        _head_stream_persistent_raw_round,
        _run_raw_recurrent_item,
        _store_raw_gated_item,
        publish_batch_group_vector_grid_to_cube,
        publish_vector_grid_to_cube,
        run_recurrent_wave,
        run_s4_raw_item,
    )
    from mega_recurrent_kda_common import (
        BETA_TILE,
        BATCH_TILE,
        CONV_KERNEL,
        D,
        HIDDEN_SIZE,
        MODEL_HEAD_COUNT,
        OPROJ_KC,
        OPROJ_KS,
        QKV_PANEL_COUNT,
        ReplayPool,
        S4_SEQUENCE_LENGTH,
        S8_SEQUENCE_LENGTH,
        Schedule,
        TILE,
        TP16_HEADS,
        TP16_TP_DEGREE,
        TP4_HEADS,
        TP4_TP_DEGREE,
        TP8_HEADS,
        TP8_TP_DEGREE,
        _CheckpointPool,
        _ConvPool,
        _checkpoint_batches,
        _checkpoint_head,
        _mega_workspace_views,
        _pack_conv_weights,
        _physical_span,
        replay_batches,
    )
    from mega_recurrent_kda_frontend import (
        Core32ProjectionSchedule,
        MultiCoreProjectionSchedule,
        ProjectionScheduleOperationsFrontendOperations,
        S4ProjectionScheduleFrontendOperations,
        S4ProjectionStage,
        SequenceConvStage,
        _accurate_auxiliary_projection,
        _auxiliary_twofold_projection,
        _twofold_qkv_projection,
        prepare_optimized_precision,
        SequenceProjectionStage,
        TP16GridScheduleFrontendOperations,
        TP16S4AuxiliarySchedule,
        TP4AuxiliaryScheduleFrontendOperations,
        TP4GridScheduleFrontendOperations,
        TP4S4AuxiliarySchedule,
        TP8AuxiliaryScheduleFrontendOperations,
        TP8S4AuxiliarySchedule,
        TPScheduleFrontendOperations,
        TailMaskedCore32ProjectionSchedule,
    )


# 模型维度与 TP 映射
# 下发几何及固定长调度参数。
VALID_CORES = (8, 16, 24, 28, 32)
MAIN_TEMPLATES = (
    (32, 384),
    (64, 384),
    (128, 384),
    (32, 256),
    (128, 256),
    (256, 256),
    (64, 256),
)

S4_DELAYED_ITEMS_PER_LANE = 6
S4_DELAYED_RAW_ROW_COUNT = S4_DELAYED_ITEMS_PER_LANE * S4_SEQUENCE_LENGTH
S4_GATE_PROJECTION_READY_FLAG = 6

HEAD_STREAM_BATCH_COUNT = 32
HEAD_STREAM_LANE_COUNT = 64
HEAD_STREAM_ROUNDS = TP8_HEADS * HEAD_STREAM_BATCH_COUNT // HEAD_STREAM_LANE_COUNT
GATE_PROJECTION_TASK_COUNT = 2 * TP8_HEADS

RECURRENT_CORES_PER_CORE_GROUP = 12
COMPACT_RECURRENT_CORE_COUNT = 2 * RECURRENT_CORES_PER_CORE_GROUP
COMPACT_CORES_PER_MODEL_BATCH = 24
DELAYED_GATE_TASK_COUNT = 4 * TP16_HEADS

SUPPORTED_HEAD_COUNTS = (TP16_HEADS, TP8_HEADS, TP4_HEADS)

# 固定长度融合路径与支持的 AIC 配额


def _shared_kda_call_state():
    """在独立加载的 KDA 模块之间共享调用锁和事件状态。"""
    key = "_cannbotdsl_kda_call_state"
    state = sys.modules.get(key)
    if state is None:
        candidate = ModuleType(key)
        candidate.call_lock = threading.RLock()
        candidate.pending = {"event": None, "keep": None}
        state = sys.modules.setdefault(key, candidate)
    return state


_kda_call_state = _shared_kda_call_state()
_call_lock = _kda_call_state.call_lock
_pending = _kda_call_state.pending


class InputValidation:
    """负责参数和存储约束校验。"""

    @staticmethod
    def require(condition, message):
        if isinstance(condition, bool):
            if not condition:
                raise ValueError(message)
        else:
            torch._check(condition, lambda: message)

    @staticmethod
    def tensor_metadata(name, tensor, shape, dtype, device):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        shape_label = {
            "hidden_states": "[B,seq_len,H] or [total_tokens,H]",
            "qkv_projection_weight": "[3*P,H]",
            "decay_projection_a_weight": "[D,H]",
            "decay_projection_b_weight": "[P,D]",
            "beta_projection_weight": "[N,H]",
            "output_gate_projection_weight": "[P,H]",
            "output_norm_weight": "[D]",
            "output_projection_weight": "[H,P]",
            "state_indices": "[B]",
            "conv_num_accepted_tokens": "[B]",
            "num_accepted_tokens": "[B]",
            "A_log": "[N]",
            "dt_bias": "[N,D]",
            "cu_seqlens": "[B+1]",
            "conv1d_weight": "[4,3*P], [3,N,4,D], or [3,4,D]",
            "replay_u": "[block_num,N,S,D]",
            "replay_k": "[block_num,N,S,D]",
            "replay_decay": "[block_num,N,S,D]",
        }.get(name, str(shape))
        InputValidation.require(
            tensor.ndim == len(shape), f"{name} must have shape {shape_label}"
        )
        for actual, expected in zip(tensor.shape, shape):
            if expected is not None:
                InputValidation.require(
                    actual == expected, f"{name} must have shape {shape_label}"
                )
        label = {
            torch.float16: "FP16",
            torch.float32: "FP32",
            torch.bfloat16: "BF16",
            torch.int32: "int32",
            torch.int64: "int64",
        }[dtype]
        if name == "hidden_states":
            InputValidation.require(
                tensor.dtype == torch.bfloat16 and tensor.is_contiguous(),
                "hidden_states must be contiguous BF16",
            )
        InputValidation.require(
            tensor.dtype == dtype, f"{name} must have dtype {label}"
        )
        InputValidation.require(
            tensor.layout == torch.strided, f"{name} must have strided layout"
        )
        InputValidation.require(tensor.is_contiguous(), f"{name} must be contiguous")
        InputValidation.require(
            tensor.device == device, f"{name} must share the input device"
        )
        InputValidation.require(
            not tensor.is_neg() and (not tensor.is_conj()),
            f"{name} must resolve negative/conjugate view bits",
        )

    @staticmethod
    def runtime_scalars(scale, lower_bound, rms_norm_eps):
        values = []
        for name, value in (
            ("scale", scale),
            ("lower_bound", lower_bound),
            ("rms_norm_eps", rms_norm_eps),
        ):
            value = float(value)
            narrowed = ctypes.c_float(value).value
            InputValidation.require(
                math.isfinite(value) and math.isfinite(narrowed),
                f"{name} must be finite and representable as FP32",
            )
            values.append(narrowed)
        InputValidation.require(
            -5.0 <= float(lower_bound) <= 0.0, "lower_bound must be in [-5,0]"
        )
        InputValidation.require(
            float(rms_norm_eps) > 0.0 and values[2] > 0.0,
            "rms_norm_eps must be finite and positive after FP32 conversion",
        )
        InputValidation.require(
            values[2] >= float.fromhex("0x1.0p-126"),
            "rms_norm_eps must be a normal FP32 value (>= 2**-126); subnormal epsilon is flushed to zero",
        )
        return tuple(values)

    @staticmethod
    def state_pool_metadata(name, tensor, shape, dtype, device):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        shape_label = (
            "[block_num,L,3*P]"
            if name == "conv_state"
            else "[block_num,N,D,D]"
            if name == "checkpoint_state"
            else "[block_num,N,S,D]"
        )
        InputValidation.require(
            tensor.ndim == len(shape), f"{name} must have shape {shape_label}"
        )
        for actual, expected in zip(tensor.shape, shape):
            if expected is not None:
                InputValidation.require(
                    actual == expected, f"{name} must have shape {shape_label}"
                )
        InputValidation.require(
            tensor.dtype == dtype, f"{name} must have dtype {dtype}"
        )
        InputValidation.require(
            tensor.layout == torch.strided and tensor.device == device,
            f"{name} must share input device and strided layout",
        )
        expected = 1
        for axis in range(tensor.ndim - 1, 0, -1):
            InputValidation.require(
                tensor.stride(axis) == expected,
                f"{name} only axis 0 may be non-contiguous",
            )
            expected = expected * tensor.shape[axis]
        InputValidation.require(
            tensor.stride(0) >= expected,
            f"{name} axis-0 stride must not overlap paged-state blocks",
        )
        InputValidation.require(
            not tensor.is_neg() and (not tensor.is_conj()),
            f"{name} must resolve view bits",
        )

    @staticmethod
    def _records(a, batch, heads, device):
        u = a["replay_u"]
        if not isinstance(u, torch.Tensor):
            raise TypeError("replay_u must be a torch.Tensor")
        InputValidation.require(u.ndim == 4, "replay_u must be [block_num,N,S,D]")
        blocks = a["checkpoint_state"].shape[0]
        InputValidation.require(
            (blocks > 0) & (batch > 0) & (batch <= blocks),
            "batch must be in [1,block_num]",
        )
        capacity = u.shape[2]
        InputValidation.require(capacity > 0, "record S capacity must be positive")
        for name, dtype in (
            ("replay_u", torch.float32),
            ("replay_k", torch.float16),
            ("replay_decay", torch.float32),
        ):
            InputValidation.state_pool_metadata(
                name, a[name], (blocks, heads, capacity, D), dtype, device
            )
        return capacity

    @staticmethod
    def mega_metadata(a):
        x = a["hidden_states"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        InputValidation.require(x.ndim in (2, 3), "hidden_states must be BSH or TH")
        cu = a.get("cu_seqlens")
        if x.ndim == 3:
            InputValidation.require(cu is None, "cu_seqlens must be None for BSH input")
            batch, sequence, width = x.shape
            InputValidation.require(sequence > 0, "BSH seq_len must be positive")
        else:
            if not isinstance(cu, torch.Tensor):
                raise TypeError("cu_seqlens is required for TH input")
            InputValidation.require(
                cu.ndim == 1 and cu.shape[0] >= 2, "cu_seqlens must be [B+1]"
            )
            InputValidation.require(
                cu.dtype in (torch.int32, torch.int64),
                "cu_seqlens must be int32 or int64",
            )
            InputValidation.tensor_metadata(
                "cu_seqlens", cu, cu.shape, cu.dtype, x.device
            )
            batch = cu.shape[0] - 1
            width = x.shape[1]
            InputValidation.require(x.shape[0] > 0, "TH token count must be positive")
        InputValidation.require(width == HIDDEN_SIZE, "hidden_states requires H=7168")
        InputValidation.tensor_metadata(
            "hidden_states", x, x.shape, torch.bfloat16, x.device
        )
        beta = a["beta_projection_weight"]
        if not isinstance(beta, torch.Tensor):
            raise TypeError("beta_projection_weight must be a torch.Tensor")
        InputValidation.require(beta.ndim == 2, "beta_projection_weight must be [N,H]")
        n = beta.shape[0]
        InputValidation.require(
            (n == TP16_HEADS) | (n == TP8_HEADS) | (n == TP4_HEADS),
            "TP4/8/16 requires N=24/12/6",
        )
        p = n * D
        device = x.device
        entries = [
            ("qkv_projection_weight", (3 * p, HIDDEN_SIZE), torch.bfloat16),
            ("decay_projection_a_weight", (D, HIDDEN_SIZE), torch.bfloat16),
            ("decay_projection_b_weight", (p, D), torch.bfloat16),
            ("beta_projection_weight", (n, HIDDEN_SIZE), torch.bfloat16),
            ("output_gate_projection_weight", (p, HIDDEN_SIZE), torch.bfloat16),
            ("output_norm_weight", (D,), torch.float32),
            ("output_projection_weight", (HIDDEN_SIZE, p), torch.bfloat16),
            ("state_indices", (batch,), torch.int32),
            ("conv_num_accepted_tokens", (batch,), torch.int32),
            ("A_log", (n,), torch.float32),
            ("dt_bias", (n, D), torch.float32),
        ]
        for name, shape, dt in entries:
            InputValidation.tensor_metadata(name, a[name], shape, dt, device)
        conv = a["conv_state"]
        checkpoint = a["checkpoint_state"]
        InputValidation.state_pool_metadata(
            "conv_state", conv, (None, None, 3 * p), torch.bfloat16, device
        )
        InputValidation.state_pool_metadata(
            "checkpoint_state",
            checkpoint,
            (conv.shape[0], n, D, D),
            torch.float32,
            device,
        )
        InputValidation.require(
            conv.shape[0] > 0 and conv.shape[1] >= 3,
            "paged-state block count must be positive and conv L >= 3",
        )
        capacity = InputValidation._records(a, batch, n, device)
        if x.ndim == 3:
            InputValidation.require(
                capacity >= x.shape[1], "replay S capacity must cover BSH seq_len"
            )
            InputValidation.require(
                conv.shape[1] >= x.shape[1] + 2, "conv_state L must be >= seq_len+2"
            )
        w = a["conv1d_weight"]
        if not isinstance(w, torch.Tensor):
            raise TypeError("conv1d_weight must be a torch.Tensor")
        InputValidation.require(
            tuple(w.shape)
            in ((CONV_KERNEL, 3 * p), (3, n, CONV_KERNEL, D), (3, CONV_KERNEL, D)),
            "invalid conv1d_weight shape",
        )
        InputValidation.tensor_metadata(
            "conv1d_weight", w, w.shape, torch.bfloat16, device
        )
        InputValidation.runtime_scalars(a["scale"], a["lower_bound"], a["rms_norm_eps"])
        for name in ("conv_state", "replay_u", "replay_k", "replay_decay"):
            InputValidation.require(
                not a[name].requires_grad,
                f"{name} is an inference-only writable buffer",
            )
        return {
            name: value for name, value in a.items() if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _bounded(values, low, high, name, message=None):
        if any((v < low or v > high for v in values)):
            raise ValueError(message or f"{name} values must be in [{low},{high}]")

    @staticmethod
    def validate_mega_values(a):
        batch = a["state_indices"].numel()
        x = a["hidden_states"]
        if x.ndim == 3:
            prefix = [i * x.shape[1] for i in range(batch + 1)]
        else:
            prefix = a["cu_seqlens"].detach().cpu().tolist()
        if torch.npu.is_current_stream_capturing():
            return prefix
        tokens = x.numel() // HIDDEN_SIZE
        InputValidation.require(
            prefix[0] == 0 and prefix[-1] == tokens,
            "cu_seqlens must start at 0 and end at T",
        )
        InputValidation.require(
            all((x < y for x, y in zip(prefix, prefix[1:]))),
            "cu_seqlens must be strictly increasing; every sequence length must be positive",
        )
        InputValidation.require(
            tokens < 2**31, "token count exceeds int32 cu_seqlens capacity"
        )
        lengths = [y - x for x, y in zip(prefix, prefix[1:])]
        InputValidation.require(
            max(lengths) <= a["replay_u"].shape[2],
            "replay S capacity must cover every sequence length",
        )
        InputValidation.require(
            max(lengths) + 2 <= a["conv_state"].shape[1],
            "conv_state L must cover every sequence length + 2",
        )
        indices = a["state_indices"].detach().cpu().tolist()
        InputValidation._bounded(
            indices, 0, a["conv_state"].shape[0] - 1, "state_indices"
        )
        InputValidation.require(
            len(indices) == len(set(indices)),
            "state_indices must be distinct across batch, including slot 0",
        )
        counts = a["conv_num_accepted_tokens"].detach().cpu().tolist()
        InputValidation._bounded(
            counts, 0, a["conv_state"].shape[1] - 2, "conv_num_accepted_tokens"
        )
        return prefix

    @staticmethod
    def validate_aliases(tensors, writable):
        spans = {}
        for name, tensor in tensors.items():
            InputValidation.require(
                tensor.device.type != "meta",
                "physical alias validation requires real tensors",
            )
            start = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            if name in (
                "conv_state",
                "checkpoint_state",
                "replay_u",
                "replay_k",
                "replay_decay",
            ):
                width = tensor[0].numel() * tensor.element_size()
                ranges = [
                    (
                        start + i * tensor.stride(0) * tensor.element_size(),
                        start + i * tensor.stride(0) * tensor.element_size() + width,
                    )
                    for i in range(tensor.shape[0])
                ]
            else:
                if tensor.device.type in ("npu", "privateuseone"):
                    import torch_npu

                    if torch_npu.get_npu_format(tensor) == 29:
                        size = (
                            int(torch_npu.get_storage_size(tensor))
                            * tensor.element_size()
                        )
                ranges = [(start, start + size)]
            spans[name] = ranges
        names = tuple(tensors)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                if left not in writable and right not in writable:
                    continue
                if tensors[left].device != tensors[right].device:
                    continue
                x = y = 0
                a = spans[left]
                b = spans[right]
                while x < len(a) and y < len(b):
                    if a[x][0] < b[y][1] and b[y][0] < a[x][1]:
                        raise ValueError(f"writable buffer overlap: {left} and {right}")
                    if a[x][1] <= b[y][0]:
                        x += 1
                    else:
                        y += 1

    @staticmethod
    def real_boundary(tensors, writable):
        for name, tensor in tensors.items():
            InputValidation.require(
                tensor.device.type in ("npu", "privateuseone"),
                f"{name} requires NPU inputs",
            )
            InputValidation.require(
                type(tensor) is torch.Tensor,
                f"{name} must be an exact Tensor; detach Parameter inputs",
            )
        InputValidation.validate_aliases(tensors, writable)
        import torch_npu

        nz_weights = (
            "qkv_projection_weight",
            "decay_projection_a_weight",
            "decay_projection_b_weight",
            "beta_projection_weight",
            "output_gate_projection_weight",
            "output_projection_weight",
        )
        for name, tensor in tensors.items():
            if name not in nz_weights:
                InputValidation.require(
                    torch_npu.get_npu_format(tensor) in (0, 2),
                    f"{name} requires ND storage; paged-state addressing does not support blocked physical formats",
                )
        if "hidden_states" in tensors:
            for name in (
                "qkv_projection_weight",
                "decay_projection_a_weight",
                "decay_projection_b_weight",
                "beta_projection_weight",
                "output_gate_projection_weight",
                "output_projection_weight",
            ):
                InputValidation.require(
                    torch_npu.get_npu_format(tensors[name]) == 29,
                    f"{name} requires BF16 FRACTAL_NZ storage",
                )
            return resolve_core_count(tensors["hidden_states"])


class KDAExecutionGuard:
    """负责调用同步和异步资源生命周期保护。"""

    @staticmethod
    def serialized(function):
        @functools.wraps(function)
        def checked(*args, **kwargs):
            ref = (
                args[0]
                if args
                else kwargs.get("hidden_states", kwargs.get("checkpoint_state"))
            )
            if not isinstance(ref, torch.Tensor) or ref.device.type not in (
                "npu",
                "privateuseone",
            ):
                return function(*args, **kwargs)
            device = ref.device
            with torch.npu.device(device):
                if torch.npu.is_current_stream_capturing():
                    with _call_lock:
                        return function(*args, **kwargs)
                with _call_lock:
                    if _pending["event"] is not None:
                        _pending["event"].synchronize()
                        _pending["event"] = None
                        _pending["keep"] = None
                    result = None
                    try:
                        result = function(*args, **kwargs)
                        return result
                    finally:
                        event = torch.npu.Event()
                        event.record(torch.npu.current_stream(device))
                        _pending["event"] = event
                        _pending["keep"] = (args, kwargs, result)

        return checked


def resolve_core_count(ref):
    """查询硬件和当前流的有效配额，校验后返回支持的最大 AIC 核数。"""
    if ref.device.type not in ("npu", "privateuseone"):
        raise ValueError("mega_recurrent_kda requires an NPU tensor")
    hardware = torch.npu.get_device_properties(ref.device)
    info = cannbotdsl.get_platform_info(stream=torch.npu.current_stream(ref.device))
    cube_count = min(int(hardware.cube_core_num), int(info.cube_core_num))
    vector_count = min(int(hardware.vector_core_num), int(info.vector_core_num))
    effective_quota = min(cube_count, vector_count // 2)
    InputValidation.require(
        effective_quota >= max(VALID_CORES) or effective_quota in VALID_CORES,
        f"AIC quota {effective_quota} is not supported; use 8/16/24/28/32 or a full stream with at least 32 AIC and 64 AIV",
    )
    return max(core for core in VALID_CORES if core <= effective_quota)


class _MainGeometry(NamedTuple):
    cost: int
    jobs: int
    rounds: int
    input_bytes: int
    weight_bytes: int
    padded_rows: int


def main_geometry(rows, heads, cores, tile_m, tile_n):
    """平衡各核的行计算量与重复投影准备开销。

    每个波次按一个 head 宽度计入准备开销。N384 要求完整的行分块；
    N256 对最后一个行分块使用掩码。字节数表示逻辑搬运量的估计值。
    """
    groups = (rows + tile_m - 1) // tile_m
    jobs = groups * (4 * heads * D // tile_n)
    rounds = (jobs + cores - 1) // cores
    a_bytes = jobs * tile_m * HIDDEN_SIZE * 2
    w_bytes = groups * 4 * heads * D * HIDDEN_SIZE * 2
    cost = rounds * (tile_m + D)
    if tile_n == 384 and rows % tile_m:
        cost = 1 << 60
    return _MainGeometry(cost, jobs, rounds, a_bytes, w_bytes, groups * tile_m)


def _tile_rows(rows, tile_rows, ragged):
    return (
        min(tile_rows, 1 << (rows - 1).bit_length())
        if ragged
        else math.gcd(rows, tile_rows)
    )


class _S4StagedPipeline:
    """三个阶段对象由各 TP 创建，统一维护阶段间的同步边界。"""

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        indices,
        accepted,
        wc,
        scale,
        lower,
        eps,
        rows,
        width,
        policy,
        output_rows,
    ):
        self.main.main(
            q, k, v, gate, x, wqkv, wz, conv, indices, accepted, wc, rows, width, policy
        )
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.auxiliary.execute(decay, beta, fa, x, wa, wb, wbeta, output_rows, policy)
        self.backend.execute(
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            gamma,
            wo,
            alog,
            bias,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
            eps,
            output_rows,
            policy,
        )


TP4_SHORT_SCHEDULE = {
    "main_rows": 32,
    "main_width": 384,
    "auxiliary_rows": 32,
    "main_weight_l2_cache_ctl": 4,
    "state_store_l2_cache_ctl": 0,
    "state_load_l2_cache_ctl": 2,
    "fa_width": 64,
}


TP4_LARGE_SCHEDULE = {
    "ragged_output": True,
    "ragged_main": True,
    "main_rows": 256,
    "auxiliary_rows": 0,
    "main_width": 256,
    "split_fb": True,
    "publish_rows": 128,
    "resident_output_weight": True,
    "state_update_unroll": 2,
    "main_weight_l2_cache_ctl": 0,
    "state_store_l2_cache_ctl": 4,
    "state_load_l2_cache_ctl": 2,
}


class TP4GridSchedule(TP4GridScheduleFrontendOperations):
    """使用编译期分支配置，并共用一套 KDA 波次调度。"""

    def __init__(self, block_num: int, hidden_size: int, num_heads: int):
        self.state_store_l2_cache_ctl = 0
        self.block_num = block_num
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.projection_groups = 8
        self.batch_tile = BATCH_TILE
        self.qkv_block_count = 3 * num_heads
        self.output_gate_block_count = num_heads
        self.beta_block = self.qkv_block_count + num_heads
        self.low_rank_block = self.beta_block + 1
        self.recurrent_wave_count = 2
        self.output_tiles_per_core = {8: 7, 16: 2, 24: 2, 28: 1, 32: 1}[block_num]
        self.output_tile_width = {8: 128, 16: 256, 24: 256, 28: 256, 32: 256}[block_num]
        activation_channel_bytes = 2 * 64 * OPROJ_KC * 2
        self.output_weight_resident = (
            self.output_tile_width * self.output_tiles_per_core * num_heads * TILE * 2
            + activation_channel_bytes
            <= cannbotdsl.get_mem_size("l1")
        )
        self.projection_input_l2_cache_ctl = 0
        self.projection_weight_l2_cache_ctl = 4
        self.batch_groups: object
        self.seq_len: object
        self.row_count: object
        self.projection_row_block: object
        self.recurrent_wave_rows: object


class TP4TPSchedule(
    TPScheduleFrontendOperations, TP4TPScheduleBackendOperations, TP4GridSchedule
):
    def __init__(self, num_heads, rows, auxiliary_rows, publish_rows, config):
        super().__init__(32, HIDDEN_SIZE, num_heads)
        self.main_rows = rows
        self.auxiliary_rows = auxiliary_rows
        self.publish_rows = publish_rows
        self.config = config
        self.seq_len = S8_SEQUENCE_LENGTH


class TP4ProjectionScheduleOperations(
    ProjectionScheduleOperationsFrontendOperations, TP4TPSchedule
):
    """为两种投影调度共用构造逻辑和 Conv 预取操作。"""

    def __init__(self, heads, rows, width):
        config = Schedule(
            main_rows=rows,
            main_width=width,
            ragged_main=width == 256,
            split_fb=True,
            ragged_output=True,
        )
        super().__init__(heads, rows, 32, 64, config)
        self.heads = heads
        self.projection_groups = 4 * heads // 3
        self.row_capacity = rows


class TP4AuxiliarySchedule(TP4AuxiliaryScheduleFrontendOperations, TP4TPSchedule):
    def __init__(self, heads, rows):
        super().__init__(heads, 32, rows, 64, Schedule(split_fb=True))
        self.heads = heads


class TP4S4ProjectionSchedule(
    S4ProjectionScheduleFrontendOperations, TP4ProjectionScheduleOperations
):
    """仅处理有效行的 S4 QKV/Conv 融合，复用当前 PA 数据面板消费逻辑。"""

    def __init__(self, heads):
        super().__init__(heads, 128, 384)
        self.seq_len = S4_SEQUENCE_LENGTH


class TP4S4MultiCoreSchedule(_S4StagedPipeline):
    """通用 S4 融合，支持 TP4/8/16 和 8/16/24/28/32 AIC。"""

    def __init__(self, heads):
        self.main = TP4S4ProjectionSchedule(heads)
        self.auxiliary = TP4S4AuxiliarySchedule(heads)
        self.backend = TP4S4BackendSchedule(heads)


class TP4PhaseSchedule(TP4TPSchedule):
    """保留经过实测的输出分块几何，并共用运行时缓存控制。"""

    @jit
    def _configure_recurrent(self, _recurrent_stage):
        _recurrent_stage.store_policy = self.store_policy
        _recurrent_stage.load_policy = self.load_policy


class _TP4S8Core32Stage:
    def __init__(self):
        self.head_backend = TP4HeadPanelBackend(TP4_HEADS)
        self.main384 = Core32ProjectionSchedule(TP4_HEADS, 128, 384)
        self.main256 = Core32ProjectionSchedule(TP4_HEADS, 256, 256)
        self.aux0 = TP4TPSchedule(TP4_HEADS, 128, 32, 64, Schedule(fa_width=64))
        self.aux1 = TP4TPSchedule(TP4_HEADS, 128, 32, 64, Schedule(fa_width=128))
        self.back0 = TP4PhaseSchedule(
            TP4_HEADS, 128, 64, 32, Schedule(publish_rows=32, ragged_output=False)
        )
        self.back1 = TP4PhaseSchedule(
            TP4_HEADS, 128, 64, 64, Schedule(publish_rows=64, ragged_output=False)
        )
        self.back2 = TP4PhaseSchedule(
            TP4_HEADS, 128, 64, 128, Schedule(publish_rows=128, ragged_output=True)
        )
        self.back3 = TP4PhaseSchedule(
            TP4_HEADS, 128, 64, 256, Schedule(publish_rows=256, ragged_output=True)
        )

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
    ):
        mb = replay_decay.indices.shape[0] // 4
        main_policy = dtypes.int32(4)
        if mb > 8:
            main_policy = dtypes.int32(0)
        store_policy = dtypes.int32(0)
        if mb >= 4:
            store_policy = dtypes.int32(4)
        load_policy = dtypes.int32(2)
        main_rows = 256
        if mb == 1:
            main_rows = 32
        elif mb == 2:
            main_rows = 64
        elif mb == 3 or mb == 4:
            main_rows = 128
        use384 = mb == 1 or mb == 2 or mb == 4
        # 对齐的 M128/N384 分块能够形成完整的投影波次。
        if mb >= 8 and out.shape[0] % 128 == 0:
            main_rows = 128
            use384 = True
        self.main256.main_policy = main_policy
        self.main256.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            TP4_HEADS,
            main_rows,
            not use384,
        )
        self.main384.main_policy = main_policy
        self.main384.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            TP4_HEADS,
            main_rows,
            use384,
        )
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.aux0.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        else:
            self.aux1.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.aux0.frontend_fb(decay, fa, wb)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.head_backend.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        elif mb == 2 or mb == 4:
            self.back1.store_policy = store_policy
            self.back1.load_policy = load_policy
            self.back1.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        elif mb == 3 or mb >= 16:
            self.back2.store_policy = store_policy
            self.back2.load_policy = load_policy
            self.back2.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        else:
            self.back3.store_policy = store_policy
            self.back3.load_policy = load_policy
            self.back3.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )


class TP4DelayedProjectionSchedule:
    """对完整的大行块重叠执行门控投影与递推。"""

    def __init__(self, heads):
        self.heads = heads
        self.zrows = 128
        self.publish_rows = 128
        self.window_rows = self.publish_rows
        self.minimum_rows = 4 * self.publish_rows
        self.main = Core32ProjectionSchedule(heads, 128, 384)
        self.main.projection_groups = heads
        self.aux = TP4TPSchedule(heads, 128, 32, 128, Schedule(publish_rows=128))

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        self.main.main_policy = dtypes.int32(0)
        self.main.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            self.heads,
            128,
        )
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.aux.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.aux.frontend_fb(decay, fa, wb)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        context = _RawRecurrentContext(
            q,
            k,
            v,
            initial.accurate_decay,
            beta,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
        )
        for chunk in range(x.shape[0] // self.publish_rows):
            config = Schedule(state_load_l2_cache_ctl=2)
            recurrent = RecurrentStage(128, self.heads, config, state_buffer_depth=1)
            output_gate = OutputGateStage()
            raw = Buffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout(
                    (ceil_div(self.publish_rows // 8 * self.heads, 64) * 8, 128)
                ),
            )
            gate_context = _RawGateContext(q, gate, raw, eps)
            zstage = _DelayedGateProjection(self.zrows, 0, 0, row_capacity=self.zrows)
            recurrent.load_gate_params(alog, bias)
            output_gate.load_gamma(gamma)
            for ztask in range(
                get_block_idx(),
                self.publish_rows // self.zrows * self.heads,
                get_block_num(),
            ):
                head = ztask // (self.publish_rows // self.zrows)
                rowpart = ztask % (self.publish_rows // self.zrows)
                zstage.project_to_gm(
                    tile_slice(
                        x,
                        (self.zrows, HIDDEN_SIZE),
                        (chunk * (self.publish_rows // self.zrows) + rowpart, 0),
                    ),
                    wz,
                    tile_slice(
                        gate,
                        (self.zrows, 128),
                        (chunk * (self.publish_rows // self.zrows) + rowpart, head),
                    ),
                    head,
                    56,
                )
            first = get_block_idx() * 2 + get_subblock_id()
            for item in range(first, self.publish_rows // 8 * self.heads, 64):
                batch = chunk * (self.publish_rows // 8) + item % (
                    self.publish_rows // 8
                )
                head = item // (self.publish_rows // 8)
                _run_raw_recurrent_item(recurrent, context, batch, head, raw, item)
            cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
            cube_sync_block_wait(PIPE.S, 0, mode=0)
            cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=2)
            vec_sync_block_wait(PIPE.S, 1, mode=2)
            for item in range(first, self.publish_rows // 8 * self.heads, 64):
                batch = chunk * (self.publish_rows // 8) + item % (
                    self.publish_rows // 8
                )
                head = item // (self.publish_rows // 8)
                _store_raw_gated_item(output_gate, gate_context, batch, head, item)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            onchip_rewind(reset_sync_id=True)
            output = OutputProjectionStage(
                256,
                self.publish_rows,
                kc=256,
                tiles_per_core=1,
                resident_weight=False,
                row_capacity=self.publish_rows,
                num_heads=self.heads,
            )
            output.project(out, q, wo, self.heads, chunk)
            cube_sync_all()
            vec_sync_all()
            onchip_rewind(reset_sync_id=True)


class S4GateProjection(_DelayedGateProjection):
    """仅计算逻辑有效 M48/M64 行的 Z 投影，不分配仅用于类型转换的 Vector 缓冲区。"""

    matmul_hidden = S4ProjectionStage.matmul_hidden
    accumulate_panel = S4ProjectionStage.accumulate_panel


class S4DelayedGateSchedule:
    """TP4 模型 batch 为 3/4、使用 32 AIC 时，在 UB 中重叠执行 Z 投影与递推。"""

    def __init__(self):
        self.main = TP4S4ProjectionSchedule(TP4_HEADS)
        self.auxiliary = TP4S4AuxiliarySchedule(TP4_HEADS)
        self.backend = TP4S4BackendSchedule(TP4_HEADS)

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        indices,
        accepted,
        wc,
        scale,
        lower,
        eps,
        rows,
        width,
        policy,
        output_rows,
    ):
        self.main.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            indices,
            accepted,
            wc,
            rows,
            width,
            policy,
            qkv_only=True,
        )
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.auxiliary.execute(decay, beta, fa, x, wa, wb, wbeta, output_rows, policy)
        # Cube 生成 Z 时，配对 Vector 引擎保留 KDA 的 FP32 原始输出。
        # B12/B16、N24 时，每个 AIV 最多处理六个任务，即 24 行。
        projection = S4GateProjection(x.shape[0], 0, 4, row_capacity=64)
        for head in range(get_block_idx(), TP4_HEADS, get_block_num()):
            projection.project_to_gm(
                x, wz, gate[None, head * 128 : (head + 1) * 128], head, 56
            )
        raw = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((S4_DELAYED_RAW_ROW_COUNT, 128))
        )
        recurrent = RecurrentStage(128, TP4_HEADS, Schedule(state_load_l2_cache_ctl=4))
        output_gate = OutputGateStage()
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        context = _RawRecurrentContext(
            q,
            k,
            v,
            initial.accurate_decay,
            beta,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
        )
        first = get_block_idx() * 2 + get_subblock_id()
        for item in range(first, replay_u.indices.shape[0] * TP4_HEADS, 64):
            run_s4_raw_item(
                recurrent, context, item // TP4_HEADS, item % TP4_HEADS, raw, item
            )
        # 配对 Vector 加载门控数据前，先完成全部 Cube Z 写入。
        # 这些标志仅用于 S4 分支，不改变 S8 的同步协议。
        cube_sync_block_arrive(PIPE.FIXPIPE, S4_GATE_PROJECTION_READY_FLAG, mode=0)
        cube_sync_block_wait(PIPE.S, S4_GATE_PROJECTION_READY_FLAG, mode=0)
        cube_sync_block_arrive(PIPE.FIXPIPE, 7, mode=2)
        vec_sync_block_wait(PIPE.S, 7, mode=2)
        for item in range(first, replay_u.indices.shape[0] * TP4_HEADS, 64):
            batch = item // TP4_HEADS
            head = item % TP4_HEADS
            output_gate.issue_gate(tile_slice(gate, (1, 128), (batch * 4, head)))
            for token in range(4):
                if token + 1 < 4:
                    output_gate.issue_gate(
                        tile_slice(gate, (1, 128), (batch * 4 + token + 1, head))
                    )
                output_gate.apply_and_store(
                    tile_slice(q, (1, 128), (batch * 4 + token, head)),
                    eps,
                    tile_slice(raw, (1, 128), (item // 64 * 4 + token, 0)),
                )
        publish_vector_grid_to_cube(3, 4, ack_flag=5)
        onchip_rewind(reset_sync_id=True)
        self.backend.project_output(out, q, wo, output_rows, policy)
        global_sync_all()


def _tp4_intermediate_schedule(num_heads, model_batch):
    rows = model_batch * (MODEL_HEAD_COUNT // num_heads) * S8_SEQUENCE_LENGTH
    if rows <= 128 and rows & (rows - 1) == 0:
        options = dict(main_rows=128, main_width=384)
    else:
        # 递推工作占主导时，TP8 按较小的数据块发布结果。
        publish_rows = 256
        options = dict(
            main_rows=256,
            main_width=256,
            ragged_main=True,
            ragged_output=True,
            publish_rows=publish_rows,
        )
    physical_rows = _tile_rows(rows, options["main_rows"], options.get("ragged_main"))
    options["main_weight_l2_cache_ctl"] = 0 if rows > physical_rows else 4
    options["state_store_l2_cache_ctl"] = 4 if model_batch >= 4 else 0
    return Schedule(**options)


def _tp4_default_schedule(num_heads, model_batch):
    """根据 TP 和模型 batch 选择分块与缓存策略。"""
    if num_heads != TP4_HEADS or not 1 <= model_batch <= 32:
        raise ValueError(f"requires N={TP4_HEADS} and model_batch=1..32")
    if model_batch > 16:
        return Schedule(**TP4_LARGE_SCHEDULE)
    if model_batch in (1, 16):
        options = TP4_SHORT_SCHEDULE if model_batch == 1 else TP4_LARGE_SCHEDULE
        return Schedule(**options)
    return _tp4_intermediate_schedule(num_heads, model_batch)


def _tp4_select_main(rows, heads, cores):
    if heads != TP4_HEADS or cores not in VALID_CORES:
        raise ValueError("unsupported heads or AIC quota")
    return min(
        range(len(MAIN_TEMPLATES)),
        key=lambda i: main_geometry(rows, heads, cores, *MAIN_TEMPLATES[i]).cost,
    )


def _tp4_core32_geometry(model_batch, heads, rows):
    config = _tp4_default_schedule(heads, model_batch)
    aligned = model_batch >= 8 and rows % 128 == 0
    return dict(
        main_m=128
        if aligned
        else _tile_rows(rows, config.main_rows, config.ragged_main),
        main_n=384 if aligned else config.main_width,
        auxiliary_rows=math.gcd(
            rows,
            config.auxiliary_rows or (MODEL_HEAD_COUNT // heads) * S8_SEQUENCE_LENGTH,
        ),
        publish_rows=_tile_rows(rows, config.publish_rows, config.ragged_output),
    )


def _tp4_core32_plan(model_batch, tp):
    rows = model_batch * tp * S8_SEQUENCE_LENGTH
    heads = MODEL_HEAD_COUNT // tp
    geometry = _tp4_core32_geometry(model_batch, heads, rows)
    lanes = 64
    return dict(
        model_batch=model_batch,
        tp=tp,
        cores=32,
        rows=rows,
        heads=heads,
        main_id=-1,
        **geometry,
        main_geometry=main_geometry(
            rows, heads, 32, geometry["main_m"], geometry["main_n"]
        ),
        kda_waves=(MODEL_HEAD_COUNT * model_batch + lanes - 1) // lanes,
        vector_lanes=lanes,
        output_width=256,
        output="nb_head_panels" if model_batch == 1 else ("measured32_stream"),
        mode="specialized32",
    )


def _tp4_output_policy(rows, heads, cores):
    if rows * heads == 768:
        return "nb_head_panels"
    if rows in (32, 64):
        return "short32" if rows == 32 else "short64"
    return "stream256_r128"


def _tp4_plan(model_batch, tp, cores):
    if tp != TP4_TP_DEGREE or not 1 <= model_batch <= 32 or cores not in VALID_CORES:
        raise ValueError("requires TP4, model batch 1..32, AIC 8/16/24/28/32")
    rows = model_batch * tp * S8_SEQUENCE_LENGTH
    heads = MODEL_HEAD_COUNT // tp
    if cores == 32:
        return _tp4_core32_plan(model_batch, tp)
    i = _tp4_select_main(rows, heads, cores)
    m, n = MAIN_TEMPLATES[i]
    return dict(
        model_batch=model_batch,
        tp=tp,
        cores=cores,
        rows=rows,
        heads=heads,
        main_id=i,
        main_m=m,
        main_n=n,
        main_geometry=main_geometry(rows, heads, cores, m, n),
        kda_waves=(MODEL_HEAD_COUNT * model_batch + 2 * cores - 1) // (2 * cores),
        auxiliary_rows=math.gcd(rows, 128),
        output=_tp4_output_policy(rows, heads, cores),
        weight_policy=0 if rows > m else 4,
    )


class TP4S8DefaultStage:
    def __init__(self, heads):
        self.heads = heads
        self.main384 = MultiCoreProjectionSchedule(heads, 128, 384)
        self.main256 = MultiCoreProjectionSchedule(heads, 256, 256)
        self.aux32 = TP4AuxiliarySchedule(heads, 32)
        self.aux64 = TP4AuxiliarySchedule(heads, 64)
        self.aux128 = TP4AuxiliarySchedule(heads, 128)
        self.backend = TP4BackendSchedule(heads)
        self.head_backend = TP4HeadPanelBackend(heads)
        self.fast32 = _TP4S8Core32Stage()

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        if get_block_num() == 32:
            self.fast32.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog[0 : self.heads,],
                bias[0 : self.heads, None],
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
            )
        else:
            n = self.heads
            logical_rows = plan_rows
            if plan_width == 384:
                self.main384.main(
                    q,
                    k,
                    v,
                    gate,
                    x,
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    n,
                    logical_rows,
                    plan_policy,
                )
            else:
                self.main256.main(
                    q,
                    k,
                    v,
                    gate,
                    x,
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    n,
                    logical_rows,
                    plan_policy,
                )
            global_sync_all()
            onchip_rewind(reset_sync_id=True)
            if x.shape[0] % 128 == 0:
                self.aux128.execute(decay, beta, fa, x, wa, wb, wbeta)
            elif x.shape[0] % 64 == 0:
                self.aux64.execute(decay, beta, fa, x, wa, wb, wbeta)
            else:
                self.aux32.execute(decay, beta, fa, x, wa, wb, wbeta)
            global_sync_all()
            onchip_rewind(reset_sync_id=True)
            if replay_decay.indices.shape[0] * n == MODEL_HEAD_COUNT:
                self.head_backend.backend(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    gamma,
                    wo,
                    alog[0:n,],
                    bias[0:n, None],
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    scale,
                    lower,
                    eps,
                )
            else:
                self.backend.backend(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    scale,
                    lower,
                    eps,
                    wbeta,
                )
        global_sync_all()


class TP4S8Dispatch(TP4S8DefaultStage):
    def __init__(self, heads):
        super().__init__(heads)
        self.delayed = TP4DelayedProjectionSchedule(heads)

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
        prefer_stream,
    ):
        q = q.view((q.shape[0], self.heads * TILE))
        k = k.view((k.shape[0], self.heads * TILE))
        v = v.view((v.shape[0], self.heads * TILE))
        decay = decay.view((decay.shape[0], self.heads * TILE))
        gate = gate.view((gate.shape[0], self.heads * TILE))
        beta = beta.view(
            (beta.shape[0], ((self.heads + BETA_TILE - 1) // BETA_TILE) * BETA_TILE)
        )
        alog = alog.view((self.heads,))
        bias = bias.view((self.heads, TILE))
        conv = _ConvPool(
            conv.data,
            conv.stride0,
            conv.length,
            3 * self.heads * TILE,
            conv.accurate_raw,
        )
        wc = wc.view((3, self.heads, CONV_KERNEL, TILE))
        if (
            get_block_num() == 32
            and x.shape[0] % self.delayed.window_rows == 0
            and (x.shape[0] >= self.delayed.minimum_rows)
        ):
            self.delayed.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )
            global_sync_all()
        else:
            TP4S8DefaultStage.execute(
                self,
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )


def _tp4_plan_s4(model_batch, tp, cores):
    """采用保守的源码调度策略，不依赖实测性能查找表。"""
    if tp != TP4_TP_DEGREE or not 1 <= model_batch <= 32 or cores not in VALID_CORES:
        raise ValueError("S4 requires TP4, model batch 1..32, AIC 8/16/24/28/32")
    total_rows = model_batch * tp * S4_SEQUENCE_LENGTH
    main_rows = math.gcd(total_rows, 128)
    width = 384
    if total_rows >= 512 and total_rows % 256 == 0:
        main_rows, width = 256, 256
    template = 0
    if cores == 32 and model_batch in (3, 4):
        template = 2
    return dict(
        main_rows=main_rows,
        width=width,
        policy=4 if total_rows <= 128 else 0,
        template=template,
    )


class TP4TPOptimizedKernel:
    """实现单个 TP，为 S8 和 S4 分别缓存编译后的可调用对象。"""

    def __init__(self, sequence):
        self.sequence = sequence
        if sequence == S8_SEQUENCE_LENGTH:
            self.stage = TP4S8Dispatch(TP4_HEADS)
        else:
            self.stage = TP4S4MultiCoreSchedule(TP4_HEADS)
            self.delayed = S4DelayedGateSchedule()

    @kernel(
        profile=cannbotdsl.ProfileSpec(
            name="mega_recurrent_kda_tp4",
            op_type="MegaRecurrentKDA",
            outputs=("out", "replay_u", "replay_k", "replay_decay"),
            inouts=("conv",),
        )
    )
    def device(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        checkpoint,
        conv,
        indices,
        accepted,
        wc,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        conv_stride0,
        conv_length,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        main_rows,
        width,
        policy,
        selector,
        accurate_raw,
        accurate_decay,
    ):
        initial = _CheckpointPool(
            checkpoint, indices, checkpoint_stride0, accurate_decay
        )
        conv = _ConvPool(
            conv, conv_stride0, conv_length, 3 * TP4_HEADS * TILE, accurate_raw
        )
        u = ReplayPool(replay_u, indices, u_stride0, TP4_HEADS, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, TP4_HEADS, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, TP4_HEADS, record_s)
        if const_expr(self.sequence == S8_SEQUENCE_LENGTH):
            self.stage.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                u,
                rk,
                rd,
                conv,
                indices,
                accepted,
                wc,
                scale,
                lower,
                eps,
                main_rows,
                width,
                policy,
                selector,
            )
        else:
            output_rows = main_rows if main_rows <= 128 else 128
            if selector == 2:
                self.delayed.execute(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    fa,
                    norm,
                    x,
                    wqkv,
                    wa,
                    wb,
                    wbeta,
                    wz,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    u,
                    rk,
                    rd,
                    conv,
                    indices,
                    accepted,
                    wc,
                    scale,
                    lower,
                    eps,
                    main_rows,
                    width,
                    policy,
                    output_rows,
                )
            else:
                self.stage.execute(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    fa,
                    norm,
                    x,
                    wqkv,
                    wa,
                    wb,
                    wbeta,
                    wz,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    u,
                    rk,
                    rd,
                    conv,
                    indices,
                    accepted,
                    wc,
                    scale,
                    lower,
                    eps,
                    main_rows,
                    width,
                    policy,
                    output_rows,
                )

    @host
    def run(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        checkpoint,
        conv,
        indices,
        accepted,
        wc,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        conv_stride0,
        conv_length,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        main_rows,
        width,
        policy,
        selector,
        cores,
        accurate_raw,
        accurate_decay,
    ):
        self.device[cores](
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            fa,
            norm,
            x,
            wqkv,
            wa,
            wb,
            wbeta,
            wz,
            gamma,
            wo,
            alog,
            bias,
            checkpoint,
            conv,
            indices,
            accepted,
            wc,
            replay_u,
            replay_k,
            replay_decay,
            checkpoint_stride0,
            conv_stride0,
            conv_length,
            u_stride0,
            k_stride0,
            decay_stride0,
            record_s,
            scale,
            lower,
            eps,
            main_rows,
            width,
            policy,
            selector,
            accurate_raw,
            accurate_decay,
        )


class TP4VariableLengthKDAKernel:
    """当前 TP 的变长兜底内核，PA 记录池使用各自独立的步长。"""

    @kernel
    def project(self, out, x, weight):
        onchip_rewind(reset_sync_id=True)
        stage = SequenceProjectionStage()
        stage.project(out, x, weight)

    @kernel
    def convolve(self, out, raw, conv, indices, accepted, cu, weight, conv_stride0):
        onchip_rewind(reset_sync_id=True)
        stage = SequenceConvStage()
        stage.convolve(out, raw, conv, indices, accepted, cu, weight, conv_stride0)

    @kernel
    def recurrent(
        self,
        out,
        convout,
        decay,
        beta,
        z,
        gamma,
        alog,
        bias,
        checkpoint,
        indices,
        cu,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
    ):
        onchip_rewind(reset_sync_id=True)
        u = ReplayPool(replay_u, indices, u_stride0, TP4_HEADS, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, TP4_HEADS, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, TP4_HEADS, record_s)
        stage = SequenceRecurrentStage(TP4_HEADS)
        stage.replay(
            out,
            convout,
            decay,
            beta,
            z,
            gamma,
            alog,
            bias,
            checkpoint,
            indices,
            cu,
            u,
            rk,
            rd,
            checkpoint_stride0,
            scale,
            lower,
            eps,
        )

    @host
    def run(
        self,
        out,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        wc,
        conv,
        checkpoint,
        indices,
        accepted,
        cu,
        alog,
        bias,
        replay_u,
        replay_k,
        replay_decay,
        raw,
        fa,
        decay,
        beta,
        z,
        convolved,
        gated,
        conv_stride0,
        checkpoint_stride0,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        cores,
    ):
        self.project[cores](z, x, wz)
        self.convolve[2 * cores](
            convolved, raw, conv, indices, accepted, cu, wc, conv_stride0
        )
        self.recurrent[2 * cores](
            gated,
            convolved,
            decay,
            beta,
            z,
            gamma,
            alog,
            bias,
            checkpoint,
            indices,
            cu,
            replay_u,
            replay_k,
            replay_decay,
            checkpoint_stride0,
            u_stride0,
            k_stride0,
            decay_stride0,
            record_s,
            scale,
            lower,
            eps,
        )
        self.project[cores](out, gated, wo)


TP8_SHORT_SCHEDULE = {
    "main_rows": 128,
    "main_width": 384,
    "auxiliary_rows": 32,
    "main_weight_l2_cache_ctl": 4,
    "state_store_l2_cache_ctl": 0,
    "state_load_l2_cache_ctl": 2,
    "fa_width": 32,
}


TP8_LARGE_SCHEDULE = {
    "ragged_output": True,
    "ragged_main": True,
    "main_rows": 256,
    "auxiliary_rows": 0,
    "main_width": 256,
    "split_fb": True,
    "publish_rows": 128,
    "resident_output_weight": True,
    "state_update_unroll": 2,
    "main_weight_l2_cache_ctl": 0,
    "state_store_l2_cache_ctl": 4,
    "state_load_l2_cache_ctl": 4,
}


class TP8TPSchedule(
    TPScheduleFrontendOperations, TP8TPScheduleBackendOperations, TP4GridSchedule
):
    def __init__(self, num_heads, rows, auxiliary_rows, publish_rows, config):
        super().__init__(32, HIDDEN_SIZE, num_heads)
        self.main_rows = rows
        self.auxiliary_rows = auxiliary_rows
        self.publish_rows = publish_rows
        self.config = config
        self.seq_len = S8_SEQUENCE_LENGTH


class TP8ProjectionScheduleOperations(
    ProjectionScheduleOperationsFrontendOperations, TP8TPSchedule
):
    """为两种投影调度共用构造逻辑和 Conv 预取操作。"""

    def __init__(self, heads, rows, width):
        config = Schedule(
            main_rows=rows,
            main_width=width,
            ragged_main=width == 256,
            split_fb=True,
            ragged_output=True,
        )
        super().__init__(heads, rows, 32, 64, config)
        self.heads = heads
        self.projection_groups = 4 * heads // 3
        self.row_capacity = rows


class TP8AuxiliarySchedule(TP8AuxiliaryScheduleFrontendOperations, TP8TPSchedule):
    def __init__(self, heads, rows):
        super().__init__(heads, 32, rows, 64, Schedule(split_fb=True))
        self.heads = heads


class TP8S4ProjectionSchedule(
    S4ProjectionScheduleFrontendOperations, TP8ProjectionScheduleOperations
):
    """仅处理有效行的 S4 QKV/Conv 融合，复用当前 PA 数据面板消费逻辑。"""

    def __init__(self, heads):
        super().__init__(heads, 128, 384)
        self.seq_len = S4_SEQUENCE_LENGTH


class TP8S4MultiCoreSchedule(_S4StagedPipeline):
    """通用 S4 融合，支持 TP4/8/16 和 8/16/24/28/32 AIC。"""

    def __init__(self, heads, pipelined=False):
        self.main = TP8S4ProjectionSchedule(heads)
        self.auxiliary = TP8S4AuxiliarySchedule(heads, pipelined=pipelined)
        self.backend = (
            S4PipelinedBackendSchedule(heads)
            if pipelined
            else TP4S4BackendSchedule(heads)
        )


class TP8PhaseSchedule(TP8TPSchedule):
    """保留经过实测的输出分块几何，并共用运行时缓存控制。"""

    @jit
    def _configure_recurrent(self, _recurrent_stage):
        _recurrent_stage.store_policy = self.store_policy
        _recurrent_stage.load_policy = self.load_policy


class _TP8S8Core32Stage:
    def __init__(self):
        self.head_backend = TP8HeadPanelBackend(TP8_HEADS)
        self.main384 = Core32ProjectionSchedule(TP8_HEADS, 128, 384)
        self.main256 = Core32ProjectionSchedule(TP8_HEADS, 256, 256)
        self.aux0 = TP8TPSchedule(TP8_HEADS, 128, 32, 64, Schedule(fa_width=32))
        self.aux1 = TP8TPSchedule(
            TP8_HEADS,
            128,
            64,
            64,
            Schedule(fa_width=128, auxiliary_weight_l2_cache_ctl=0),
        )
        self.aux64 = TP8TPSchedule(TP8_HEADS, 128, 64, 64, Schedule(fa_width=64))
        self.back0 = TP8PhaseSchedule(
            TP8_HEADS, 128, 64, 64, Schedule(publish_rows=64, ragged_output=False)
        )
        self.back1 = TP8PhaseSchedule(
            TP8_HEADS, 128, 64, 256, Schedule(publish_rows=256, ragged_output=True)
        )
        self.back2 = TP8PhaseSchedule(
            TP8_HEADS,
            128,
            64,
            64,
            Schedule(publish_rows=64, ragged_output=True, output_kc=256),
        )

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
    ):
        mb = replay_decay.indices.shape[0] // 8
        main_policy = dtypes.int32(4)
        if mb > 4:
            main_policy = dtypes.int32(1)
        store_policy = dtypes.int32(0)
        if mb >= 4:
            store_policy = dtypes.int32(4)
        load_policy = dtypes.int32(2)
        if mb >= 16:
            load_policy = dtypes.int32(4)
        main_rows = 256
        if mb == 1:
            main_rows = 64
        elif mb == 2:
            main_rows = 128
        use384 = mb <= 2
        # 对齐的 M128/N384 分块能够形成完整的投影波次。
        if mb >= 8 and out.shape[0] % 128 == 0:
            main_rows = 128
            use384 = True
        self.main256.main_policy = main_policy
        self.main256.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            TP8_HEADS,
            main_rows,
            not use384,
        )
        self.main384.main_policy = main_policy
        self.main384.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            TP8_HEADS,
            main_rows,
            use384,
        )
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.aux0.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        elif x.shape[0] // 64 * 3 <= get_block_num():
            # 每个核只保留一种 Beta/Fa 分块几何；N64 能占满更多波次位置。
            self.aux64.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        else:
            self.aux1.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.aux0.frontend_fb(decay, fa, wb)
        elif x.shape[0] // 64 * 3 <= get_block_num():
            self.aux64.frontend_fb(decay, fa, wb)
        else:
            self.aux1.frontend_fb(decay, fa, wb)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.head_backend.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        elif mb == 2 or mb == 5:
            self.back0.store_policy = store_policy
            self.back0.load_policy = load_policy
            self.back0.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        # 前面的分支已处理 1、2、5，因此这里只剩 3、4、6、7。
        elif mb < 8:
            self.back1.store_policy = store_policy
            self.back1.load_policy = load_policy
            self.back1.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        else:
            self.back2.store_policy = store_policy
            self.back2.load_policy = load_policy
            self.back2.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )


class _TP8S8Core32TailStage(_TP8S8Core32Stage):
    def __init__(self):
        super().__init__()
        self.heads = TP8_HEADS
        self.main384 = TailMaskedCore32ProjectionSchedule(TP8_HEADS, 128, 384)

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
    ):
        mb = replay_decay.indices.shape[0] // 8
        store_policy = dtypes.int32(0)
        load_policy = dtypes.int32(2)
        # TP8 的 N384 投影有 16 个列任务，两个 M 组占满一个 32 Cube 波次；
        # 将每组的有效行数向上对齐到 64。
        target_m_groups = 32 // (4 * self.heads // 3)
        main_rows = ceil_div(x.shape[0], target_m_groups * 64) * 64
        self.main384.main_policy = dtypes.int32(4)
        self.main384.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            TP8_HEADS,
            main_rows,
        )
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.aux0.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        elif x.shape[0] // 64 * 3 <= get_block_num():
            # 每个核只保留一种 Beta/Fa 分块几何；N64 能占满更多波次位置。
            self.aux64.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        else:
            self.aux1.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.aux0.frontend_fb(decay, fa, wb)
        elif x.shape[0] // 64 * 3 <= get_block_num():
            self.aux64.frontend_fb(decay, fa, wb)
        else:
            self.aux1.frontend_fb(decay, fa, wb)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if mb == 1:
            self.head_backend.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        elif mb == 2 or mb == 5:
            self.back0.store_policy = store_policy
            self.back0.load_policy = load_policy
            self.back0.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        # 前面的分支已处理 1、2、5，因此这里只剩 3、4、6、7。
        elif mb < 8:
            self.back1.store_policy = store_policy
            self.back1.load_policy = load_policy
            self.back1.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )
        else:
            self.back2.store_policy = store_policy
            self.back2.load_policy = load_policy
            self.back2.backend(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                scale,
                lower,
                eps,
            )


class S4ParallelSchedule:
    """TP8、B8/B16、32 AIC：30 个 QKV/Z 执行核，另两个核分别计算 beta 和融合的 Fa/Fb。"""

    def __init__(self):
        self.main = TP8S4ProjectionSchedule(TP8_HEADS)
        self.auxiliary = TP8S4AuxiliarySchedule(TP8_HEADS)
        self.backend = TP4S4BackendSchedule(TP8_HEADS)

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        indices,
        accepted,
        wc,
        scale,
        lower,
        eps,
        rows,
        width,
        policy,
        output_rows,
    ):
        if get_block_idx() < 30:
            for head in range(get_block_idx(), 48, 30):
                self.main.panel(
                    q,
                    k,
                    v,
                    gate,
                    x,
                    wqkv,
                    wz,
                    conv,
                    indices,
                    accepted,
                    wc,
                    0,
                    head,
                    rows,
                    policy,
                )
        cube_sync_all()
        vec_sync_all()
        if get_block_idx() == 30:
            self.auxiliary.beta_panel(beta, x, wbeta, 0, rows, policy)
        if get_block_idx() == 31:
            self.auxiliary.decay_panel(decay, x, wa, wb, 0, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.backend.execute(
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            gamma,
            wo,
            alog,
            bias,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
            eps,
            output_rows,
            policy,
        )


class TP8DelayedProjectionSchedule:
    """对完整的大行块重叠执行门控投影与递推。"""

    def __init__(self, heads):
        self.heads = heads
        self.zrows = 128
        self.publish_rows = 256
        self.window_rows = self.publish_rows
        self.minimum_rows = 4 * self.publish_rows
        self.main = Core32ProjectionSchedule(heads, 128, 384)
        self.main.projection_groups = heads
        self.aux = TP8TPSchedule(heads, 128, 64, 128, Schedule(publish_rows=128))
        self.aux64 = TP8TPSchedule(
            heads, 128, 128, 128, Schedule(publish_rows=128, fa_width=64)
        )
        self.aux128 = TP8TPSchedule(
            heads,
            128,
            128,
            128,
            Schedule(publish_rows=128, auxiliary_weight_l2_cache_ctl=0),
        )

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        self.main.main_policy = dtypes.int32(1)
        self.main.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            self.heads,
            128,
        )
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        if x.shape[0] // 128 * 3 <= get_block_num():
            self.aux64.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        else:
            self.aux128.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if x.shape[0] // 128 * 3 <= get_block_num():
            self.aux64.frontend_fb(decay, fa, wb)
        else:
            self.aux128.frontend_fb(decay, fa, wb)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        context = _RawRecurrentContext(
            q,
            k,
            v,
            initial.accurate_decay,
            beta,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
        )
        for chunk in range(x.shape[0] // self.publish_rows):
            config = Schedule(state_load_l2_cache_ctl=4)
            recurrent = RecurrentStage(128, self.heads, config, state_buffer_depth=2)
            output_gate = OutputGateStage()
            raw = Buffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout(
                    (ceil_div(self.publish_rows // 8 * self.heads, 64) * 8, 128)
                ),
            )
            gate_context = _RawGateContext(q, gate, raw, eps)
            zstage = _DelayedGateProjection(self.zrows, 0, 0, row_capacity=self.zrows)
            recurrent.load_gate_params(alog, bias)
            output_gate.load_gamma(gamma)
            for ztask in range(
                get_block_idx(),
                self.publish_rows // self.zrows * self.heads,
                get_block_num(),
            ):
                head = ztask // (self.publish_rows // self.zrows)
                rowpart = ztask % (self.publish_rows // self.zrows)
                zstage.project_to_gm(
                    tile_slice(
                        x,
                        (self.zrows, HIDDEN_SIZE),
                        (chunk * (self.publish_rows // self.zrows) + rowpart, 0),
                    ),
                    wz,
                    tile_slice(
                        gate,
                        (self.zrows, 128),
                        (chunk * (self.publish_rows // self.zrows) + rowpart, head),
                    ),
                    head,
                    56,
                )
            first = get_block_idx() * 2 + get_subblock_id()
            if first < self.publish_rows // 8 * self.heads:
                first_batch = chunk * (self.publish_rows // 8) + first % (
                    self.publish_rows // 8
                )
                first_head = first // (self.publish_rows // 8)
                recurrent.load_state(
                    tile_slice(
                        _checkpoint_head(initial, first_batch, first_head),
                        (128, 128),
                        (0, 0),
                    )
                )
            for item in range(first, self.publish_rows // 8 * self.heads, 64):
                batch = chunk * (self.publish_rows // 8) + item % (
                    self.publish_rows // 8
                )
                head = item // (self.publish_rows // 8)
                next_item = item + 64
                next_batch = chunk * (self.publish_rows // 8) + next_item % (
                    self.publish_rows // 8
                )
                next_head = next_item // (self.publish_rows // 8)
                _run_raw_recurrent_item(
                    recurrent,
                    context,
                    batch,
                    head,
                    raw,
                    item,
                    checkpoint_prefetched=True,
                    next_batch=next_batch,
                    next_head=next_head,
                    next_valid=next_item < self.publish_rows // 8 * self.heads,
                )
            cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
            cube_sync_block_wait(PIPE.S, 0, mode=0)
            cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=2)
            vec_sync_block_wait(PIPE.S, 1, mode=2)
            for item in range(first, self.publish_rows // 8 * self.heads, 64):
                batch = chunk * (self.publish_rows // 8) + item % (
                    self.publish_rows // 8
                )
                head = item // (self.publish_rows // 8)
                _store_raw_gated_item(output_gate, gate_context, batch, head, item)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            onchip_rewind(reset_sync_id=True)
            output = OutputProjectionStage(
                224,
                self.publish_rows,
                kc=256,
                tiles_per_core=1,
                resident_weight=False,
                row_capacity=self.publish_rows,
                num_heads=self.heads,
            )
            output.project(out, q, wo, self.heads, chunk)
            cube_sync_all()
            vec_sync_all()
            onchip_rewind(reset_sync_id=True)


class _HeadStreamSchedule(TP8DelayedProjectionSchedule):
    def __init__(self, warmup):
        super().__init__(TP8_HEADS)
        self.warmup = warmup

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        self.main.main_policy = dtypes.int32(1)
        self.main.main(
            q,
            k,
            v,
            gate,
            x,
            wqkv,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
            self.heads,
            128,
        )
        # Fa/Beta 读取 X，并通过 Fixpipe 发布结果，不复用 Conv 的 UB。
        # 在此处排空 Cube 存储；辅助计算后的网格屏障
        # 还会确保 Conv 完成，后端才能消费 Q/K/V。
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        if x.shape[0] // 128 * 3 <= get_block_num():
            self.aux64.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        else:
            self.aux128.frontend_auxiliary(decay, beta, fa, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        if x.shape[0] // 128 * 3 <= get_block_num():
            self.aux64.frontend_fb(decay, fa, wb)
        else:
            self.aux128.frontend_fb(decay, fa, wb)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        context = _RawRecurrentContext(
            q,
            k,
            v,
            initial.accurate_decay,
            beta,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
        )
        recurrent = RecurrentStage(
            128, TP8_HEADS, Schedule(state_load_l2_cache_ctl=4), state_buffer_depth=2
        )
        output_gate = OutputGateStage()
        raw = Buffer(MemLoc.UB, dtypes.float32, make_layout((48, 128)))
        gate_context = _RawGateContext(q, gate, raw, eps)
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        recurrent.load_state(
            tile_slice(
                _checkpoint_head(initial, first % 32, first // 32), (128, 128), (0, 0)
            )
        )
        for chunk in range(x.shape[0] // 256):
            # 持久 UB 对象保持有效。Z 和 OProj 仅复用 Cube 存储，
            # 并在每次重置前沿用原有的显式排空操作。
            zstage = _DelayedGateProjection(128, 0, 0, row_capacity=128)
            for ztask in range(get_block_idx(), GATE_PROJECTION_TASK_COUNT, 32):
                head = ztask // 2
                rowpart = ztask % 2
                zstage.project_to_gm(
                    tile_slice(x, (128, HIDDEN_SIZE), (chunk * 2 + rowpart, 0)),
                    wz,
                    tile_slice(gate, (128, 128), (chunk * 2 + rowpart, head)),
                    head,
                    56,
                )
            for row in range(self.warmup):
                _head_stream_persistent_raw_round(
                    recurrent, context, raw, first, row, chunk
                )
            # 输出缓冲区取得 Z 的 Cube 地址前，Z 存储的生命周期已经结束。
            cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
            cube_sync_block_wait(PIPE.S, 0, mode=0)
            cube_sync_all()
            onchip_rewind(reset_sync_id=True)
            output = _HeadStreamOutput()
            output.prefetch_panel(wo, 0)
            cube_sync_block_arrive(PIPE.FIXPIPE, 15, mode=2)
            vec_sync_block_wait(PIPE.S, 15, mode=2)
            for row in range(HEAD_STREAM_ROUNDS):
                if row >= self.warmup:
                    _head_stream_persistent_raw_round(
                        recurrent, context, raw, first, row, chunk
                    )
                item = first + row * 64
                batch = chunk * 32 + item % 32
                head = item // 32
                _store_raw_gated_item(output_gate, gate_context, batch, head, item)
                # 使用独立标志，避免在同一个数据块内执行反向确认。
                vec_sync_block_arrive(PIPE.MTE3, 2 + row, mode=2)
                cube_sync_block_wait(PIPE.S, 2 + row, mode=2)
                cube_sync_block_arrive(PIPE.FIXPIPE, 8 + row, mode=0)
                cube_sync_block_wait(PIPE.S, 8 + row, mode=0)
                output.accumulate_panel(q, row, chunk)
                if row + 1 < HEAD_STREAM_ROUNDS:
                    output.prefetch_panel(wo, row + 1)
            output.finish(out, chunk)
            cube_sync_all()
            vec_sync_all()
            onchip_rewind(reset_sync_id=True)


@jit
def _execute_stream_prefix_tail(
    stages,
    out,
    q,
    k,
    v,
    decay,
    gate,
    beta,
    fa,
    norm,
    x,
    wqkv,
    wa,
    wb,
    wbeta,
    wz,
    gamma,
    wo,
    alog,
    bias,
    initial,
    replay_u,
    replay_k,
    replay_decay,
    conv,
    state_indices,
    conv_accepted,
    wc,
    scale,
    lower,
    eps,
    plan_rows,
    plan_width,
    plan_policy,
):
    prefix_rows = x.shape[0] // 256 * 256
    prefix_batch = prefix_rows // 8
    total_rows = x.shape[0]
    total_batch = initial.indices.shape[0]
    stages.delayed.execute(
        out[0:prefix_rows, None],
        q[0:prefix_rows, None],
        k[0:prefix_rows, None],
        v[0:prefix_rows, None],
        decay[0:prefix_rows, None],
        gate[0:prefix_rows, None],
        beta[0:prefix_rows, None],
        fa[0:prefix_rows, None],
        norm,
        x[0:prefix_rows, None],
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        _checkpoint_batches(initial, 0, prefix_batch),
        replay_batches(replay_u, 0, prefix_batch),
        replay_batches(replay_k, 0, prefix_batch),
        replay_batches(replay_decay, 0, prefix_batch),
        conv,
        state_indices[0:prefix_batch,],
        conv_accepted[0:prefix_batch,],
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    )
    # 必须先完成前缀的全部发布与消费，尾部存储或同步标识
    # 才能复用任何片上地址或通道。
    global_sync_all(flag_ids=(1, 14, 0))
    cube_sync_all()
    vec_sync_all()
    onchip_rewind(reset_sync_id=True)
    # Conv 池始终使用全局坐标。只有 batch 索引向量以及
    # 互不重叠的状态、replay 和行视图会偏移到尾部。
    stages.tail32.execute(
        out[prefix_rows:total_rows, None],
        q[prefix_rows:total_rows, None],
        k[prefix_rows:total_rows, None],
        v[prefix_rows:total_rows, None],
        decay[prefix_rows:total_rows, None],
        gate[prefix_rows:total_rows, None],
        beta[prefix_rows:total_rows, None],
        fa[prefix_rows:total_rows, None],
        norm,
        x[prefix_rows:total_rows, None],
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        _checkpoint_batches(initial, prefix_batch, total_batch),
        replay_batches(replay_u, prefix_batch, total_batch),
        replay_batches(replay_k, prefix_batch, total_batch),
        replay_batches(replay_decay, prefix_batch, total_batch),
        _ConvPool(
            conv.data,
            conv.stride0,
            conv.length,
            conv.width,
            conv.accurate_raw[prefix_rows:total_rows, None],
        ),
        state_indices[prefix_batch:total_batch,],
        conv_accepted[prefix_batch:total_batch,],
        wc,
        scale,
        lower,
        eps,
    )
    global_sync_all()


def _tp8_intermediate_schedule(num_heads, model_batch):
    rows = model_batch * (MODEL_HEAD_COUNT // num_heads) * S8_SEQUENCE_LENGTH
    if rows <= 128 and rows & (rows - 1) == 0:
        options = dict(main_rows=128, main_width=384)
    else:
        # 递推工作占主导时，TP8 按较小的数据块发布结果。
        publish_rows = 256
        publish_rows = 128 if model_batch >= 8 else (64 if model_batch == 5 else 256)
        options = dict(
            main_rows=256,
            main_width=256,
            ragged_main=True,
            ragged_output=True,
            publish_rows=publish_rows,
        )
    physical_rows = _tile_rows(rows, options["main_rows"], options.get("ragged_main"))
    options["main_weight_l2_cache_ctl"] = 0 if rows > physical_rows else 4
    options["state_store_l2_cache_ctl"] = 4 if model_batch >= 4 else 0
    return Schedule(**options)


def _tp8_default_schedule(num_heads, model_batch):
    """根据 TP 和模型 batch 选择分块与缓存策略。"""
    if num_heads != TP8_HEADS or not 1 <= model_batch <= 32:
        raise ValueError(f"requires N={TP8_HEADS} and model_batch=1..32")
    if model_batch > 16:
        return Schedule(**TP8_LARGE_SCHEDULE)
    if model_batch in (1, 16):
        options = TP8_SHORT_SCHEDULE if model_batch == 1 else TP8_LARGE_SCHEDULE
        return Schedule(**options)
    return _tp8_intermediate_schedule(num_heads, model_batch)


def _tp8_select_main(rows, heads, cores):
    if heads != TP8_HEADS or cores not in VALID_CORES:
        raise ValueError("unsupported heads or AIC quota")
    return min(
        range(len(MAIN_TEMPLATES)),
        key=lambda i: main_geometry(rows, heads, cores, *MAIN_TEMPLATES[i]).cost,
    )


def _tp8_core32_geometry(model_batch, heads, rows):
    config = _tp8_default_schedule(heads, model_batch)
    aligned = model_batch >= 8 and rows % 128 == 0
    return dict(
        main_m=128
        if aligned
        else _tile_rows(rows, config.main_rows, config.ragged_main),
        main_n=384 if aligned else config.main_width,
        auxiliary_rows=math.gcd(
            rows,
            config.auxiliary_rows or (MODEL_HEAD_COUNT // heads) * S8_SEQUENCE_LENGTH,
        ),
        publish_rows=_tile_rows(rows, config.publish_rows, config.ragged_output),
    )


def _tp8_core32_plan(model_batch, tp):
    rows = model_batch * tp * S8_SEQUENCE_LENGTH
    heads = MODEL_HEAD_COUNT // tp
    geometry = _tp8_core32_geometry(model_batch, heads, rows)
    lanes = 64
    return dict(
        model_batch=model_batch,
        tp=tp,
        cores=32,
        rows=rows,
        heads=heads,
        main_id=-1,
        **geometry,
        main_geometry=main_geometry(
            rows, heads, 32, geometry["main_m"], geometry["main_n"]
        ),
        kda_waves=(MODEL_HEAD_COUNT * model_batch + lanes - 1) // lanes,
        vector_lanes=lanes,
        output_width=256,
        output="nb_head_panels" if model_batch == 1 else ("measured32_stream"),
        mode="specialized32",
    )


def _tp8_output_policy(rows, heads, cores):
    if rows * heads == 768:
        return "nb_head_panels"
    if rows in (32, 64):
        return "short32" if rows == 32 else "short64"
    return "input128_r128"


def _tp8_plan(model_batch, tp, cores):
    if tp != TP8_TP_DEGREE or not 1 <= model_batch <= 32 or cores not in VALID_CORES:
        raise ValueError("requires TP8, model batch 1..32, AIC 8/16/24/28/32")
    rows = model_batch * tp * S8_SEQUENCE_LENGTH
    heads = MODEL_HEAD_COUNT // tp
    if cores == 32:
        return _tp8_core32_plan(model_batch, tp)
    i = _tp8_select_main(rows, heads, cores)
    m, n = MAIN_TEMPLATES[i]
    return dict(
        model_batch=model_batch,
        tp=tp,
        cores=cores,
        rows=rows,
        heads=heads,
        main_id=i,
        main_m=m,
        main_n=n,
        main_geometry=main_geometry(rows, heads, cores, m, n),
        kda_waves=(MODEL_HEAD_COUNT * model_batch + 2 * cores - 1) // (2 * cores),
        auxiliary_rows=math.gcd(rows, 128),
        output=_tp8_output_policy(rows, heads, cores),
        weight_policy=0 if rows > m else 4,
    )


class TP8S8DefaultStage:
    def __init__(self, heads):
        self.heads = heads
        self.main384 = MultiCoreProjectionSchedule(heads, 128, 384)
        self.main256 = MultiCoreProjectionSchedule(heads, 256, 256)
        self.aux32 = TP8AuxiliarySchedule(heads, 32)
        self.aux64 = TP8AuxiliarySchedule(heads, 64)
        self.aux128 = TP8AuxiliarySchedule(heads, 128)
        self.backend = TP8BackendSchedule(heads)
        self.head_backend = TP8HeadPanelBackend(heads)
        self.fast32 = _TP8S8Core32Stage()

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        if get_block_num() == 32:
            self.fast32.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog[0 : self.heads,],
                bias[0 : self.heads, None],
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
            )
        else:
            n = self.heads
            logical_rows = plan_rows
            if plan_width == 384:
                self.main384.main(
                    q,
                    k,
                    v,
                    gate,
                    x,
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    n,
                    logical_rows,
                    plan_policy,
                )
            else:
                self.main256.main(
                    q,
                    k,
                    v,
                    gate,
                    x,
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    n,
                    logical_rows,
                    plan_policy,
                )
            global_sync_all()
            onchip_rewind(reset_sync_id=True)
            if x.shape[0] % 128 == 0:
                self.aux128.execute(decay, beta, fa, x, wa, wb, wbeta)
            elif x.shape[0] % 64 == 0:
                self.aux64.execute(decay, beta, fa, x, wa, wb, wbeta)
            else:
                self.aux32.execute(decay, beta, fa, x, wa, wb, wbeta)
            global_sync_all()
            onchip_rewind(reset_sync_id=True)
            if replay_decay.indices.shape[0] * n == MODEL_HEAD_COUNT:
                self.head_backend.backend(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    gamma,
                    wo,
                    alog[0:n,],
                    bias[0:n, None],
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    scale,
                    lower,
                    eps,
                )
            else:
                self.backend.backend(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    scale,
                    lower,
                    eps,
                    wbeta,
                )
        global_sync_all()


class TP8S8Dispatch(TP8S8DefaultStage):
    def __init__(self, heads):
        super().__init__(heads)
        self.delayed = _HeadStreamSchedule(5)
        self.tail32 = _TP8S8Core32TailStage()

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
        prefer_stream,
    ):
        q = q.view((q.shape[0], self.heads * TILE))
        k = k.view((k.shape[0], self.heads * TILE))
        v = v.view((v.shape[0], self.heads * TILE))
        decay = decay.view((decay.shape[0], self.heads * TILE))
        gate = gate.view((gate.shape[0], self.heads * TILE))
        beta = beta.view(
            (beta.shape[0], ((self.heads + BETA_TILE - 1) // BETA_TILE) * BETA_TILE)
        )
        alog = alog.view((self.heads,))
        bias = bias.view((self.heads, TILE))
        conv = _ConvPool(
            conv.data,
            conv.stride0,
            conv.length,
            3 * self.heads * TILE,
            conv.accurate_raw,
        )
        wc = wc.view((3, self.heads, CONV_KERNEL, TILE))
        if (
            get_block_num() == 32
            and x.shape[0] % self.delayed.window_rows == 0
            and (x.shape[0] >= self.delayed.minimum_rows or (prefer_stream != 0))
        ):
            self.delayed.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )
            global_sync_all(flag_ids=(1, 14, 0))
        elif (
            get_block_num() == 32
            and x.shape[0] >= self.delayed.minimum_rows
            and (x.shape[0] % 256 != 0)
        ):
            _execute_stream_prefix_tail(
                self,
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )
        else:
            TP8S8DefaultStage.execute(
                self,
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )


def _tp8_prefer_streamed_main(rows, heads, cores, normal_m, normal_n):
    """比较完整主投影波次的开销；此结果用于运行时选择，不作为编译缓存键。"""
    if cores != 32 or rows % 256:
        return 0
    normal_jobs = ((rows + normal_m - 1) // normal_m) * (4 * heads * TILE // normal_n)
    delayed_jobs = rows // 128 * heads
    normal_work = ((normal_jobs + cores - 1) // cores) * normal_m * normal_n
    delayed_work = ((delayed_jobs + cores - 1) // cores) * 128 * 384
    return int(delayed_work < normal_work)


def _tp8_plan_s4(model_batch, tp, cores):
    """采用保守的源码调度策略，不依赖实测性能查找表。"""
    if tp != TP8_TP_DEGREE or not 1 <= model_batch <= 32 or cores not in VALID_CORES:
        raise ValueError("S4 requires TP8, model batch 1..32, AIC 8/16/24/28/32")
    total_rows = model_batch * tp * S4_SEQUENCE_LENGTH
    main_rows = math.gcd(total_rows, 128)
    width = 384
    if total_rows >= 512 and total_rows % 256 == 0:
        main_rows, width = 256, 256
    template = 0
    if cores == 32:
        if model_batch <= 2:
            main_rows, width, template = total_rows, 128, 1
        elif model_batch in (8, 16):
            main_rows, width, template = 128, 384, 2

    return dict(
        main_rows=main_rows,
        width=width,
        policy=4 if total_rows <= 128 else 0,
        template=template,
    )


class TP8TPOptimizedKernel:
    """实现单个 TP，为 S8 和 S4 分别缓存编译后的可调用对象。"""

    def __init__(self, sequence):
        self.sequence = sequence
        if sequence == S8_SEQUENCE_LENGTH:
            self.stage = TP8S8Dispatch(TP8_HEADS)
        else:
            self.stage = TP8S4MultiCoreSchedule(TP8_HEADS)
            self.parallel = S4ParallelSchedule()
            self.pipelined = TP8S4MultiCoreSchedule(TP8_HEADS, pipelined=True)

    @kernel(
        profile=cannbotdsl.ProfileSpec(
            name="mega_recurrent_kda_tp8",
            op_type="MegaRecurrentKDA",
            outputs=("out", "replay_u", "replay_k", "replay_decay"),
            inouts=("conv",),
        )
    )
    def device(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        checkpoint,
        conv,
        indices,
        accepted,
        wc,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        conv_stride0,
        conv_length,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        main_rows,
        width,
        policy,
        selector,
        accurate_raw,
        accurate_decay,
    ):
        initial = _CheckpointPool(
            checkpoint, indices, checkpoint_stride0, accurate_decay
        )
        conv = _ConvPool(
            conv, conv_stride0, conv_length, 3 * TP8_HEADS * TILE, accurate_raw
        )
        u = ReplayPool(replay_u, indices, u_stride0, TP8_HEADS, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, TP8_HEADS, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, TP8_HEADS, record_s)
        if const_expr(self.sequence == S8_SEQUENCE_LENGTH):
            self.stage.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                u,
                rk,
                rd,
                conv,
                indices,
                accepted,
                wc,
                scale,
                lower,
                eps,
                main_rows,
                width,
                policy,
                selector,
            )
        else:
            output_rows = main_rows if main_rows <= 128 else 128
            if selector == 1:
                self.parallel.execute(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    fa,
                    norm,
                    x,
                    wqkv,
                    wa,
                    wb,
                    wbeta,
                    wz,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    u,
                    rk,
                    rd,
                    conv,
                    indices,
                    accepted,
                    wc,
                    scale,
                    lower,
                    eps,
                    main_rows,
                    width,
                    policy,
                    output_rows,
                )
            elif selector == 2:
                self.pipelined.execute(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    fa,
                    norm,
                    x,
                    wqkv,
                    wa,
                    wb,
                    wbeta,
                    wz,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    u,
                    rk,
                    rd,
                    conv,
                    indices,
                    accepted,
                    wc,
                    scale,
                    lower,
                    eps,
                    main_rows,
                    width,
                    policy,
                    output_rows,
                )
            else:
                self.stage.execute(
                    out,
                    q,
                    k,
                    v,
                    decay,
                    gate,
                    beta,
                    fa,
                    norm,
                    x,
                    wqkv,
                    wa,
                    wb,
                    wbeta,
                    wz,
                    gamma,
                    wo,
                    alog,
                    bias,
                    initial,
                    u,
                    rk,
                    rd,
                    conv,
                    indices,
                    accepted,
                    wc,
                    scale,
                    lower,
                    eps,
                    main_rows,
                    width,
                    policy,
                    output_rows,
                )

    @host
    def run(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        checkpoint,
        conv,
        indices,
        accepted,
        wc,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        conv_stride0,
        conv_length,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        main_rows,
        width,
        policy,
        selector,
        cores,
        accurate_raw,
        accurate_decay,
    ):
        self.device[cores](
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            fa,
            norm,
            x,
            wqkv,
            wa,
            wb,
            wbeta,
            wz,
            gamma,
            wo,
            alog,
            bias,
            checkpoint,
            conv,
            indices,
            accepted,
            wc,
            replay_u,
            replay_k,
            replay_decay,
            checkpoint_stride0,
            conv_stride0,
            conv_length,
            u_stride0,
            k_stride0,
            decay_stride0,
            record_s,
            scale,
            lower,
            eps,
            main_rows,
            width,
            policy,
            selector,
            accurate_raw,
            accurate_decay,
        )


class TP8VariableLengthKDAKernel:
    """当前 TP 的变长兜底内核，PA 记录池使用各自独立的步长。"""

    @kernel
    def project(self, out, x, weight):
        onchip_rewind(reset_sync_id=True)
        stage = SequenceProjectionStage()
        stage.project(out, x, weight)

    @kernel
    def convolve(self, out, raw, conv, indices, accepted, cu, weight, conv_stride0):
        onchip_rewind(reset_sync_id=True)
        stage = SequenceConvStage()
        stage.convolve(out, raw, conv, indices, accepted, cu, weight, conv_stride0)

    @kernel
    def recurrent(
        self,
        out,
        convout,
        decay,
        beta,
        z,
        gamma,
        alog,
        bias,
        checkpoint,
        indices,
        cu,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
    ):
        onchip_rewind(reset_sync_id=True)
        u = ReplayPool(replay_u, indices, u_stride0, TP8_HEADS, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, TP8_HEADS, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, TP8_HEADS, record_s)
        stage = SequenceRecurrentStage(TP8_HEADS)
        stage.replay(
            out,
            convout,
            decay,
            beta,
            z,
            gamma,
            alog,
            bias,
            checkpoint,
            indices,
            cu,
            u,
            rk,
            rd,
            checkpoint_stride0,
            scale,
            lower,
            eps,
        )

    @host
    def run(
        self,
        out,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        wc,
        conv,
        checkpoint,
        indices,
        accepted,
        cu,
        alog,
        bias,
        replay_u,
        replay_k,
        replay_decay,
        raw,
        fa,
        decay,
        beta,
        z,
        convolved,
        gated,
        conv_stride0,
        checkpoint_stride0,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        cores,
    ):
        self.project[cores](z, x, wz)
        self.convolve[2 * cores](
            convolved, raw, conv, indices, accepted, cu, wc, conv_stride0
        )
        self.recurrent[2 * cores](
            gated,
            convolved,
            decay,
            beta,
            z,
            gamma,
            alog,
            bias,
            checkpoint,
            indices,
            cu,
            replay_u,
            replay_k,
            replay_decay,
            checkpoint_stride0,
            u_stride0,
            k_stride0,
            decay_stride0,
            record_s,
            scale,
            lower,
            eps,
        )
        self.project[cores](out, gated, wo)


class TP16GridSchedule(
    TP16GridScheduleFrontendOperations, GridScheduleBackendOperations
):
    """使用编译期分支配置，并共用一套 KDA 波次调度。"""

    def __init__(self, block_num: int, hidden_size: int, num_heads: int):
        self.state_store_l2_cache_ctl = 0
        self.block_num = block_num
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.projection_groups = 8
        self.batch_tile = BATCH_TILE
        self.qkv_block_count = 3 * num_heads
        self.output_gate_block_count = num_heads
        self.beta_block = self.qkv_block_count + num_heads
        self.low_rank_block = self.beta_block + 1
        self.recurrent_wave_count = 2
        self.output_tiles_per_core = {8: 7, 16: 2, 24: 2, 28: 1, 32: 1}[block_num]
        self.output_tile_width = {8: 128, 16: 256, 24: 256, 28: 256, 32: 256}[block_num]
        activation_channel_bytes = 2 * 64 * OPROJ_KC * 2
        self.output_weight_resident = (
            self.output_tile_width * self.output_tiles_per_core * num_heads * TILE * 2
            + activation_channel_bytes
            <= cannbotdsl.get_mem_size("l1")
        )
        self.projection_input_l2_cache_ctl = 0
        self.projection_weight_l2_cache_ctl = 4
        self.batch_groups: object
        self.seq_len: object
        self.row_count: object
        self.projection_row_block: object
        self.recurrent_wave_rows: object

    @jit
    def _execute_compact_s8(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        initial_state_gm: Tensor,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        self.batch_groups = replay_decay_gm.indices.shape[0] // BATCH_TILE
        self.seq_len = S8_SEQUENCE_LENGTH
        self.row_count = replay_decay_gm.indices.shape[0] * self.seq_len
        self.projection_row_block = self.row_count // 2
        recurrent_items = replay_decay_gm.indices.shape[0] * self.num_heads
        recurrent_lanes = min(48, 2 * self.block_num)
        recurrent_wave_count = ceil_div(recurrent_items, recurrent_lanes)
        self.recurrent_wave_rows = self.row_count // recurrent_wave_count
        if const_expr(self.block_num == 24):
            self._frontend_24_single_group(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                fa_workspace_gm,
                norm_exchange_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
        else:
            self._frontend_large(
                out_gm,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                fa_workspace_gm,
                norm_exchange_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                output_norm_weight_gm,
                o_proj_weight_gm,
                a_log_gm,
                dt_bias_gm,
                initial_state_gm,
                replay_u_gm,
                replay_k_gm,
                replay_decay_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                scale_value,
                lower_bound,
                rms_norm_eps,
            )
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        state_row_block = TILE
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        if const_expr(self.block_num == 32):
            group_idx = block_idx // 16
            local_idx = block_idx % 16
            block_idx = (
                group_idx * RECURRENT_CORES_PER_CORE_GROUP + local_idx
                if local_idx < RECURRENT_CORES_PER_CORE_GROUP
                else COMPACT_RECURRENT_CORE_COUNT
                + group_idx * 4
                + local_idx
                - RECURRENT_CORES_PER_CORE_GROUP
            )
        recurrent_active = block_idx < recurrent_lanes // 2
        recurrent_stage = RecurrentKDAStage(
            state_row_block,
            self.num_heads,
            state_key_unroll=KDA_STATE_KEY_UNROLL,
            state_update_unroll=KDA_STATE_UPDATE_UNROLL,
        )
        output_gate_stage = OutputGateStage()
        output_projection_stage = _TiledOutputProjectionStage(
            self.output_tile_width,
            self.recurrent_wave_rows,
            kc=OPROJ_KS * 2 if self.output_weight_resident else OPROJ_KC,
            tiles_per_core=self.output_tiles_per_core,
            resident_input=False,
            resident_weight=self.output_weight_resident,
            tile_stride=24 if self.block_num == 24 else 1,
            l0c_depth=2 if self.block_num == 24 else 1,
        )
        if const_expr(self.output_weight_resident):
            output_projection_stage.prefetch_weight(o_proj_weight_gm)
        if recurrent_active:
            recurrent_stage.load_gate_params(a_log_gm, dt_bias_gm)
            output_gate_stage.load_gamma(output_norm_weight_gm)
        for wave in range(recurrent_wave_count):
            item_start = wave * recurrent_lanes + block_idx * 2 + subblock_idx
            item_count = 1
            if recurrent_active:
                run_recurrent_wave(
                    recurrent_stage,
                    output_gate_stage,
                    q_workspace_gm,
                    k_workspace_gm,
                    v_workspace_gm,
                    initial_state_gm.accurate_decay,
                    output_gate_workspace_gm,
                    beta_workspace_gm,
                    initial_state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    item_start,
                    item_start + item_count,
                    replay_decay_gm.indices.shape[0],
                    self.seq_len,
                    self.num_heads,
                    state_row_block,
                    scale_value,
                    lower_bound,
                    rms_norm_eps,
                    nb_group_batch=self.recurrent_wave_rows // self.seq_len,
                )
            publish_batch_group_vector_grid_to_cube(3 + wave * 2, 4 + wave * 2)
            output_projection_stage.project(
                out_gm, q_workspace_gm, o_proj_weight_gm, self.num_heads, wave
            )

    @jit
    def _execute_task_pool_s8(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        state_gm: Tensor,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        self.batch_groups = replay_decay_gm.indices.shape[0] // BATCH_TILE
        self.seq_len = S8_SEQUENCE_LENGTH
        self.row_count = 128
        self.projection_row_block = 64
        self.recurrent_wave_rows = 64
        # 一个 N384 任务包含三个 N128 数据面板；当每个 batch 组的八个任务
        # 至少能占满启动网格的四分之三时，选择此任务布局。
        n384_task_count = self.batch_groups * 8
        use_n384 = n384_task_count * 4 >= self.block_num * 3
        use_32core_g2_tail = self.batch_groups == 2 and self.block_num == 32

        if use_n384:
            onchip_rewind(reset_sync_id=True)
            self._frontend_adaptive_n384(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
            # 在重置通道标识并转入辅助任务前，必须排空 Cube 流水线。
            # 此时 Vector 侧已没有尚未结束的生产者或消费者。
            cube_sync_all()
            onchip_rewind(reset_sync_id=True)
            self._frontend_adaptive_auxiliary(
                decay_workspace_gm,
                beta_workspace_gm,
                hidden_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
            )
        elif use_32core_g2_tail:
            onchip_rewind(reset_sync_id=True)
            self._frontend_tail(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                replay_decay_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
        else:
            onchip_rewind(reset_sync_id=True)
            self._frontend_flat_m128(
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                beta_weight_gm,
                output_gate_weight_gm,
                replay_decay_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
            )
        global_sync_all()
        onchip_rewind(reset_sync_id=True)

        self._run_backend_stream(
            out_gm,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            decay_workspace_gm,
            output_gate_workspace_gm,
            beta_workspace_gm,
            fa_workspace_gm,
            norm_exchange_gm,
            hidden_gm,
            qkv_weight_gm,
            fa_weight_gm,
            fb_weight_gm,
            beta_weight_gm,
            output_gate_weight_gm,
            output_norm_weight_gm,
            o_proj_weight_gm,
            a_log_gm,
            dt_bias_gm,
            state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            scale_value,
            lower_bound,
            rms_norm_eps,
        )

    @jit
    def execute(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        initial_state_gm: Tensor,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        execution_args = (
            out_gm,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            decay_workspace_gm,
            output_gate_workspace_gm,
            beta_workspace_gm,
            fa_workspace_gm,
            norm_exchange_gm,
            hidden_gm,
            qkv_weight_gm,
            fa_weight_gm,
            fb_weight_gm,
            beta_weight_gm,
            output_gate_weight_gm,
            output_norm_weight_gm,
            o_proj_weight_gm,
            a_log_gm,
            dt_bias_gm,
            initial_state_gm,
            replay_u_gm,
            replay_k_gm,
            replay_decay_gm,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            scale_value,
            lower_bound,
            rms_norm_eps,
        )
        batch_groups = replay_decay_gm.indices.shape[0] // BATCH_TILE
        frontend_panel_tasks = batch_groups * (
            self.qkv_block_count + self.output_gate_block_count
        )
        if const_expr(self.block_num >= 24) and frontend_panel_tasks <= self.block_num:
            self._execute_compact_s8(*execution_args)
        else:
            self._execute_task_pool_s8(*execution_args)


@jit
def _run_tp16_grid(
    grid,
    out,
    q,
    k,
    v,
    decay,
    gate,
    beta,
    fa,
    norm,
    x,
    wqkv,
    wa,
    wb,
    wbeta,
    wz,
    gamma,
    wo,
    alog,
    bias,
    initial,
    replay_u,
    replay_k,
    replay_decay,
    conv,
    state_indices,
    conv_accepted,
    wc,
    scale,
    lower,
    eps,
):
    """B16 的紧凑缓冲区保持静态形状，更大的 batch 沿用原有任务池。"""
    if replay_decay.indices.shape[0] == 16:
        grid.execute(
            out[0:128, None],
            q[0:128, None],
            k[0:128, None],
            v[0:128, None],
            decay[0:128, None],
            gate[0:128, None],
            beta[0:128, None],
            fa[0:128, None],
            norm,
            x[0:128, None],
            wqkv,
            wa,
            wb,
            wbeta,
            wz,
            gamma,
            wo,
            alog,
            bias,
            initial,
            replay_batches(replay_u, 0, 16),
            replay_batches(replay_k, 0, 16),
            replay_batches(replay_decay, 0, 16),
            conv,
            state_indices[0:16,],
            conv_accepted[0:16,],
            wc,
            scale,
            lower,
            eps,
        )
    else:
        grid._execute_task_pool_s8(
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            fa,
            norm,
            x,
            wqkv,
            wa,
            wb,
            wbeta,
            wz,
            gamma,
            wo,
            alog,
            bias,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            conv,
            state_indices,
            conv_accepted,
            wc,
            scale,
            lower,
            eps,
        )


class TP16TPSchedule(TP16GridSchedule):
    def __init__(self, num_heads, rows, auxiliary_rows, publish_rows, config):
        super().__init__(32, HIDDEN_SIZE, num_heads)
        self.main_rows = rows
        self.auxiliary_rows = auxiliary_rows
        self.publish_rows = publish_rows
        self.config = config
        self.seq_len = S8_SEQUENCE_LENGTH


class TP16ProjectionScheduleOperations(
    ProjectionScheduleOperationsFrontendOperations, TP16TPSchedule
):
    """为两种投影调度共用构造逻辑和 Conv 预取操作。"""

    def __init__(self, heads, rows, width):
        config = Schedule(
            main_rows=rows,
            main_width=width,
            ragged_main=width == 256,
            split_fb=True,
            ragged_output=True,
        )
        super().__init__(heads, rows, 32, 64, config)
        self.heads = heads
        self.projection_groups = 4 * heads // 3
        self.row_capacity = rows


class TP16S4ProjectionSchedule(
    S4ProjectionScheduleFrontendOperations, TP16ProjectionScheduleOperations
):
    """仅处理有效行的 S4 QKV/Conv 融合，复用当前 PA 数据面板消费逻辑。"""

    def __init__(self, heads):
        super().__init__(heads, 128, 384)
        self.seq_len = S4_SEQUENCE_LENGTH


class TP16S4MultiCoreSchedule(_S4StagedPipeline):
    """通用 S4 融合，支持 TP4/8/16 和 8/16/24/28/32 AIC。"""

    def __init__(self, heads):
        self.main = TP16S4ProjectionSchedule(heads)
        self.auxiliary = TP16S4AuxiliarySchedule(heads)
        self.backend = TP16S4BackendSchedule(heads)


class _ResidentDelayedZResources:
    """复用 OProj 临时空间，同时将完整 Wo 分片保留在 L1。"""

    def __init__(self):
        self.output = _TiledOutputProjectionStage(
            256,
            64,
            kc=256,
            tiles_per_core=1,
            row_capacity=64,
            resident_input=False,
            resident_weight=True,
            l0c_depth=1,
        )
        self.z_weight = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((128, 128)), depth=2
        )
        self.z_l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((128, 128)), depth=1
        )

    @jit
    def project_z(self, gate, x, weight, row, head):
        accumulator = from_tensor(
            from_tensor(
                self.output.l0c.next_write(),
                layout=make_layout(self.output.l0c_shape),
                data_format="nz",
            ),
            layout=make_layout((128, 128)),
            data_format="nz",
        )
        for kc in range(56):
            a = from_tensor(
                from_tensor(
                    self.output.a_l1.next_write(),
                    layout=make_layout(self.output.a_l1_shape),
                    data_format="nz",
                ),
                layout=make_layout((128, 128)),
                data_format="nz",
            )
            b = self.z_weight.next_write()
            mem_copy(
                a,
                tile_slice(x, (128, 128), (row, kc)),
                options=copy_options(l2_cache_ctl=0),
            )
            mem_copy(
                b,
                tile_slice(weight, (128, 128), (head, kc)),
                options=copy_options(l2_cache_ctl=1),
            )
            l0a = self.z_l0a.next_write()
            l0a = l0a[tuple(map(slice, a.shape))]
            mem_copy(l0a, a)
            l0b = from_tensor(
                self.output.l0b.next_write(),
                layout=make_layout((128, 128)),
                data_format="nz",
            )
            mem_copy(l0b, b)
            matmul(accumulator, l0a, l0b, init=kc == 0)
        mem_copy(tile_slice(gate, (128, 128), (row, head)), accumulator)


class _ResidentDelayedZSchedule(TP16GridSchedule):
    def __init__(self, block_num, hidden_size, num_heads):
        super().__init__(block_num, hidden_size, num_heads)
        self.projection_groups = QKV_PANEL_COUNT
        self.window_rows = 512
        self.minimum_rows = 8 * self.window_rows
        self.projection_weight_l2_cache_ctl = 1

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        self.batch_groups = replay_decay.indices.shape[0] // 16
        self.seq_len = S8_SEQUENCE_LENGTH
        self.row_count = 128
        self.projection_row_block = 64
        self.recurrent_wave_rows = 64
        self._frontend_adaptive_n384(
            q,
            k,
            v,
            decay,
            gate,
            x,
            wqkv,
            wa,
            wb,
            wz,
            conv,
            state_indices,
            conv_accepted,
            wc,
        )
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        self._frontend_adaptive_auxiliary(decay, beta, x, wa, wb, wbeta)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        resources = _ResidentDelayedZResources()
        recurrent = RecurrentKDAStage(
            128, TP16_HEADS, state_key_unroll=4, state_update_unroll=2
        )
        output_gate = OutputGateStage()
        raw = Buffer(MemLoc.UB, dtypes.float32, make_layout((48, 128)))
        resources.output.prefetch_weight(wo)
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        context = _RawRecurrentContext(
            q,
            k,
            v,
            initial.accurate_decay,
            beta,
            initial,
            replay_u,
            replay_k,
            replay_decay,
            scale,
            lower,
        )
        gate_context = _RawGateContext(q, gate, raw, eps)
        for chunk in range(x.shape[0] // 512):
            for ztask in range(
                get_block_idx(), DELAYED_GATE_TASK_COUNT, get_block_num()
            ):
                resources.project_z(
                    gate, x, wz, chunk * 4 + ztask // TP16_HEADS, ztask % TP16_HEADS
                )
            for item in range(first, 384, 64):
                batch = chunk * 64 + item % 64
                head = item // 64
                _run_raw_recurrent_item(recurrent, context, batch, head, raw, item)
            cube_sync_block_arrive(PIPE.FIXPIPE, 0, mode=0)
            cube_sync_block_wait(PIPE.S, 0, mode=0)
            cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=2)
            vec_sync_block_wait(PIPE.S, 1, mode=2)
            for item in range(first, 384, 64):
                batch = chunk * 64 + item % 64
                head = item // 64
                _store_raw_gated_item(output_gate, gate_context, batch, head, item)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for piece in range(8):
                resources.output.project(out, q, wo, TP16_HEADS, chunk * 8 + piece)
            cube_sync_all()
            vec_sync_all()
            onchip_rewind(reset_sync_id=True)


def _tp16_plan(model_batch, tp, cores):
    if tp != TP16_TP_DEGREE or not 1 <= model_batch <= 32 or cores not in VALID_CORES:
        raise ValueError("requires TP16, model batch 1..32, AIC 8/16/24/28/32")
    rows = model_batch * tp * S8_SEQUENCE_LENGTH
    heads = MODEL_HEAD_COUNT // tp
    compact = cores >= 24 and COMPACT_CORES_PER_MODEL_BATCH * model_batch <= cores
    width = 384 if not compact and model_batch * 32 >= cores * 3 else 128
    return dict(
        model_batch=model_batch,
        tp=tp,
        cores=cores,
        rows=rows,
        heads=heads,
        main_m=128,
        main_n=width,
        main_geometry=main_geometry(rows, heads, cores, 128, width),
        auxiliary_rows=128,
        kda_waves=(MODEL_HEAD_COUNT * model_batch + 2 * cores - 1) // (2 * cores),
        output="tp16_grid_r64",
        weight_policy=4 if cores == 32 and model_batch == 1 else 1,
    )


class TP16S8DefaultStage:
    def __init__(self, heads):
        self.heads = heads
        self.grid8 = TP16GridSchedule(8, HIDDEN_SIZE, TP16_HEADS)
        self.grid16 = TP16GridSchedule(16, HIDDEN_SIZE, TP16_HEADS)
        self.grid24 = TP16GridSchedule(24, HIDDEN_SIZE, TP16_HEADS)
        self.grid28 = TP16GridSchedule(28, HIDDEN_SIZE, TP16_HEADS)
        self.grid32 = TP16GridSchedule(32, HIDDEN_SIZE, TP16_HEADS)
        self.compact32 = TP16GridSchedule(32, HIDDEN_SIZE, TP16_HEADS)
        for grid in (self.grid8, self.grid16, self.grid24, self.grid28, self.grid32):
            grid.projection_weight_l2_cache_ctl = 1

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
    ):
        arguments = (
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            fa,
            norm,
            x,
            wqkv,
            wa,
            wb,
            wbeta,
            wz,
            gamma,
            wo,
            alog[0:TP16_HEADS,],
            bias[0:TP16_HEADS, None],
            initial,
            replay_u,
            replay_k,
            replay_decay,
            conv,
            state_indices,
            conv_accepted,
            wc,
            scale,
            lower,
            eps,
        )
        # Keep quota dispatch as a decision tree. Bisheng's jump-table
        # relocation for equality dispatch fails in the OPP device loader.
        # Host validation restricts the quotas to 8, 16, 24, 28 and 32.
        block_num = get_block_num()
        if block_num <= 16:
            if block_num <= 8:
                onchip_rewind(reset_sync_id=True)
                _run_tp16_grid(self.grid8, *arguments)
            else:
                onchip_rewind(reset_sync_id=True)
                _run_tp16_grid(self.grid16, *arguments)
        elif block_num <= 28:
            if block_num <= 24:
                onchip_rewind(reset_sync_id=True)
                _run_tp16_grid(self.grid24, *arguments)
            else:
                onchip_rewind(reset_sync_id=True)
                _run_tp16_grid(self.grid28, *arguments)
        elif replay_decay.indices.shape[0] == 16:
            onchip_rewind(reset_sync_id=True)
            _run_tp16_grid(self.compact32, *arguments)
        else:
            onchip_rewind(reset_sync_id=True)
            self.grid32._execute_task_pool_s8(*arguments)


class TP16S8Dispatch(TP16S8DefaultStage):
    def __init__(self, heads):
        super().__init__(heads)
        self.delayed = _ResidentDelayedZSchedule(32, HIDDEN_SIZE, TP16_HEADS)

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        conv,
        state_indices,
        conv_accepted,
        wc,
        scale,
        lower,
        eps,
        plan_rows,
        plan_width,
        plan_policy,
        prefer_stream,
    ):
        q = q.view((q.shape[0], self.heads * TILE))
        k = k.view((k.shape[0], self.heads * TILE))
        v = v.view((v.shape[0], self.heads * TILE))
        decay = decay.view((decay.shape[0], self.heads * TILE))
        gate = gate.view((gate.shape[0], self.heads * TILE))
        beta = beta.view(
            (beta.shape[0], ((self.heads + BETA_TILE - 1) // BETA_TILE) * BETA_TILE)
        )
        alog = alog.view((self.heads,))
        bias = bias.view((self.heads, TILE))
        conv = _ConvPool(
            conv.data,
            conv.stride0,
            conv.length,
            3 * self.heads * TILE,
            conv.accurate_raw,
        )
        wc = wc.view((3, self.heads, CONV_KERNEL, TILE))
        if (
            get_block_num() == 32
            and x.shape[0] % self.delayed.window_rows == 0
            and (x.shape[0] >= self.delayed.minimum_rows)
        ):
            self.delayed.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )
            global_sync_all()
        else:
            TP16S8DefaultStage.execute(
                self,
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                conv,
                state_indices,
                conv_accepted,
                wc,
                scale,
                lower,
                eps,
                plan_rows,
                plan_width,
                plan_policy,
            )


def _tp16_plan_s4(model_batch, tp, cores):
    """采用保守的源码调度策略，不依赖实测性能查找表。"""
    if tp != TP16_TP_DEGREE or not 1 <= model_batch <= 32 or cores not in VALID_CORES:
        raise ValueError("S4 requires TP16, model batch 1..32, AIC 8/16/24/28/32")
    total_rows = model_batch * tp * S4_SEQUENCE_LENGTH
    main_rows = math.gcd(total_rows, 128)
    width = 128 if total_rows <= 128 else 384
    template = 0

    return dict(
        main_rows=main_rows,
        width=width,
        policy=4 if total_rows <= 128 else 0,
        template=template,
    )


class TP16TPOptimizedKernel:
    """实现单个 TP，为 S8 和 S4 分别缓存编译后的可调用对象。"""

    def __init__(self, sequence):
        self.sequence = sequence
        if sequence == S8_SEQUENCE_LENGTH:
            self.stage = TP16S8Dispatch(TP16_HEADS)
        else:
            self.stage = TP16S4MultiCoreSchedule(TP16_HEADS)

    @kernel(
        profile=cannbotdsl.ProfileSpec(
            name="mega_recurrent_kda_tp16",
            op_type="MegaRecurrentKDA",
            outputs=("out", "replay_u", "replay_k", "replay_decay"),
            inouts=("conv",),
        )
    )
    def device(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        checkpoint,
        conv,
        indices,
        accepted,
        wc,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        conv_stride0,
        conv_length,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        main_rows,
        width,
        policy,
        selector,
        accurate_raw,
        accurate_decay,
    ):
        initial = _CheckpointPool(
            checkpoint, indices, checkpoint_stride0, accurate_decay
        )
        conv = _ConvPool(
            conv, conv_stride0, conv_length, 3 * TP16_HEADS * TILE, accurate_raw
        )
        u = ReplayPool(replay_u, indices, u_stride0, TP16_HEADS, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, TP16_HEADS, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, TP16_HEADS, record_s)
        if const_expr(self.sequence == S8_SEQUENCE_LENGTH):
            self.stage.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                u,
                rk,
                rd,
                conv,
                indices,
                accepted,
                wc,
                scale,
                lower,
                eps,
                main_rows,
                width,
                policy,
                selector,
            )
        else:
            output_rows = main_rows if main_rows <= 128 else 128
            self.stage.execute(
                out,
                q,
                k,
                v,
                decay,
                gate,
                beta,
                fa,
                norm,
                x,
                wqkv,
                wa,
                wb,
                wbeta,
                wz,
                gamma,
                wo,
                alog,
                bias,
                initial,
                u,
                rk,
                rd,
                conv,
                indices,
                accepted,
                wc,
                scale,
                lower,
                eps,
                main_rows,
                width,
                policy,
                output_rows,
            )

    @host
    def run(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        fa,
        norm,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        alog,
        bias,
        checkpoint,
        conv,
        indices,
        accepted,
        wc,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        conv_stride0,
        conv_length,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        main_rows,
        width,
        policy,
        selector,
        cores,
        accurate_raw,
        accurate_decay,
    ):
        self.device[cores](
            out,
            q,
            k,
            v,
            decay,
            gate,
            beta,
            fa,
            norm,
            x,
            wqkv,
            wa,
            wb,
            wbeta,
            wz,
            gamma,
            wo,
            alog,
            bias,
            checkpoint,
            conv,
            indices,
            accepted,
            wc,
            replay_u,
            replay_k,
            replay_decay,
            checkpoint_stride0,
            conv_stride0,
            conv_length,
            u_stride0,
            k_stride0,
            decay_stride0,
            record_s,
            scale,
            lower,
            eps,
            main_rows,
            width,
            policy,
            selector,
            accurate_raw,
            accurate_decay,
        )


class TP16VariableLengthKDAKernel:
    """当前 TP 的变长兜底内核，PA 记录池使用各自独立的步长。"""

    @kernel
    def project(self, out, x, weight):
        onchip_rewind(reset_sync_id=True)
        stage = SequenceProjectionStage()
        stage.project(out, x, weight)

    @kernel
    def convolve(self, out, raw, conv, indices, accepted, cu, weight, conv_stride0):
        onchip_rewind(reset_sync_id=True)
        stage = SequenceConvStage()
        stage.convolve(out, raw, conv, indices, accepted, cu, weight, conv_stride0)

    @kernel
    def recurrent(
        self,
        out,
        convout,
        decay,
        beta,
        z,
        gamma,
        alog,
        bias,
        checkpoint,
        indices,
        cu,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
    ):
        onchip_rewind(reset_sync_id=True)
        u = ReplayPool(replay_u, indices, u_stride0, TP16_HEADS, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, TP16_HEADS, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, TP16_HEADS, record_s)
        stage = SequenceRecurrentStage(TP16_HEADS)
        stage.replay(
            out,
            convout,
            decay,
            beta,
            z,
            gamma,
            alog,
            bias,
            checkpoint,
            indices,
            cu,
            u,
            rk,
            rd,
            checkpoint_stride0,
            scale,
            lower,
            eps,
        )

    @host
    def run(
        self,
        out,
        x,
        wqkv,
        wa,
        wb,
        wbeta,
        wz,
        gamma,
        wo,
        wc,
        conv,
        checkpoint,
        indices,
        accepted,
        cu,
        alog,
        bias,
        replay_u,
        replay_k,
        replay_decay,
        raw,
        fa,
        decay,
        beta,
        z,
        convolved,
        gated,
        conv_stride0,
        checkpoint_stride0,
        u_stride0,
        k_stride0,
        decay_stride0,
        record_s,
        scale,
        lower,
        eps,
        cores,
    ):
        self.project[cores](z, x, wz)
        self.convolve[2 * cores](
            convolved, raw, conv, indices, accepted, cu, wc, conv_stride0
        )
        self.recurrent[2 * cores](
            gated,
            convolved,
            decay,
            beta,
            z,
            gamma,
            alog,
            bias,
            checkpoint,
            indices,
            cu,
            replay_u,
            replay_k,
            replay_decay,
            checkpoint_stride0,
            u_stride0,
            k_stride0,
            decay_stride0,
            record_s,
            scale,
            lower,
            eps,
        )
        self.project[cores](out, gated, wo)


def _optimized_specification(sequence, heads, tp):
    b = cannbotdsl.Dim("batch", min=tp, max=32 * tp, multiple_of=tp)
    rows = sequence * b
    p = heads * D
    beta_width = ((heads + BETA_TILE - 1) // BETA_TILE) * BETA_TILE
    cp = cannbotdsl.Dim("checkpoint_span", min=1)
    cv = cannbotdsl.Dim("conv_span", min=1)
    us = cannbotdsl.Dim("replay_u_span", min=1)
    ks = cannbotdsl.Dim("replay_k_span", min=1)
    ds = cannbotdsl.Dim("replay_decay_span", min=1)
    bf, fp = dtypes.bfloat16, dtypes.float32

    def nd(shape, dt=bf):
        return TensorSpec(shape, dt)

    def nz(shape):
        return TensorSpec(shape, bf, storage_format="nz")

    return (
        nd((rows, HIDDEN_SIZE)),
        *(nd((rows, p)) for _ in range(5)),
        nd((rows, beta_width)),
        nd((rows, D)),
        nd((D, D), fp),
        nd((rows, HIDDEN_SIZE)),
        nz((3 * p, HIDDEN_SIZE)),
        nz((D, HIDDEN_SIZE)),
        nz((p, D)),
        nz((heads, HIDDEN_SIZE)),
        nz((p, HIDDEN_SIZE)),
        nd((D,), fp),
        nz((HIDDEN_SIZE, p)),
        nd((heads,), fp),
        nd((heads, D), fp),
        nd((cp,), fp),
        nd((cv,)),
        nd((b,), dtypes.int32),
        nd((b,), dtypes.int32),
        nd((3, heads, CONV_KERNEL, D)),
        nd((us,), fp),
        nd((ks,), dtypes.float16),
        nd((ds,), fp),
        # checkpoint 和 Conv 步长、Conv 长度、U/K/decay 步长及记录长度 S。
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        # 缩放系数、门控下界和 RMSNorm epsilon。
        dtypes.float32,
        dtypes.float32,
        dtypes.float32,
        # 主投影行数、列宽、缓存策略、路径选择器和启动核数。
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        nd((rows, 3 * p), fp),
        nd((rows, p)),
    )


def _variable_specification(heads):
    t = cannbotdsl.Dim("rows", min=TILE, multiple_of=TILE)
    b = cannbotdsl.Dim("batch", min=1)
    cp = cannbotdsl.Dim("checkpoint_span", min=1)
    cv = cannbotdsl.Dim("conv_span", min=1)
    us = cannbotdsl.Dim("replay_u_span", min=1)
    ks = cannbotdsl.Dim("replay_k_span", min=1)
    ds = cannbotdsl.Dim("replay_decay_span", min=1)
    p = heads * D
    bf, fp = dtypes.bfloat16, dtypes.float32

    def nd(shape, dt=bf):
        return TensorSpec(shape, dt)

    def nz(shape):
        return TensorSpec(shape, bf, storage_format="nz")

    return (
        nd((t, HIDDEN_SIZE)),
        nd((t, HIDDEN_SIZE)),
        nz((3 * p, HIDDEN_SIZE)),
        nz((D, HIDDEN_SIZE)),
        nz((p, D)),
        nz((TILE, HIDDEN_SIZE)),
        nz((p, HIDDEN_SIZE)),
        nd((D,), fp),
        nz((HIDDEN_SIZE, p)),
        nd((CONV_KERNEL, 3 * p)),
        nd((cv,)),
        nd((cp,), fp),
        nd((b,), dtypes.int32),
        nd((b,), dtypes.int32),
        nd((b + 1,), dtypes.int32),
        nd((heads,), fp),
        nd((heads, D), fp),
        nd((us,), fp),
        nd((ks,), dtypes.float16),
        nd((ds,), fp),
        nd((t, 3 * p), fp),
        nd((t, D)),
        nd((t, p)),
        nd((t, TILE)),
        nd((t, p)),
        nd((t, 3 * p)),
        nd((t, p)),
        # Conv、checkpoint、U、K、decay 步长及记录长度 S。
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        # 缩放系数、门控下界、RMSNorm epsilon 和启动核数。
        dtypes.float32,
        dtypes.float32,
        dtypes.float32,
        dtypes.int64,
    )


def _compiled_handle(impl, sequence, device):
    """从当前 TP 实例的缓存取句柄，首次使用时编译。"""
    key = (sequence, device)
    if key not in impl._compiled:
        impl._compiled[key] = impl._compile(sequence)
    return impl._compiled[key]


class _TPImplementation:
    """按 TP 保存计划、内核工厂和 Host 缓存。"""

    def __init__(self, heads, tp, plan, plan_s4, optimized_kernel, variable_kernel):
        self.heads = heads
        self.tp = tp
        self.plan = plan
        self.plan_s4 = plan_s4
        self.optimized_kernel = optimized_kernel
        self.variable_kernel = variable_kernel
        self._compiled = {}
        self._workspaces = {}

    def _compile(self, sequence):
        if sequence:
            run = self.optimized_kernel(sequence).run
            spec = _optimized_specification(sequence, self.heads, self.tp)
        else:
            run = self.variable_kernel().run
            spec = _variable_specification(self.heads)
        return cannbotdsl.compile(run, *spec)

    def compiled_handle(self, sequence, device):
        return _compiled_handle(self, sequence, device)

    def execute_optimized(self, a, batch, sequence, cores):
        rows = batch * sequence
        x = a["hidden_states"].view(rows, HIDDEN_SIZE)
        key = (sequence, batch, x.device)
        if key not in self._workspaces:
            beta_width = ((self.heads + BETA_TILE - 1) // BETA_TILE) * BETA_TILE
            widths = (self.heads * D,) * 5 + (beta_width, D)
            workspace = torch.empty(
                rows * sum(widths), dtype=torch.bfloat16, device=x.device
            )
            views = _mega_workspace_views(workspace, rows, widths)
            norm = torch.empty((D, D), dtype=torch.float32, device=x.device)
            self._workspaces[key] = (views, norm)
        views, norm = self._workspaces[key]

        accurate_raw, accurate_decay, precision_keep = prepare_optimized_precision(
            x, a, self.heads, cores
        )
        model_batch = batch // self.tp
        if sequence == S8_SEQUENCE_LENGTH:
            selected = self.plan(model_batch, self.tp, cores)
            main_m, main_n = selected["main_m"], selected["main_n"]
            prefer_stream = 0
            if self.tp == TP8_TP_DEGREE:
                prefer_stream = _tp8_prefer_streamed_main(
                    rows, self.heads, cores, main_m, main_n
                )
            geometry = (main_m, main_n, selected.get("weight_policy", 4), prefer_stream)
        else:
            selected = self.plan_s4(model_batch, self.tp, cores)
            geometry = (
                selected["main_rows"],
                selected["width"],
                selected["policy"],
                selected["template"],
            )

        out = torch.empty((rows, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device)
        self.compiled_handle(sequence, x.device)(
            out,
            *views,
            norm,
            x,
            a["qkv_projection_weight"],
            a["decay_projection_a_weight"],
            a["decay_projection_b_weight"],
            a["beta_projection_weight"],
            a["output_gate_projection_weight"],
            a["output_norm_weight"],
            a["output_projection_weight"],
            a["A_log"],
            a["dt_bias"],
            _physical_span(a["checkpoint_state"]),
            _physical_span(a["conv_state"]),
            a["state_indices"],
            a["conv_num_accepted_tokens"],
            _pack_conv_weights(a["conv1d_weight"], self.heads),
            _physical_span(a["replay_u"]),
            _physical_span(a["replay_k"]),
            _physical_span(a["replay_decay"]),
            a["checkpoint_state"].stride(0),
            a["conv_state"].stride(0),
            a["conv_state"].shape[1],
            a["replay_u"].stride(0),
            a["replay_k"].stride(0),
            a["replay_decay"].stride(0),
            a["replay_u"].shape[2],
            a["scale"],
            a["lower_bound"],
            a["rms_norm_eps"],
            *geometry,
            cores,
            accurate_raw,
            accurate_decay,
        )
        return out.view(a["hidden_states"].shape), (
            views,
            norm,
            out,
            accurate_raw,
            accurate_decay,
            precision_keep,
        )

    def execute_variable(self, a, prefix, cores):
        import torch_npu

        x = a["hidden_states"]
        total = prefix[-1]
        p = self.heads * D
        rows = ((total + TILE - 1) // TILE) * TILE
        padded = torch.zeros((rows, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device)
        padded[:total].copy_(x.reshape(total, HIDDEN_SIZE))
        beta_weight = torch.zeros(
            (TILE, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device
        )
        beta_weight[: self.heads].copy_(
            torch_npu.npu_format_cast(a["beta_projection_weight"], 2)
        )
        beta_weight = torch_npu.npu_format_cast(beta_weight, 29)
        if x.ndim == 3:
            sequence_length = x.shape[1]
            cu = torch.arange(
                0,
                total + sequence_length,
                sequence_length,
                dtype=torch.int32,
                device=x.device,
            )
        else:
            cu = torch.tensor(prefix, dtype=torch.int32, device=x.device)
        wc = a["conv1d_weight"]
        conv_shape = (CONV_KERNEL, 3 * p)
        if tuple(wc.shape) != conv_shape:
            wc = (
                _pack_conv_weights(wc, self.heads)
                .permute(2, 0, 1, 3)
                .reshape(*conv_shape)
                .contiguous()
            )

        def work(width, dtype=torch.bfloat16):
            return torch.empty((rows, width), dtype=dtype, device=x.device)

        out = work(HIDDEN_SIZE)
        raw = work(3 * p, torch.float32)
        qkv_keep = _twofold_qkv_projection(x, a["qkv_projection_weight"], raw, cores)
        fa, decay, decay_keep = _accurate_auxiliary_projection(
            padded,
            a["decay_projection_a_weight"],
            a["decay_projection_b_weight"],
            cores,
        )
        beta, beta_keep = _auxiliary_twofold_projection(padded, beta_weight, cores)
        z = work(p)
        convolved = work(3 * p)
        gated = work(p)
        self.compiled_handle(0, x.device)(
            out,
            padded,
            a["qkv_projection_weight"],
            a["decay_projection_a_weight"],
            a["decay_projection_b_weight"],
            beta_weight,
            a["output_gate_projection_weight"],
            a["output_norm_weight"],
            a["output_projection_weight"],
            wc,
            _physical_span(a["conv_state"]),
            _physical_span(a["checkpoint_state"]),
            a["state_indices"],
            a["conv_num_accepted_tokens"],
            cu,
            a["A_log"],
            a["dt_bias"],
            _physical_span(a["replay_u"]),
            _physical_span(a["replay_k"]),
            _physical_span(a["replay_decay"]),
            raw,
            fa,
            decay,
            beta,
            z,
            convolved,
            gated,
            a["conv_state"].stride(0),
            a["checkpoint_state"].stride(0),
            a["replay_u"].stride(0),
            a["replay_k"].stride(0),
            a["replay_decay"].stride(0),
            a["replay_u"].shape[2],
            a["scale"],
            a["lower_bound"],
            a["rms_norm_eps"],
            cores,
        )
        keep = (
            padded,
            beta_weight,
            cu,
            wc,
            raw,
            fa,
            decay,
            beta,
            z,
            convolved,
            gated,
            out,
            qkv_keep,
            decay_keep,
            beta_keep,
        )
        return out[:total].view(x.shape), keep

    def export_kernels(self):
        for sequence in (S8_SEQUENCE_LENGTH, S4_SEQUENCE_LENGTH, 0):
            self._compile(sequence).close()


def _make_implementation_lookup():
    instances = {}
    lock = threading.Lock()

    def implementation(heads):
        # 查询和构造共用一把锁，确保同一 TP 只初始化一次。
        with lock:
            impl = instances.get(heads)
            if impl is not None:
                return impl

            if heads == TP4_HEADS:
                impl = _TPImplementation(
                    TP4_HEADS,
                    TP4_TP_DEGREE,
                    _tp4_plan,
                    _tp4_plan_s4,
                    TP4TPOptimizedKernel,
                    TP4VariableLengthKDAKernel,
                )
            elif heads == TP8_HEADS:
                impl = _TPImplementation(
                    TP8_HEADS,
                    TP8_TP_DEGREE,
                    _tp8_plan,
                    _tp8_plan_s4,
                    TP8TPOptimizedKernel,
                    TP8VariableLengthKDAKernel,
                )
            elif heads == TP16_HEADS:
                impl = _TPImplementation(
                    TP16_HEADS,
                    TP16_TP_DEGREE,
                    _tp16_plan,
                    _tp16_plan_s4,
                    TP16TPOptimizedKernel,
                    TP16VariableLengthKDAKernel,
                )
            else:
                raise ValueError("TP4/8/16 requires N=24/12/6")
            instances[heads] = impl
            return impl

    return implementation


implementation = _make_implementation_lookup()


def _execute_mega(arguments):
    tensors = InputValidation.mega_metadata(arguments)
    cores = InputValidation.real_boundary(
        tensors, ("conv_state", "replay_u", "replay_k", "replay_decay")
    )
    prefix = InputValidation.validate_mega_values(arguments)
    lengths = [end - start for start, end in zip(prefix, prefix[1:])]
    batch = len(lengths)
    heads = arguments["beta_projection_weight"].shape[0]
    tp = MODEL_HEAD_COUNT // heads
    module = implementation(heads)
    if (
        batch % tp == 0
        and 1 <= batch // tp <= 32
        and all(length == S8_SEQUENCE_LENGTH for length in lengths)
    ):
        globals()["_last_backend"] = "optimized_s8"
        result, keep = module.execute_optimized(
            arguments, batch, S8_SEQUENCE_LENGTH, cores
        )
    elif (
        batch % tp == 0
        and 1 <= batch // tp <= 32
        and all(length == S4_SEQUENCE_LENGTH for length in lengths)
    ):
        globals()["_last_backend"] = "optimized_s4"
        result, keep = module.execute_optimized(
            arguments, batch, S4_SEQUENCE_LENGTH, cores
        )
    else:
        globals()["_last_backend"] = "variable_length"
        result, keep = module.execute_variable(arguments, prefix, cores)
    globals()["_workspace_keep"] = keep
    return result


@KDAExecutionGuard.serialized
def mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
    cu_seqlens: torch.Tensor | None = None,
):
    """通过 PA 状态池和记录池回放 BSH/TH 序列。

    所有池均有 block_num 个块，并通过 state_indices[b] 寻址。
    仅第 0 轴允许非连续，各池独立使用自己的 stride(0)。
    replay 形状为 [block_num,N,S,D]；U、K、decay 分别为 FP32、FP16、FP32。
    当前 D=128。S 表示记录的 token 容量，必须覆盖每个请求的序列长度。
    块 0 是有效块，checkpoint_state 保持只读。
    返回 (out,conv_state,replay_u,replay_k,replay_decay)，后四项与同名输入互为别名。
    """
    out = _execute_mega(locals().copy())
    return out, conv_state, replay_u, replay_k, replay_decay


@export("mega_recurrent_kda")
def export_mega_recurrent_kda():
    """统一导出补偿投影及三个 TP 的 S8、S4 和变长内核。"""
    if __package__:
        from .mega_recurrent_kda_frontend import (
            _get_auxiliary_compiled,
            _get_twofold_compiled,
        )
    else:
        from mega_recurrent_kda_frontend import (
            _get_auxiliary_compiled,
            _get_twofold_compiled,
        )

    for width in sorted({D, TILE}):  # Fa 和变长路径的 Beta
        _get_auxiliary_compiled(HIDDEN_SIZE, width)
    for heads in SUPPORTED_HEAD_COUNTS:
        _get_twofold_compiled(3 * heads * D)
        _get_auxiliary_compiled(D, heads * D)  # Fb
        implementation(heads).export_kernels()
