# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mega KDA 公开入口，负责校验并分发到三个 TP 实现。"""

from __future__ import annotations
import ctypes
import functools
import math
import sys
import threading
from types import ModuleType
import torch

if __package__:
    from . import _kda_platform as _platform
    from . import mega_recurrent_kda_arch92 as _variant
else:
    import _kda_platform as _platform
    import mega_recurrent_kda_arch92 as _variant

import cannbotdsl
from cannbotdsl.aot import export

if __package__:
    from .mega_recurrent_kda_common import (
        HIDDEN_SIZE,
        CONV_KERNEL,
        TP4_HEADS,
        TP8_HEADS,
        TP16_HEADS,
    )
else:
    from mega_recurrent_kda_common import (
        HIDDEN_SIZE,
        CONV_KERNEL,
        TP4_HEADS,
        TP8_HEADS,
        TP16_HEADS,
    )

# 模型维度与 TP 映射
MODEL_HEAD_COUNT = 96
D = 128
TP4_DEGREE = 4
TP8_DEGREE = 8
TP16_DEGREE = 16
TP4_HEAD_COUNT = MODEL_HEAD_COUNT // TP4_DEGREE
TP8_HEAD_COUNT = MODEL_HEAD_COUNT // TP8_DEGREE
TP16_HEAD_COUNT = MODEL_HEAD_COUNT // TP16_DEGREE
SUPPORTED_HEAD_COUNTS = (TP16_HEAD_COUNT, TP8_HEAD_COUNT, TP4_HEAD_COUNT)

# 固定长度融合路径与支持的 AIC 配额
S4_SEQUENCE_LENGTH = 4
S8_SEQUENCE_LENGTH = 8
VALID_CORES = _platform.architecture().mega_quotas


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
    """返回当前流支持的最大 AIC 配额。"""
    available = _platform.available_aic(ref)
    quotas = VALID_CORES
    InputValidation.require(
        available >= max(quotas) or available in quotas,
        f"AIC quota {available} is not supported; use {quotas}",
    )
    return max(core for core in quotas if core <= available)


def implementation(heads):
    """加载选定架构的实现，共用公开接口注册。"""
    if heads not in SUPPORTED_HEAD_COUNTS:
        raise ValueError("TP4/8/16 requires N=24/12/6")
    return _variant.for_heads(heads)


def _execute_mega(arguments):
    tensors = InputValidation.mega_metadata(arguments)
    cores = InputValidation.real_boundary(
        tensors, ("conv_state", "replay_u", "replay_k", "replay_decay")
    )
    prefix = InputValidation.validate_mega_values(arguments)
    heads = arguments["beta_projection_weight"].shape[0]
    result, keep = implementation(heads).execute_variable(arguments, prefix, cores)
    globals()["_last_backend"] = "variable_length"
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
    """按架构导出三个 TP 的全部可达主程序。"""
    for heads in SUPPORTED_HEAD_COUNTS:
        implementation(heads).export_kernels()
    _variant.export_precision_kernels(SUPPORTED_HEAD_COUNTS)
