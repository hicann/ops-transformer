# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent PyTorch composition of the EngramGate contract (saved raw dot)."""

import numpy as np
import torch
import ml_dtypes

__spec__ = {
    "engram_gate": "EngramGateKernelSpec",
    "aclnnEngramGate": "EngramGateAclnnSpec",
    "torch.ops.cann_ops_transformer.engram_gate": "EngramGateTorchSpec",
}


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    array = np.ascontiguousarray(value)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _numpy(value):
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().copy().view(ml_dtypes.bfloat16)
    return value.numpy()


def compute(
    hidden_states,
    kv,
    weight_fused,
    eps=1e-6,
    clamp_value=1e-6,
    save_for_backward=True,
    work_dtype=torch.float64,
):
    """CPU FP64 reference; work_dtype=float32 is the composed competitor."""
    hidden_states, kv, weight_fused = map(_tensor, (hidden_states, kv, weight_fused))
    tokens, heads, dim = hidden_states.shape
    x = hidden_states.to(work_dtype)
    k = kv[:, : heads * dim].reshape(tokens, heads, dim).to(work_dtype)
    v = kv[:, heads * dim :].reshape(tokens, 1, dim).to(work_dtype)
    w = weight_fused.to(work_dtype)
    rx = torch.rsqrt(torch.mean(torch.square(x), dim=-1) + eps)
    rk = torch.rsqrt(torch.mean(torch.square(k), dim=-1) + eps)
    raw_dot = torch.sum(x * w * k, dim=-1)
    dot = raw_dot * rx * rk * dim**-0.5
    s = torch.copysign(torch.sqrt(torch.clamp_min(torch.abs(dot), clamp_value)), dot)
    gate = 1 / (1 + torch.exp(-s))
    y = (x + gate.unsqueeze(-1) * v).to(hidden_states.dtype)
    if not save_for_backward:
        return [y] + [
            torch.empty(0, dtype=torch.float32, device=x.device) for _ in range(4)
        ]
    return [y] + [a.float() for a in (raw_dot, gate, rx, rk)]


class _Compose:
    def __init__(self, eps=1e-6, clamp_value=1e-6, save_for_backward=True, **kwargs):
        self.attrs = (float(eps), float(clamp_value), bool(save_for_backward))

    def __call__(self, hidden_states, kv, weight_fused, **kwargs):
        # GPU eager composition is the third-party reference. Keep compilation
        # out of this path so cross_check measures operator accuracy rather than
        # remote torch.compile cache/backend availability.
        return compute(
            hidden_states, kv, weight_fused, *self.attrs, work_dtype=torch.float32
        )


class _AclnnCompose:
    def __init__(self, eps=1e-6, clampValue=1e-6, saveForBackward=True, **kwargs):
        self.attrs = (float(eps), float(clampValue), bool(saveForBackward))

    def __call__(self, hiddenStates, kv, weightFused, **kwargs):
        return compute(
            hiddenStates, kv, weightFused, *self.attrs, work_dtype=torch.float32
        )


class EngramGateKernelSpec:
    third_party = {"torch": _Compose}
    tolerance = {
        d: {"standard": "cross_check", "level": "L1"}
        for d in ("bfloat16", "float16", "float32")
    }

    @staticmethod
    def golden(hidden_states, kv, weight_fused, **kwargs):
        attrs = {
            k: kwargs[k]
            for k in ("eps", "clamp_value", "save_for_backward")
            if k in kwargs
        }
        return [_numpy(a) for a in compute(hidden_states, kv, weight_fused, **attrs)]


class EngramGateAclnnSpec:
    third_party = {"torch": _AclnnCompose}
    tolerance = {
        d: {"standard": "cross_check", "level": "L1"}
        for d in ("bfloat16", "float16", "float32")
    }

    @staticmethod
    def golden(
        hiddenStates,
        kv,
        weightFused,
        eps,
        clampValue,
        saveForBackward,
        output,
        dot,
        gateScore,
        rstdX,
        rstdK,
        **kwargs,
    ):
        result = compute(
            hiddenStates,
            kv,
            weightFused,
            float(eps),
            float(clampValue),
            bool(saveForBackward),
        )
        if kwargs.get("use_numpy", not isinstance(output, torch.Tensor)):
            return tuple(_numpy(value) for value in result)
        return tuple(result)


class EngramGateTorchSpec:
    # TTK e2e currently does not dispatch third-party benchmarks.
    tolerance = EngramGateAclnnSpec.tolerance

    @staticmethod
    def golden(
        hidden_states,
        kv,
        weight_fused,
        eps=1e-6,
        clamp_value=1e-6,
        save_for_backward=True,
        **kwargs,
    ):
        outputs = compute(
            hidden_states, kv, weight_fused, eps, clamp_value, save_for_backward
        )
        return (
            tuple(outputs)
            if save_for_backward
            else (outputs[0], None, None, None, None)
        )


def engram_gate_golden(
    hidden_states,
    kv,
    weight_fused,
    eps=1e-6,
    clamp_value=1e-6,
    save_for_backward=True,
):
    """Module-level entry used by generic golden probes."""
    return EngramGateKernelSpec.golden(
        hidden_states,
        kv,
        weight_fused,
        eps=eps,
        clamp_value=clamp_value,
        save_for_backward=save_for_backward,
    )
