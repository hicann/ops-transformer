# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Use the shared numerical stages with architecture-specific launch counts."""

from __future__ import annotations

import cannbotdsl
import torch
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.sync import onchip_rewind

if __package__:
    from . import _kda_platform as _platform
    from . import mega_recurrent_kda_frontend as _frontend
    from .mega_recurrent_kda_backend import SequenceRecurrentStage as _RecurrentStage
    from .mega_recurrent_kda_common import (
        CONV_KERNEL,
        D,
        HIDDEN_SIZE,
        TILE,
        ReplayPool,
        _pack_conv_weights,
    )
else:
    import _kda_platform as _platform
    import mega_recurrent_kda_frontend as _frontend
    from mega_recurrent_kda_backend import SequenceRecurrentStage as _RecurrentStage
    from mega_recurrent_kda_common import (
        CONV_KERNEL,
        D,
        HIDDEN_SIZE,
        TILE,
        ReplayPool,
        _pack_conv_weights,
    )


class SequenceProjectionStage:
    @kernel
    def kernel(self, out, x, weight):
        onchip_rewind(reset_sync_id=True)
        stage = _frontend.SequenceProjectionStage()
        stage.project(out, x, weight)


class SequenceConvStage:
    @kernel
    def kernel(self, out, raw, conv, indices, accepted, cu, weight_gm, conv_stride0):
        onchip_rewind(reset_sync_id=True)
        stage = _frontend.SequenceConvStage()
        stage.convolve(out, raw, conv, indices, accepted, cu, weight_gm, conv_stride0)


class SequenceRecurrentStage:
    def __init__(self, heads):
        self.heads = heads

    @kernel
    def kernel(
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
        u = ReplayPool(replay_u, indices, u_stride0, self.heads, record_s)
        rk = ReplayPool(replay_k, indices, k_stride0, self.heads, record_s)
        rd = ReplayPool(replay_decay, indices, decay_stride0, self.heads, record_s)
        stage = _RecurrentStage(self.heads)
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


class VariableLengthKDAKernel:
    def __init__(self, aiv_per_aic=2):
        self.aiv_per_aic = aiv_per_aic

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
        gemm = SequenceProjectionStage()
        conv_kernel = SequenceConvStage()
        # Host supplies compensated raw/Fa/Fb/Beta with their original ABI.
        gemm.kernel[cores](z, x, wz)
        conv_kernel.kernel[self.aiv_per_aic * cores](
            convolved, raw, conv, indices, accepted, cu, wc, conv_stride0
        )
        n = alog.shape[0]
        if n == 6:
            recurrent = SequenceRecurrentStage(6)
            recurrent.kernel[self.aiv_per_aic * cores](
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
        elif n == 12:
            recurrent = SequenceRecurrentStage(12)
            recurrent.kernel[self.aiv_per_aic * cores](
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
        else:
            recurrent = SequenceRecurrentStage(24)
            recurrent.kernel[self.aiv_per_aic * cores](
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
        gemm.kernel[cores](out, gated, wo)


def _sequence_specification(heads=6):
    t = cannbotdsl.Dim("rows", min=TILE, multiple_of=TILE)
    b = cannbotdsl.Dim("batch", min=1)
    cp = cannbotdsl.Dim("checkpoint_span", min=1)
    cv = cannbotdsl.Dim("conv_span", min=1)
    us = cannbotdsl.Dim("replay_u_span", min=1)
    ks = cannbotdsl.Dim("replay_k_span", min=1)
    ds = cannbotdsl.Dim("replay_decay_span", min=1)
    p = heads * D
    bf, fp = dtypes.bfloat16, dtypes.float32
    nd = lambda shape, dt=bf: TensorSpec(shape, dt)
    nz = lambda shape: TensorSpec(shape, bf, storage_format="nz")
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


def _physical_span(tensor):
    span = (tensor.shape[0] - 1) * tensor.stride(0) + tensor[0].numel()
    return tensor.as_strided((span,), (1,))


_SEQUENCE_COMPILED = {}
_WORKSPACE_KEEP = None


def compiled_sequence(heads, target=None):
    target = target or _platform.architecture()
    key = (target.directory, heads, target.aiv_per_aic)
    if key not in _SEQUENCE_COMPILED:
        program = VariableLengthKDAKernel(target.aiv_per_aic)
        _SEQUENCE_COMPILED[key] = cannbotdsl.compile(
            program.run, *_sequence_specification(heads)
        )
    return _SEQUENCE_COMPILED[key]


def export_sequence(heads=6):
    target = _platform.architecture()
    handle = cannbotdsl.compile(
        VariableLengthKDAKernel(target.aiv_per_aic).run, *_sequence_specification(heads)
    )
    handle.close()


def execute_sequence(a, prefix, cores, heads):
    import torch_npu

    x = a["hidden_states"]
    total = prefix[-1]
    p = heads * D
    rows = ((total + TILE - 1) // TILE) * TILE
    padded = torch.zeros((rows, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device)
    padded[:total].copy_(x.reshape(total, HIDDEN_SIZE))
    beta_weight = torch.zeros(
        (TILE, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device
    )
    beta_weight[:heads].copy_(torch_npu.npu_format_cast(a["beta_projection_weight"], 2))
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
            _pack_conv_weights(wc, heads)
            .permute(2, 0, 1, 3)
            .reshape(*conv_shape)
            .contiguous()
        )

    def work(width, dtype=torch.bfloat16):
        return torch.empty((rows, width), dtype=dtype, device=x.device)

    out = work(HIDDEN_SIZE)
    raw = work(3 * p, torch.float32)
    fa = work(D)
    decay = work(p)
    beta = work(TILE)
    z = work(p)
    convolved = work(3 * p)
    gated = work(p)
    qkv_keep = _frontend._twofold_qkv_projection(
        x, a["qkv_projection_weight"], raw, cores
    )
    fa, decay, auxiliary_keep = _frontend._accurate_auxiliary_projection(
        padded,
        a["decay_projection_a_weight"],
        a["decay_projection_b_weight"],
        cores,
    )
    beta, beta_keep = _frontend._auxiliary_twofold_projection(
        padded, beta_weight, cores
    )
    compiled_sequence(heads)(
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
        auxiliary_keep,
        beta_keep,
    )
    globals()["_WORKSPACE_KEEP"] = keep
    return out[:total].view(x.shape)
