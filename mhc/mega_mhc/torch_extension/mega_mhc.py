# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 950 MegaMhc forward interface and Meta output inference."""

from typing import Optional, Tuple
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MegaMhcOpBuilder(OpBuilder):
    """Build the repository-local ACLNN wrapper and register its Torch schema."""

    def __init__(self):
        """Register mega_mhc in the mhc category."""
        super().__init__("mega_mhc", category="mhc")

    def sources(self):
        """Resolve the source path produced by torch_extension/setup.py."""
        return ["csrc/mhc/mega_mhc.cpp"]

    def schema(self) -> str:
        """Expose the documented input order, keyword defaults and seven outputs."""
        return "mega_mhc(Tensor x, Tensor residual, Tensor post_mix, Tensor fn, Tensor mix_scales, Tensor mix_bases, Tensor rms_norm_1_weight, *, Tensor? comb_res_mix=None, Tensor? shifted_prev_mix=None, Tensor? rms_norm_0_weight=None, Tensor? rms_norm_mid_weight=None, float rms_norm_eps=1e-6, float hc_norm_eps=1e-6, float hc_pre_eps=1e-6, int inner_precise=0, int hc_mult=4, float hc_post_scale=2.0, float sinkhorn_eps=1e-6, int num_sinkhorn_iters=0, float rms_norm_scale=1.0, int output_mode=0) -> (Tensor, Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?)"

    def register_meta(self):
        """Infer active outputs without loading the NPU extension."""

        @impl(get_as_library(), self.name, "Meta")
        def mega_mhc_meta(
            x,
            residual,
            post_mix,
            fn,
            mix_scales,
            mix_bases,
            rms_norm_1_weight,
            *,
            comb_res_mix=None,
            shifted_prev_mix=None,
            rms_norm_0_weight=None,
            rms_norm_mid_weight=None,
            rms_norm_eps=1e-6,
            hc_norm_eps=1e-6,
            hc_pre_eps=1e-6,
            inner_precise=0,
            hc_mult=4,
            hc_post_scale=2.0,
            sinkhorn_eps=1e-6,
            num_sinkhorn_iters=0,
            rms_norm_scale=1.0,
            output_mode=0,
        ):
            torch._check(
                output_mode in (0, 1, 2), lambda: "output_mode must be 0, 1 or 2"
            )
            torch._check(
                output_mode == 0,
                lambda: "MX Quant output modes 1 and 2 are reserved in this version",
            )
            torch._check(x.dim() in (2, 3), lambda: "x must be [T,D] or [B,S,D]")
            leading = x.shape[:-1]
            return (
                x.new_empty(residual.shape),
                x.new_empty((*leading, 4, 1), dtype=torch.float32),
                x.new_empty((*leading, 4, 4), dtype=torch.float32),
                None
                if shifted_prev_mix is None
                else x.new_empty((*leading, 4, 1), dtype=torch.float32),
                x.new_empty(x.shape),
                None,
                None,
            )


_mega_mhc_op_builder = MegaMhcOpBuilder()
_mega_mhc_op_builder._ensure_initialized()


@impl(get_as_library(), _mega_mhc_op_builder.name, "PrivateUse1")
def _mega_mhc(
    x,
    residual,
    post_mix,
    fn,
    mix_scales,
    mix_bases,
    rms_norm_1_weight,
    *,
    comb_res_mix=None,
    shifted_prev_mix=None,
    rms_norm_0_weight=None,
    rms_norm_mid_weight=None,
    rms_norm_eps=1e-6,
    hc_norm_eps=1e-6,
    hc_pre_eps=1e-6,
    inner_precise=0,
    hc_mult=4,
    hc_post_scale=2.0,
    sinkhorn_eps=1e-6,
    num_sinkhorn_iters=0,
    rms_norm_scale=1.0,
    output_mode=0,
):
    """Dispatch NPU tensors; Host validates the operator contract."""
    return _mega_mhc_op_builder.load().mega_mhc(
        x,
        residual,
        post_mix,
        fn,
        mix_scales,
        mix_bases,
        rms_norm_1_weight,
        comb_res_mix,
        shifted_prev_mix,
        rms_norm_0_weight,
        rms_norm_mid_weight,
        rms_norm_eps,
        hc_norm_eps,
        hc_pre_eps,
        inner_precise,
        hc_mult,
        hc_post_scale,
        sinkhorn_eps,
        num_sinkhorn_iters,
        rms_norm_scale,
        output_mode,
    )


def mega_mhc(
    x,
    residual,
    post_mix,
    fn,
    mix_scales,
    mix_bases,
    rms_norm_1_weight,
    *,
    comb_res_mix=None,
    shifted_prev_mix=None,
    rms_norm_0_weight=None,
    rms_norm_mid_weight=None,
    rms_norm_eps=1e-6,
    hc_norm_eps=1e-6,
    hc_pre_eps=1e-6,
    inner_precise=0,
    hc_mult=4,
    hc_post_scale=2.0,
    sinkhorn_eps=1e-6,
    num_sinkhorn_iters=0,
    rms_norm_scale=1.0,
    output_mode=0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Compute optional RMSNorm_0 -> Post -> optional Mid -> Pre -> RMSNorm_1.

    Coefficients post_mix/new_post_mix retain a trailing singleton dimension.
    x2 is the rounded Post result after optional Mid. Projection RMS scaling
    generates the current coefficients while the weighted sum still uses x2.
    shifted_prev_mix, when provided, replaces the current pre_mix for the
    head-wise weighted sum of x2; None uses the current pre_mix.
    The Pre branch's sigmoid and hc_pre_eps apply only to the current pre_mix.
    RMSNorm_1 is still applied to the summed result.
    comb_res_mix must be provided in this version.
    rms_norm_0_weight=None skips RMSNorm_0 and sends x directly to Post.
    rms_norm_mid_weight=None skips Mid; the projection RMS scaling still runs.
    hc_post_scale scales the post sigmoid; Sinkhorn and quantized outputs are reserved.
    Only output_mode=0 is supported; modes 1/2 are reserved for MX Quant outputs.
    Returns (new_residual, new_post_mix, new_comb_res_mix, new_prev_mix, y, None, None).
    new_prev_mix contains the current pre_mix when shifted_prev_mix is provided;
    otherwise it is None. The quantized outputs remain reserved and return None.
    """
    return torch.ops.cann_ops_transformer.mega_mhc(
        x,
        residual,
        post_mix,
        fn,
        mix_scales,
        mix_bases,
        rms_norm_1_weight,
        comb_res_mix=comb_res_mix,
        shifted_prev_mix=shifted_prev_mix,
        rms_norm_0_weight=rms_norm_0_weight,
        rms_norm_mid_weight=rms_norm_mid_weight,
        rms_norm_eps=rms_norm_eps,
        hc_norm_eps=hc_norm_eps,
        hc_pre_eps=hc_pre_eps,
        inner_precise=inner_precise,
        hc_mult=hc_mult,
        hc_post_scale=hc_post_scale,
        sinkhorn_eps=sinkhorn_eps,
        num_sinkhorn_iters=num_sinkhorn_iters,
        rms_norm_scale=rms_norm_scale,
        output_mode=output_mode,
    )
