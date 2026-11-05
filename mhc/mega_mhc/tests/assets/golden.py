# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MegaMhc reference with explicit activation-dtype rounding at each stage."""

import numpy as np
import ml_dtypes

__spec__ = {"mega_mhc": "MegaMhcTestSpec"}


def _rms_norm(value, weight, epsilon):
    source = value.astype(np.float32)
    inverse = np.float32(1.0) / np.sqrt(
        np.mean(source * source, axis=-1, keepdims=True) + np.float32(epsilon)
    )
    return (source * inverse * weight.astype(np.float32)).astype(value.dtype)


class MegaMhcTestSpec:
    @staticmethod
    def golden(
        x,
        residual,
        post_mix,
        fn,
        mix_scales,
        mix_bases,
        rms_norm_1_weight,
        comb_res_mix=None,
        shifted_prev_mix=None,
        rms_norm_0_weight=None,
        rms_norm_mid_weight=None,
        *,
        rms_norm_eps=1e-6,
        hc_norm_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_post_scale=2.0,
        golden_activation_dtype=None,
        **kwargs,
    ):
        # These casts are graph semantics even when TTK promotes reference inputs.
        # CSV carries the original activation dtype as golden-only metadata.
        input_dtype = (
            golden_activation_dtype or kwargs.get("input_dtypes", (x.dtype,))[0]
        )
        activation_dtype = (
            ml_dtypes.bfloat16
            if str(input_dtype) == "bfloat16"
            else np.dtype(input_dtype)
        )
        x = x.astype(activation_dtype)
        residual = residual.astype(activation_dtype)
        leading = x.shape[:-1]
        d = x.shape[-1]
        x1 = (
            x
            if rms_norm_0_weight is None
            else _rms_norm(x, rms_norm_0_weight, rms_norm_eps)
        )
        mapped = np.einsum(
            "...jn,...jd->...nd",
            comb_res_mix.astype(np.float32),
            residual.astype(np.float32),
        )
        x2 = (
            mapped + post_mix.astype(np.float32) * x1.astype(np.float32)[..., None, :]
        ).astype(x.dtype)
        if rms_norm_mid_weight is not None:
            x2 = _rms_norm(
                x2.reshape(*leading, 4 * d), rms_norm_mid_weight, rms_norm_eps
            ).reshape(x2.shape)
        flat = x2.astype(np.float32).reshape(-1, 4 * d)
        inverse = np.float32(1.0) / np.sqrt(
            np.mean(flat * flat, axis=-1, keepdims=True) + np.float32(hc_norm_eps)
        )
        projected = (flat @ fn.astype(np.float32).T) * inverse
        scales = np.repeat(mix_scales.astype(np.float32), (4, 4, 16))
        logits = projected * scales + mix_bases.astype(np.float32)
        sigmoid = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits[:, :8]))
        pre = (sigmoid[:, :4] + np.float32(hc_pre_eps)).reshape(*leading, 4, 1)
        post = (sigmoid[:, 4:8] * np.float32(hc_post_scale)).reshape(*leading, 4, 1)
        comb = logits[:, 8:24].reshape(*leading, 4, 4)
        weights = pre if shifted_prev_mix is None else shifted_prev_mix
        h_in = np.sum(
            weights.astype(np.float32) * x2.astype(np.float32), axis=-2
        ).astype(x.dtype)
        y = _rms_norm(h_in, rms_norm_1_weight, rms_norm_eps)
        # Keep all prototype positions: reserved outputs are represented by empty tensors.
        empty = np.empty((0,), dtype=np.float32)
        return [x2, post, comb, empty, y, empty.copy(), empty.copy()]
