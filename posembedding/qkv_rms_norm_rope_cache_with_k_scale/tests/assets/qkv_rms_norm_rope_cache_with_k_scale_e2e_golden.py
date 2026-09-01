#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""PTA E2E TestSpecs for QkvRmsNormRopeCacheWithKScale."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import qkv_rms_norm_rope_cache_with_k_scale_kernel_golden as _kernel  # noqa: E402


def _pta_outputs(result):
    """Preserve the public PTA five-slot return contract, including None."""

    return [
        result.q_out,
        result.q_scale,
        result.k_cache,
        result.v_cache,
        result.k_scale_cache,
    ]


def pta_customize_inputs(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=1e-6,
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
    q_out_dtype=None,
    **kwargs,
):
    """PTA input entry, independently overridable from other paths."""

    _kernel._prepare(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        rotation,
        v_scale,
        mrope_position,
        testcase_name=kwargs.get("testcase_name"),
        head_nums=head_nums,
        layout_qkv=layout_qkv,
        q_quant_mode=q_quant_mode,
        k_quant_mode=k_quant_mode,
    )


def pta_compare(*values, compare_context):
    """PTA comparison entry, independently overridable from other paths."""

    return _kernel.compare_outputs(*values, compare_context=compare_context)


class PtaQkvRmsNormRopeCacheWithKScaleFunctionalTestSpec:
    tolerance = _kernel.TOLERANCE
    customize_inputs = staticmethod(pta_customize_inputs)
    compare = staticmethod(pta_compare)

    @staticmethod
    def golden(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        head_nums,
        *,
        rotation=None,
        v_scale=None,
        layout_qkv="TND",
        layout_q_out="NTD",
        epsilon=1e-6,
        mrope_position=None,
        mrope_section=None,
        q_quant_mode="PerTokenPerHead",
        k_quant_mode="PerTokenPerHead",
        q_out_dtype=None,
        **kwargs,
    ):
        """Functional PTA golden, including no-mutation cache checks."""

        dtype = (
            _kernel.GE_DT_BF16
            if str(q_quant_mode) == "NoQuant"
            else (_kernel.GE_DT_FLOAT8_E4M3FN if q_out_dtype is None else q_out_dtype)
        )
        result = _kernel.numpy_result(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            query_start_loc,
            seq_lens,
            rotation,
            v_scale,
            mrope_position,
            head_nums=head_nums,
            layout_qkv=layout_qkv,
            layout_q_out=layout_q_out,
            epsilon=epsilon,
            mrope_section=mrope_section,
            q_quant_mode=q_quant_mode,
            k_quant_mode=k_quant_mode,
            q_out_dtype=dtype,
        )
        return _pta_outputs(result) + [
            _kernel.to_numpy(k_cache).copy(),
            _kernel.to_numpy(v_cache).copy(),
            _kernel.to_numpy(k_scale_cache).copy(),
        ]


class PtaQkvRmsNormRopeCacheWithKScaleInplaceTestSpec:
    tolerance = _kernel.TOLERANCE
    customize_inputs = staticmethod(pta_customize_inputs)
    compare = staticmethod(pta_compare)

    @staticmethod
    def golden(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        query_start_loc,
        seq_lens,
        head_nums,
        *,
        rotation=None,
        v_scale=None,
        layout_qkv="TND",
        layout_q_out="NTD",
        epsilon=1e-6,
        mrope_position=None,
        mrope_section=None,
        q_quant_mode="PerTokenPerHead",
        k_quant_mode="PerTokenPerHead",
        q_out_dtype=None,
        **kwargs,
    ):
        """In-place PTA golden; TTK appends cache inputs to actual outputs."""

        dtype = (
            _kernel.GE_DT_BF16
            if str(q_quant_mode) == "NoQuant"
            else (_kernel.GE_DT_FLOAT8_E4M3FN if q_out_dtype is None else q_out_dtype)
        )
        result = _kernel.numpy_result(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            query_start_loc,
            seq_lens,
            rotation,
            v_scale,
            mrope_position,
            head_nums=head_nums,
            layout_qkv=layout_qkv,
            layout_q_out=layout_q_out,
            epsilon=epsilon,
            mrope_section=mrope_section,
            q_quant_mode=q_quant_mode,
            k_quant_mode=k_quant_mode,
            q_out_dtype=dtype,
        )
        return _pta_outputs(result)


__spec__ = {
    "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale": "PtaQkvRmsNormRopeCacheWithKScaleFunctionalTestSpec",
    "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_": "PtaQkvRmsNormRopeCacheWithKScaleInplaceTestSpec",
    "torch.ops.cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale": "PtaQkvRmsNormRopeCacheWithKScaleFunctionalTestSpec",
    "torch.ops.cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_": "PtaQkvRmsNormRopeCacheWithKScaleInplaceTestSpec",
}
