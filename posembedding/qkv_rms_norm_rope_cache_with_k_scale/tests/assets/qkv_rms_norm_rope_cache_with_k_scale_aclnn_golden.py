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

"""ACLNN TestSpec for QkvRmsNormRopeCacheWithKScale."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import qkv_rms_norm_rope_cache_with_k_scale_kernel_golden as _kernel  # noqa: E402


def aclnn_customize_inputs(
    qkv,
    qGamma,
    kGamma,
    cosSin,
    slotMapping,
    kCacheRef,
    vCacheRef,
    kScaleCacheRef,
    queryStartLocOptional=None,
    seqLensOptional=None,
    rotationOptional=None,
    vScaleOptional=None,
    mropePositionOptional=None,
    headNums=None,
    layoutQkv="TND",
    layoutQOut="NTD",
    epsilon=1e-6,
    mropeSectionOptional=None,
    qQuantMode="PerTokenPerHead",
    kQuantMode="PerTokenPerHead",
    qOut=None,
    qScaleOptional=None,
    **kwargs,
):
    """ACLNN input entry; parameter names follow GetWorkspaceSize."""

    _kernel._prepare(
        qkv,
        qGamma,
        kGamma,
        cosSin,
        slotMapping,
        kCacheRef,
        vCacheRef,
        kScaleCacheRef,
        queryStartLocOptional,
        seqLensOptional,
        rotationOptional,
        vScaleOptional,
        mropePositionOptional,
        testcase_name=kwargs.get("testcase_name"),
        head_nums=headNums,
        layout_qkv=layoutQkv,
        q_quant_mode=qQuantMode,
        k_quant_mode=kQuantMode,
    )


def aclnn_compare(*values, compare_context):
    """ACLNN comparison entry, independently overridable from other paths."""

    return _kernel.compare_outputs(*values, compare_context=compare_context)


class AclnnQkvRmsNormRopeCacheWithKScaleTestSpec:
    tolerance = _kernel.TOLERANCE
    customize_inputs = staticmethod(aclnn_customize_inputs)
    compare = staticmethod(aclnn_compare)

    @staticmethod
    def golden(
        qkv,
        qGamma,
        kGamma,
        cosSin,
        slotMapping,
        kCacheRef,
        vCacheRef,
        kScaleCacheRef,
        queryStartLocOptional=None,
        seqLensOptional=None,
        rotationOptional=None,
        vScaleOptional=None,
        mropePositionOptional=None,
        headNums=None,
        layoutQkv="TND",
        layoutQOut="NTD",
        epsilon=1e-6,
        mropeSectionOptional=None,
        qQuantMode="PerTokenPerHead",
        kQuantMode="PerTokenPerHead",
        qOut=None,
        qScaleOptional=None,
        **kwargs,
    ):
        """ACLNN golden; output order follows the public ACLNN contract."""

        q_out_dtype = (
            _kernel.GE_DT_BF16
            if str(qQuantMode) == "NoQuant"
            else _kernel.GE_DT_FLOAT8_E4M3FN
        )
        result = _kernel.numpy_result(
            qkv,
            qGamma,
            kGamma,
            cosSin,
            slotMapping,
            kCacheRef,
            vCacheRef,
            kScaleCacheRef,
            queryStartLocOptional,
            seqLensOptional,
            rotationOptional,
            vScaleOptional,
            mropePositionOptional,
            head_nums=headNums,
            layout_qkv=layoutQkv,
            layout_q_out=layoutQOut,
            epsilon=epsilon,
            mrope_section=mropeSectionOptional,
            q_quant_mode=qQuantMode,
            k_quant_mode=kQuantMode,
            q_out_dtype=q_out_dtype,
        )
        outputs = [result.q_out]
        if result.q_scale is not None:
            outputs.append(result.q_scale)
        outputs.extend((result.k_cache, result.v_cache, result.k_scale_cache))
        return outputs


__spec__ = {
    "aclnnQkvRmsNormRopeCacheWithKScale": "AclnnQkvRmsNormRopeCacheWithKScaleTestSpec",
}
