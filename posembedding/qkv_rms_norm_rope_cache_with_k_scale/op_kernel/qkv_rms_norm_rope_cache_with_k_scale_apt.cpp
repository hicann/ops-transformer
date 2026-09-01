/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_with_k_scale_apt.cpp
 * \brief Kernel entry for QkvRmsNormRopeCacheWithKScale.
 */
#include "arch35/qkv_rms_norm_rope_cache_with_k_scale_controller.h"
#include "arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using QkvRmsNormRopeCacheWithKScale::GlobalTensors;
using QkvRmsNormRopeCacheWithKScale::QkvRmsNormRopeCacheWithKScaleController;
using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

template <uint32_t HEAD_DIM, uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE, uint32_t K_CACHE_DTYPE,
          uint32_t Q_QUANT_MODE, uint32_t K_QUANT_MODE>
__global__ __aicore__ void qkv_rms_norm_rope_cache_with_k_scale(
    GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos_sin, GM_ADDR slot_mapping, GM_ADDR k_cache_in,
    GM_ADDR v_cache_in, GM_ADDR k_scale_cache_in, GM_ADDR query_start_loc, GM_ADDR seq_lens, GM_ADDR rotation,
    GM_ADDR v_scale, GM_ADDR mrope_position, GM_ADDR q_out, GM_ADDR q_scale, GM_ADDR k_cache_out, GM_ADDR v_cache_out,
    GM_ADDR k_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QkvRmsNormRopeCacheWithKScaleTilingData);
    AscendC::InitSocState();
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    GET_TILING_DATA_WITH_STRUCT(QkvRmsNormRopeCacheWithKScaleTilingData, tilingDataIn, tiling);
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData = &tilingDataIn;

    const GlobalTensors tensors = {
        qkv,         q_gamma,     k_gamma,           cos_sin,         slot_mapping,
        k_cache_in,  v_cache_in,  k_scale_cache_in,  query_start_loc, seq_lens,
        rotation,    v_scale,     mrope_position,    q_out,           q_scale,
        k_cache_out, v_cache_out, k_scale_cache_out, workspace,
    };

    QkvRmsNormRopeCacheWithKScaleController<HEAD_DIM, QKV_LAYOUT, Q_OUT_LAYOUT, ROPE_MODE, K_CACHE_DTYPE, Q_QUANT_MODE,
                                            K_QUANT_MODE>
        controller;
    controller.Process(tensors, tilingData);
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
