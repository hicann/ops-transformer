/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file norm_rope_concat_grad.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "norm_rope_concat_grad_kernel.h"

using namespace AscendC;
using namespace nrcg;

#define Impl(tilingKey, normType, addedNormType, ropeType, concatOrder) \
    if (TILING_KEY_IS(tilingKey)) { \
        NormRopeConcatGrad<normType, addedNormType, ropeType, concatOrder> op( \
            gm_grad_query_output, gm_grad_key_output, gm_grad_value_output, gm_query, gm_key, gm_encoder_query, \
            gm_encoder_key, gm_norm_query_weight, gm_norm_query_mean, gm_norm_query_rstd, gm_norm_key_weight, \
            gm_norm_key_mean, gm_norm_key_rstd, gm_norm_added_query_weight, gm_norm_added_query_mean, \
            gm_norm_added_query_rstd, gm_norm_added_key_weight, gm_norm_added_key_mean, gm_norm_added_key_rstd, \
            gm_rope_sin, gm_rope_cos, gm_grad_query, gm_grad_key, gm_grad_value, gm_grad_encoder_query, \
            gm_grad_encoder_key, gm_grad_encoder_value, gm_grad_norm_query_weight, gm_grad_norm_query_bias, \
            gm_grad_norm_key_weight, gm_grad_norm_key_bias, gm_grad_norm_added_query_weight, \
            gm_grad_norm_added_query_bias, gm_grad_norm_added_key_weight, gm_grad_norm_added_key_bias, workspaceGM); \
        op.Init(&pipe, tilingData); \
        op.Process(); \
        return; \
    }

extern "C" __global__ __aicore__ void norm_rope_concat_grad(
    GM_ADDR gm_grad_query_output, GM_ADDR gm_grad_key_output, GM_ADDR gm_grad_value_output, GM_ADDR gm_query,
    GM_ADDR gm_key, GM_ADDR gm_encoder_query, GM_ADDR gm_encoder_key, GM_ADDR gm_norm_query_weight,
    GM_ADDR gm_norm_query_mean, GM_ADDR gm_norm_query_rstd, GM_ADDR gm_norm_key_weight, GM_ADDR gm_norm_key_mean,
    GM_ADDR gm_norm_key_rstd, GM_ADDR gm_norm_added_query_weight, GM_ADDR gm_norm_added_query_mean,
    GM_ADDR gm_norm_added_query_rstd, GM_ADDR gm_norm_added_key_weight, GM_ADDR gm_norm_added_key_mean,
    GM_ADDR gm_norm_added_key_rstd, GM_ADDR gm_rope_sin, GM_ADDR gm_rope_cos, GM_ADDR gm_grad_query,
    GM_ADDR gm_grad_key, GM_ADDR gm_grad_value, GM_ADDR gm_grad_encoder_query, GM_ADDR gm_grad_encoder_key,
    GM_ADDR gm_grad_encoder_value, GM_ADDR gm_grad_norm_query_weight, GM_ADDR gm_grad_norm_query_bias,
    GM_ADDR gm_grad_norm_key_weight, GM_ADDR gm_grad_norm_key_bias, GM_ADDR gm_grad_norm_added_query_weight,
    GM_ADDR gm_grad_norm_added_query_bias, GM_ADDR gm_grad_norm_added_key_weight, GM_ADDR gm_grad_norm_added_key_bias,
    GM_ADDR workspaceGM, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    Impl(0, NormType::NONE, NormType::NONE, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
        Impl(10, NormType::NONE, NormType::NONE, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
            Impl(100, NormType::NONE, NormType::NONE, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                Impl(110, NormType::NONE, NormType::NONE, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                    Impl(200, NormType::NONE, NormType::NONE, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                        Impl(210, NormType::NONE, NormType::NONE, RopeType::HALF, ConcatOrder::AFTER_ENCODER)

                            Impl(100000, NormType::NONE, NormType::LAYER_NORM, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                Impl(100010, NormType::NONE, NormType::LAYER_NORM, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                    Impl(100100, NormType::NONE, NormType::LAYER_NORM, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                        Impl(100110, NormType::NONE, NormType::LAYER_NORM, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                            Impl(100200, NormType::NONE, NormType::LAYER_NORM, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                Impl(100210, NormType::NONE, NormType::LAYER_NORM, RopeType::HALF, ConcatOrder::AFTER_ENCODER)
                                                    Impl(200000, NormType::NONE, NormType::LAYER_NORM_AFFINE, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                        Impl(200010, NormType::NONE, NormType::LAYER_NORM_AFFINE, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                            Impl(200100, NormType::NONE, NormType::LAYER_NORM_AFFINE, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                Impl(200110, NormType::NONE, NormType::LAYER_NORM_AFFINE, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                    Impl(200200, NormType::NONE, NormType::LAYER_NORM_AFFINE, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                        Impl(200210, NormType::NONE, NormType::LAYER_NORM_AFFINE, RopeType::HALF, ConcatOrder::AFTER_ENCODER)

                                                                            Impl(100000000, NormType::LAYER_NORM, NormType::NONE, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                                                Impl(100000010, NormType::LAYER_NORM, NormType::NONE, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                                                    Impl(100000100, NormType::LAYER_NORM, NormType::NONE, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                                        Impl(100000110, NormType::LAYER_NORM, NormType::NONE, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                                            Impl(100000200, NormType::LAYER_NORM, NormType::NONE, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                                                Impl(100000210, NormType::LAYER_NORM, NormType::NONE, RopeType::HALF, ConcatOrder::AFTER_ENCODER)
                                                                                                    Impl(100100000, NormType::LAYER_NORM, NormType::LAYER_NORM, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                                                                        Impl(100100010, NormType::LAYER_NORM, NormType::LAYER_NORM, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                                                                            Impl(100100100, NormType::LAYER_NORM, NormType::LAYER_NORM, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                Impl(100100110, NormType::LAYER_NORM, NormType::LAYER_NORM, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                                                                    Impl(100100200, NormType::LAYER_NORM, NormType::LAYER_NORM, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                                                                        Impl(100100210, NormType::LAYER_NORM, NormType::LAYER_NORM, RopeType::HALF, ConcatOrder::AFTER_ENCODER)
                                                                                                                            Impl(100200000, NormType::LAYER_NORM, NormType::LAYER_NORM_AFFINE, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                Impl(100200010, NormType::LAYER_NORM, NormType::LAYER_NORM_AFFINE, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                    Impl(100200100, NormType::LAYER_NORM, NormType::LAYER_NORM_AFFINE, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                        Impl(100200110, NormType::LAYER_NORM, NormType::LAYER_NORM_AFFINE, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                            Impl(100200200, NormType::LAYER_NORM, NormType::LAYER_NORM_AFFINE, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                Impl(100200210, NormType::LAYER_NORM, NormType::LAYER_NORM_AFFINE, RopeType::HALF, ConcatOrder::AFTER_ENCODER)

                                                                                                                                                    Impl(200000000, NormType::LAYER_NORM_AFFINE, NormType::NONE, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                        Impl(200000010, NormType::LAYER_NORM_AFFINE, NormType::NONE, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                            Impl(200000100, NormType::LAYER_NORM_AFFINE, NormType::NONE, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                Impl(200000110, NormType::LAYER_NORM_AFFINE, NormType::NONE, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                    Impl(200000200, NormType::LAYER_NORM_AFFINE, NormType::NONE, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                        Impl(200000210, NormType::LAYER_NORM_AFFINE, NormType::NONE, RopeType::HALF, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                            Impl(200100000, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                                Impl(200100010, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                                    Impl(200100100, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                                        Impl(200100110, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                                            Impl(200100200, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                                                Impl(200100210, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM, RopeType::HALF, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                                                    Impl(200200000, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM_AFFINE, RopeType::NONE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                                                        Impl(200200010, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM_AFFINE, RopeType::NONE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                                                            Impl(200200100, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM_AFFINE, RopeType::INTERLEAVE, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                                                                Impl(200200110, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM_AFFINE, RopeType::INTERLEAVE, ConcatOrder::AFTER_ENCODER)
                                                                                                                                                                                                                    Impl(200200200, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM_AFFINE, RopeType::HALF, ConcatOrder::BEFORE_ENCODER)
                                                                                                                                                                                                                        Impl(200200210, NormType::LAYER_NORM_AFFINE, NormType::LAYER_NORM_AFFINE, RopeType::HALF, ConcatOrder::AFTER_ENCODER)
}
