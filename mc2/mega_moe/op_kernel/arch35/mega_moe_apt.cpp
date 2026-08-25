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
 * \file mega_moe_apt.cpp
 */

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)
#define ENABLE_TENSOR_API
#endif

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MEGA_MOE_LAYERED_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#ifdef ENABLE_TENSOR_API
#include "mega_moe_wave_a8w8.h"
#include "mega_moe_wave_a4w4.h"
#include "mega_moe_wave_a8w4.h"
#endif

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "mega_moe_layered.h"
#endif

#include "mega_moe_tiling.h"
#include "mega_moe_tiling_key.h"

using namespace AscendC;

#ifndef MEGA_MOE_WEIGHT1_INTERLEAVED
#define MEGA_MOE_WEIGHT1_INTERLEAVED 0
#endif
#if MEGA_MOE_WEIGHT1_INTERLEAVED != 0 && MEGA_MOE_WEIGHT1_INTERLEAVED != 1
#error "MEGA_MOE_WEIGHT1_INTERLEAVED must be 0 or 1"
#endif

// Use the conventional contiguous gate/up weight1 and MX-scale layout by
// default. Builds that provide an interleaved weight can override this to 1.
static constexpr bool WEIGHT1_INTERLEAVED = MEGA_MOE_WEIGHT1_INTERLEAVED != 0;

#ifdef ENABLE_TENSOR_API
namespace MegaMoeImpl {

template <typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode,
          int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved>
class MegaMoeMteWave : public MegaMoeA8W8Wave<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode,
                                              CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved> {
    static_assert((QuantMode == DISPATCH_QUANT_OUT_DTYPE_E5M2 && Std::IsSame<Weight1Type, fp8_e5m2_t>::value) ||
                      (QuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN && Std::IsSame<Weight1Type, fp8_e4m3fn_t>::value),
                  "A8W8 requires matching FP8 activation and weight types");
};

template <typename XType, typename OutputType, typename TopkWeightsType, int32_t CombineQuantMode,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved>
class MegaMoeMteWave<XType, OutputType, TopkWeightsType, fp4x2_e2m1_t, DISPATCH_QUANT_OUT_DTYPE_E4M3FN,
                     CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved>
    : public MegaMoeA8W4Wave<XType, OutputType, TopkWeightsType, fp4x2_e2m1_t, DISPATCH_QUANT_OUT_DTYPE_E4M3FN,
                             CombineQuantMode, TopkWeightsPrefetch> {};

template <typename XType, typename OutputType, typename TopkWeightsType, int32_t CombineQuantMode,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved>
class MegaMoeMteWave<XType, OutputType, TopkWeightsType, fp4x2_e2m1_t, DISPATCH_QUANT_OUT_DTYPE_E2M1, CombineQuantMode,
                     TopkWeightsPrefetch, IsGmm1Interleaved>
    : public MegaMoeA4W4Wave<XType, OutputType, TopkWeightsType, fp4x2_e2m1_t, DISPATCH_QUANT_OUT_DTYPE_E2M1,
                             CombineQuantMode, TopkWeightsPrefetch> {};

} // namespace MegaMoeImpl
#endif

template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType, uint8_t CommModeType,
          bool TopkWeightsPrefetch>
__global__ __aicore__ void mega_moe(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                    GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR bias1,
                                    GM_ADDR bias2, GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR sharedWeight1,
                                    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2,
                                    GM_ADDR sharedBias1, GM_ADDR sharedBias2, GM_ADDR maskBuffer, GM_ADDR yOut,
                                    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
#if defined(ENABLE_TENSOR_API) && defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_BF16) && defined(ORIG_DTYPE_Y) && \
    (ORIG_DTYPE_Y == DT_BF16) && defined(ORIG_DTYPE_WEIGHT1) && \
    ((ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E5M2) || (ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E4M3FN) || \
     (ORIG_DTYPE_WEIGHT1 == DT_FLOAT4_E2M1)) && \
    defined(ORIG_DTYPE_WEIGHT2) && (ORIG_DTYPE_WEIGHT2 == ORIG_DTYPE_WEIGHT1)
    if constexpr (CommModeType == TILINGKEY_TPL_MTE) {
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
            constexpr bool isSupportedMteDtypePair = (DispatchQuantOutType == DISPATCH_QUANT_OUT_DTYPE_E5M2 &&
                                                      Std::IsSame<DTYPE_WEIGHT1, fp8_e5m2_t>::value) ||
                                                     (DispatchQuantOutType == DISPATCH_QUANT_OUT_DTYPE_E4M3FN &&
                                                      (Std::IsSame<DTYPE_WEIGHT1, fp8_e4m3fn_t>::value ||
                                                       Std::IsSame<DTYPE_WEIGHT1, fp4x2_e2m1_t>::value)) ||
                                                     (DispatchQuantOutType == DISPATCH_QUANT_OUT_DTYPE_E2M1 &&
                                                      Std::IsSame<DTYPE_WEIGHT1, fp4x2_e2m1_t>::value);
            if constexpr (isSupportedMteDtypePair) {
                MegaMoeImpl::MegaMoeMteWave<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1, DispatchQuantOutType,
                                            CombineQuantOutType, TopkWeightsPrefetch, WEIGHT1_INTERLEAVED>
                    op;
                op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                        scales, sharedWeight1, sharedWeight2, sharedWeightScales1, sharedWeightScales2, yOut,
                        expertTokenNumsOut, workspaceGM, &tilingData, tilingGM);
                op.Process();
            }
        }
    } else if constexpr (CommModeType == TILINGKEY_TPL_URMA) {
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
            MegaMoeImpl::MegaMoeLayered<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1, DispatchQuantOutType,
                                        CombineQuantOutType, TopkWeightsPrefetch>
                op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                    scales, sharedWeight1, sharedWeight2, sharedWeightScales1, sharedWeightScales2, yOut,
                    expertTokenNumsOut, workspaceGM, &tilingData);
            op.Process();
        }
#endif
    }
#endif
}
