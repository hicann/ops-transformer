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
 * \file quant_flash_attn.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "arch35/quant_flash_attn_common_def.h"
#include "util.h"
#include "arch35/quant_flash_attn_kernel_mxfp8.h"
#include "arch35/quant_flash_attn_kernel_fp8.h"
#include "arch35/quant_flash_attn_kernel_hif8.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_tiling_data.h"
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/vector_common.h"
#else
#include "../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../common/op_kernel/vector_common.h"
#endif

using namespace AscendC;
using namespace optiling;

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__aicore__ inline void quant_flash_attn_mxfp8(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    using INPUT_T = fp8_e4m3fn_t;
    using OUT_T = bfloat16_t;

    fa_base_matmul::ResetIdCounter();

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(
        (static_cast<std::underlying_type_t<inferDTemplateType>>(ConfigValue[config].d) + 63) >> 6 << 6);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(
        (static_cast<std::underlying_type_t<inferDTemplateType>>(ConfigValue[config].dv) + 63) >> 6 << 6);
    constexpr bool isDAligned = ConfigValue[config].d != inferDTemplateType::NotAligned &&
                                static_cast<int32_t>(ConfigValue[config].d) % 64 == 0;

    constexpr bool isFdConst = false;
    constexpr bool useDn = (quantMode == QFA_MXFP8_FP32_PREFILL);

    using CubeBlock =
        BaseApi::QuantFlashAttnBlockCubeMxfp8<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                              dTemplateType, dVTemplateType, KvLayoutType, useDn, isDAligned>;
    using VecFaBlock =
        BaseApi::QuantFlashAttnBlockVecMxfp8<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                             s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType,
                                             isFdConst, useDn, isDAligned>;
    using VecFdBlock =
        BaseApi::QuantFlashAttnBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                   s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                   hasAttenMask, KvLayoutType, useDn>;

    using CubeBlockDummy =
        BaseApi::QuantFlashAttnBlockCubeMxfp8Dummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                                   dTemplateType, dVTemplateType, KvLayoutType, useDn>;
    using VecFaBlockDummy =
        BaseApi::QuantFlashAttnBlockVecMxfp8Dummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                  s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                  hasAttenMask, KvLayoutType, isFdConst, useDn>;
    using VecFdBlockDummy =
        BaseApi::QuantFlashAttnBlockVecFlashDecodeDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                        hasAttenMask, KvLayoutType, useDn>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::QuantFlashAttnKernelMxfp8<CubeBlock, VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::QuantFlashAttnKernelMxfp8<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    const __gm__ QuantFlashAttnTilingData *__restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData *__restrict)tiling;

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, sinks, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, dequantScaleQuery, dequantScaleKey,
            dequantScaleValue, pScale, softmaxLse, attnOut, workspace, metadata, sequsedQ, sequsedKv, tilingData,
            &tPipe);
    op.Process();
}

// ─────────────────────────────────────────────────────────
// quant_flash_attn_kernel_run_gqa: GQA FP8 Fullquant
// ─────────────────────────────────────────────────────────
template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint8_t config>
inline __aicore__ void quant_flash_attn_gqa_fp8(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *qDescale,
    __gm__ uint8_t *kDescale, __gm__ uint8_t *vDescale, __gm__ uint8_t *blockTable, __gm__ uint8_t *pScale,
    __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    fa_base_matmul::ResetIdCounter();

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    constexpr bool useDn = true;

    using Fp8CubeBlock =
        BaseApi::QuantFlashAttnBlockCubeGqaFp8<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                               dTemplateType, dVTemplateType, KvLayoutType, useDn>;
    using Fp8VecFaBlock =
        BaseApi::QuantFlashAttnBlockVecGqaFp8<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                              s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType,
                                              false, useDn>;
    using VecFdBlock =
        BaseApi::QuantFlashAttnBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                   s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                   hasAttenMask, KvLayoutType, useDn>;

    using Fp8CubeBlockDummy =
        BaseApi::QuantFlashAttnBlockCubeGqaFp8Dummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                                    dTemplateType, dVTemplateType, KvLayoutType, useDn>;
    using Fp8VecFaBlockDummy =
        BaseApi::QuantFlashAttnBlockVecGqaFp8Dummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                   s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                   hasAttenMask, KvLayoutType, false, useDn>;
    using VecFdBlockDummy =
        BaseApi::QuantFlashAttnBlockVecFlashDecodeDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                        hasAttenMask, KvLayoutType, useDn>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::QuantFlashAttnKernelFp8<Fp8CubeBlock, Fp8VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::QuantFlashAttnKernelFp8<Fp8CubeBlockDummy, Fp8VecFaBlock, VecFdBlock>;
#endif

    const __gm__ QuantFlashAttnTilingData *__restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData *__restrict)tiling;

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, sinks, cuSeqLensQ, cuSeqLensKv, blockTable, qDescale, kDescale, vDescale, pScale,
            softmaxLse, attnOut, workspace, metadata, sequsedQ, sequsedKv, tilingData, &tPipe);
    op.Process();
}

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__aicore__ inline void quant_flash_attn_hif8(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    using INPUT_T = hifloat8_t;
    using OUT_T = bfloat16_t;

    fa_base_matmul::ResetIdCounter();

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    constexpr bool isFdConst = false;
    constexpr bool useDn = true;

    using CubeBlock =
        BaseApi::QuantFlashAttnBlockCubeHif8<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                             dTemplateType, dVTemplateType, KvLayoutType, useDn>;
    using VecFaBlock =
        BaseApi::QuantFlashAttnBlockVecHif8<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                            s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType,
                                            isFdConst, useDn>;
    using VecFdBlock =
        BaseApi::QuantFlashAttnBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                   s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                   hasAttenMask, KvLayoutType, useDn>;

    using CubeBlockDummy =
        BaseApi::QuantFlashAttnBlockCubeHif8Dummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                                  dTemplateType, dVTemplateType, KvLayoutType, useDn>;
    using VecFaBlockDummy =
        BaseApi::QuantFlashAttnBlockVecHif8Dummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                 s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                 hasAttenMask, KvLayoutType, isFdConst, useDn>;
    using VecFdBlockDummy =
        BaseApi::QuantFlashAttnBlockVecFlashDecodeDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                        hasAttenMask, KvLayoutType, useDn>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::QuantFlashAttnKernelHif8<CubeBlock, VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::QuantFlashAttnKernelHif8<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    const __gm__ QuantFlashAttnTilingData *__restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData *__restrict)tiling;

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, sinks, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, dequantScaleQuery, dequantScaleKey,
            dequantScaleValue, pScale, softmaxLse, attnOut, workspace, metadata, sequsedQ, sequsedKv, tilingData,
            &tPipe);
    op.Process();
}

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
    if constexpr (quantMode == QFA_MXFP8_FP32_PREFILL || quantMode == QFA_MXFP8_FP32_DECODE) {
        quant_flash_attn_mxfp8<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd>(
            query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
            cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
    }
    if constexpr (quantMode == QFA_GQA_FP8_FULLQUANT) {
        quant_flash_attn_gqa_fp8<fp8_e4m3fn_t, bfloat16_t, inOutLayoutType, KvLayoutType, hasAttenMask, config>(
            query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
            cuSeqLensKv, sequsedQ, sequsedKv, sinks, metadata, attnOut, softmaxLse, user, tiling);
    }
#elif (ORIG_DTYPE_Q == DT_HIFLOAT8)
    if constexpr (quantMode == QFA_HIF8_FP32) {
        quant_flash_attn_hif8<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd>(
            query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
            cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
    }
#endif
}
