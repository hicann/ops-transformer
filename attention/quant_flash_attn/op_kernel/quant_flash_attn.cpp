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
#include "../../../common/op_kernel/fia_public_define.h"
#include "util.h"
#include "../../../common/op_kernel/vector_common.h"
#include "arch35/quant_flash_attn_kernel_mxfp8.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_tiling_data.h"

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

    fa_base_matmul::idCounterNum = 0;

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    constexpr bool isFdConst = false;
    constexpr PseTypeEnum pseModeConst = PseTypeEnum::PSE_NONE_TYPE;
    constexpr bool enableKVPrefixConst = false;

    constexpr TPosition bmm2OutPos =
        BaseApi::GetC2Position(dVTemplateType,
                               BaseApi::UbOutCondition<INPUT_T>(false, pseModeConst, hasAttenMask, false, false,
                                                                (uint32_t)s1TemplateType == 64),
                               ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn = (quantMode == QFA_MXFP8_FP32_PREFILL);
    constexpr bool bmm2Write2Ub = (bmm2OutPos == TPosition::VECCALC);
    constexpr bool splitD = ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256);

    using CubeBlock =
        BaseApi::FAFullQuantMxBlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                                        dVTemplateType, KvLayoutType, enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock =
        BaseApi::FAFullQuantMxBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                       s2TemplateType, dTemplateType, dVTemplateType, pseModeConst, hasAttenMask, false,
                                       KvLayoutType, isFdConst, enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock =
        BaseApi::FiaBlockVecFlashDecodeFullQuant<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                 s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                 pseModeConst, hasAttenMask, false, KvLayoutType, enableKVPrefixConst,
                                                 useDn, bmm2Write2Ub, splitD>;

    using CubeBlockDummy =
        BaseApi::FAFullQuantMxBlockCubeDummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                             dTemplateType, dVTemplateType, KvLayoutType, enableKVPrefixConst, useDn,
                                             bmm2Write2Ub, splitD>;
    using VecFaBlockDummy =
        BaseApi::FAFullQuantMxBlockVecDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                            s2TemplateType, dTemplateType, dVTemplateType, pseModeConst, hasAttenMask,
                                            false, KvLayoutType, isFdConst, enableKVPrefixConst, useDn, bmm2Write2Ub,
                                            splitD>;
    using VecFdBlockDummy =
        BaseApi::FiaBlockVecFlashDecodeFullQuantDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                      s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                      pseModeConst, hasAttenMask, false, KvLayoutType,
                                                      enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::FlashAttentionFullQuantMxKernel<CubeBlock, VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::FlashAttentionFullQuantMxKernel<CubeBlockDummy, VecFaBlock, VecFdBlock>;
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
#endif
}
