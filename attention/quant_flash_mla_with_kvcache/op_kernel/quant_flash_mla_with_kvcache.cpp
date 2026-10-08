/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache.cpp
 * \brief QuantFlashMlaWithKvcache kernel入口（MLA FP8全量化KV-cache推理）
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "arch35/quant_flash_mla_with_kvcache_common_def.h"
#include "util.h"
#include "arch35/quant_flash_mla_with_kvcache_kernel_fp8.h"
#include "arch35/quant_flash_mla_with_kvcache_template_tiling_key.h"
#include "arch35/quant_flash_mla_with_kvcache_tiling_data.h"
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/vector_common.h"
#else
#include "../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../common/op_kernel/vector_common.h"
#endif

using namespace AscendC;
using namespace optiling;

// ─────────────────────────────────────────────────────────
// MLA FP8全量化: Q[fp8_e4m3, per-token-head descale] @ K[fp8_e4m3, per-tensor descale]
// rope方案A: Q/K的D维576(nope 512+rope 64)拼接, bmm1单次matmul; MLA无独立v_cache, V=K_nope
// ─────────────────────────────────────────────────────────
template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint16_t config, bool isFd>
inline __aicore__ void quant_flash_mla_with_kvcache_fp8(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                                        __gm__ uint8_t* qDescale, __gm__ uint8_t* kDescale,
                                                        __gm__ uint8_t* blockTable, __gm__ uint8_t* cacheSeqLens,
                                                        __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* sequsedQ,
                                                        __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
                                                        __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse,
                                                        __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    fa_base_matmul::ResetIdCounter();

    // 输入layout固定TND, 输出layout由inOutLayoutType编码（BSND输出与BSH模板等价）
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    using CubeBlock = BaseApi::QuantFlashMlaBlockCubeFp8<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                         s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType>;
    using VecFaBlock = BaseApi::QuantFlashMlaBlockVecFp8<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                         s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                         hasAttenMask, KvLayoutType, isFd>;
    using VecFdBlock =
        BaseApi::QuantFlashMlaBlockVecFlashDecodeFp8<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                     s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType>;

    using CubeBlockDummy =
        BaseApi::QuantFlashMlaBlockCubeFp8Dummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                                dTemplateType, dVTemplateType, KvLayoutType>;
    using VecFaBlockDummy =
        BaseApi::QuantFlashMlaBlockVecFp8Dummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                               s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask,
                                               KvLayoutType, isFd>;
    using VecFdBlockDummy =
        BaseApi::QuantFlashMlaBlockVecFlashDecodeFp8Dummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                          s1TemplateType, s2TemplateType, dTemplateType,
                                                          dVTemplateType>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::QuantFlashMlaKernelFp8<CubeBlock, VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::QuantFlashMlaKernelFp8<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    // Static online compilation passes nullptr here and supplies constant tiling via this macro.
    GET_TILING_DATA(tilingData, tiling);

    // TPipe仅用于初始化全局g_tPipePtr，使vec block的GetTPipePtr()->InitBuffer(vselrIndexesBuf_)可用
    // 不传递给kernel Init，所有主buffer使用静态LocalTensor
    TPipe tPipe;
    Kernel op;
    op.Init(query, key, qDescale, kDescale, blockTable, cacheSeqLens, cuSeqLensQ, sequsedQ, attnMask, metadata, attnOut,
            softmaxLse, workspace, tilingData);
    op.Process();
}

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__global__ __aicore__ void quant_flash_mla_with_kvcache(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                                        __gm__ uint8_t* qDescale, __gm__ uint8_t* kDescale,
                                                        __gm__ uint8_t* blockTable, __gm__ uint8_t* cacheSeqLens,
                                                        __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* sequsedQ,
                                                        __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
                                                        __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse,
                                                        __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashMlaWithKvcacheTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
    if constexpr (quantMode == QMLA_MLA_FP8_E4M3_FULLQUANT) {
        quant_flash_mla_with_kvcache_fp8<fp8_e4m3fn_t, bfloat16_t, inOutLayoutType, KvLayoutType, hasAttenMask, config,
                                         isFd>(query, key, qDescale, kDescale, blockTable, cacheSeqLens, cuSeqLensQ,
                                               sequsedQ, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
    }
#endif
}
