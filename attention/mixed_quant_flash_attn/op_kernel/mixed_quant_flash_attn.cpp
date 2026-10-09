/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file mixed_quant_flash_attn.cpp
 * \brief MixedQuantFlashAttn Kernel Entry
 */

#include "kernel_operator.h"
#include "../../common/op_kernel/arch_info.h"
#include "utils/mixed_quant_flash_attn_utils.h"
#include "mixed_quant_flash_attn_template_tiling_key.h"
#include "mixed_quant_flash_attn_tiling_data.h"
#include "arch92/mixed_quant_flash_attn_kernel_gqa.h"

using namespace AscendC;

template <uint8_t inOutLayoutType, uint8_t kvLayoutType, bool hasAttenMask, uint8_t config, uint8_t quantComputeMode>
__global__ __aicore__ void mixed_quant_flash_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* kDescale,
    __gm__ uint8_t* vDescale, __gm__ uint8_t* blockTable, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* seqUsedQ,
    __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(optiling::MixedQuantFlashAttnTilingData);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    // dc_preload(reinterpret_cast<__gm__ uint64_t*>(seqUsedQ), 0);
    dc_preload(reinterpret_cast<__gm__ uint64_t*>(seqUsedKv), 0);
    // dc_preload(reinterpret_cast<__gm__ uint64_t*>(tiling), 0);
    // if ASCEND_IS_AIC {
    //     uint32_t aicIdx = GetBlockIdx();
    // dc_preload(reinterpret_cast<__gm__ uint64_t*>(metadata), 4*(16+16*aicIdx));
    // }

    fa_base_matmul::idCounterNum = 0;

    GET_TILING_DATA_PTR_WITH_STRUCT(optiling::MixedQuantFlashAttnTilingData, tilingData, tiling);

    AscendC::InitSocState();

#if (ORIG_DTYPE_Q == DT_BF16)
    using Q_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    using Q_T = half;
    using OUT_T = half;
#endif
    constexpr LayOutTypeEnum qLayout = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outLayout = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);
    using KV_T = typename TypeLookup<Q_T, quantComputeMode>::kv_dtype;
    using KVSCALE_T = typename TypeLookup<Q_T, quantComputeMode>::kvscale_dtype;
    constexpr bool useDn = false;
    constexpr bool pageAttention = (kvLayoutType != KvLayoutType_NO_PA);

    using MQFA_T = MQFAType<Q_T, OUT_T, KV_T, KVSCALE_T, pageAttention, qLayout, kvLayoutType, outLayout,
                            s1TemplateType, s2TemplateType, dTemplateType, hasAttenMask, useDn, quantComputeMode>;

    using CubeBlock = BaseApi::FAAntiQuantGqaBlockCube<MQFA_T>;
    using VecFaBlock = BaseApi::FAAntiQuantGqaBlockVec<MQFA_T>;
    using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<MQFA_T>;
    using CubeBlockDummy = BaseApi::CubeBlockBase<MQFA_T>;
    using VecBlockDummy = BaseApi::VecBlockBase<MQFA_T>;
    using VecFdBlockDummy = BaseApi::FiaBlockVecFlashDecodeBase<MQFA_T>;

#ifdef __DAV_CUBE__
    using Kernel = BaseApi::FlashAttentionAntiQuantGqaKernel<CubeBlock, VecBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::FlashAttentionAntiQuantGqaKernel<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    Kernel op;
    // printf("ly debug 004\n");
    op.Init(query, key, value, kDescale, vDescale, blockTable, cuSeqLensQ, nullptr, // cuSeqLenskv placeholder
            seqUsedQ, seqUsedKv, sinks, attnMask, attnOut, softmaxLse, user, metadata, &tilingData->baseTiling);
    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
}
