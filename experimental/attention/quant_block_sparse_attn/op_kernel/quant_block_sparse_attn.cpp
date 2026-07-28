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
 * \file quant_block_sparse_attn.cpp
 * \brief QuantBlockSparseAttn kernel entry: dispatch into the kernel-side Process().
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_block_sparse_attn_template_tiling_key.h"
#include "quant_block_sparse_attn_common.h"
#include "arch35/quant_block_sparse_attn_kernel.h"

using namespace AscendC;

// 仿 kv_quant_sparse_attn_sharedkv.cpp 的 kernel 侧调用写法：
//   - 按 g_coreType 选择实/Dummy 的 Cube/Vec block（AIC 跑真 Cube + Dummy Vec，AIV 跑 Dummy Cube + 真 Vec）
//   - 实例化 QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>
//   - GET_TILING_DATA -> InitBaseAPI -> Process()
// __VA_ARGS__ 为 block 模板参数（严格对齐 CUBE_BLOCK_TRAITS 字段顺序，共 17 个）。

#define BSA_OP_IMPL(...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::BSABlockCube<__VA_ARGS__>, \
                                      BaseApi::BSABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::BSABlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::BSABlockVec<__VA_ARGS__>>::type; \
        BaseApi::QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType> op; \
        GET_TILING_DATA_WITH_STRUCT(QuantBlockSparseAttnTilingData, tilingDataIn, tiling); \
        const QuantBlockSparseAttnTilingData *__restrict tilingData = &tilingDataIn; \
        op.InitBaseAPI(query, key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqlensQ, cuSeqlensKv, \
                       seqUsedQ, seqUsedKv, blockTable, nullptr /* queryPaddingSize */, nullptr /* kvPaddingSize */, \
                       qScale /* deqScaleQ */, kScale /* deqScaleK */, vScale /* deqScaleV */, pScale /* pScale */, \
                       nullptr /* softmaxMax */, nullptr /* softmaxSum */, softmaxLse, attentionOut, \
                       user /* workspace */, tilingData, &tPipe); \
        op.Process(); \
    } while (0)

template <uint32_t QKV_DTYPE, uint32_t LAYOUT_T, uint32_t KV_LAYOUT_T, uint32_t MASK_MODE, bool RETURN_SOFTMAX_LSE>
__global__ __aicore__ void quant_block_sparse_attn(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *qScale, __gm__ uint8_t *kScale,
    __gm__ uint8_t *vScale, __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *attenMask, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // mask 模式 / softmaxLse 使能同时进入 tilingKey 与 tilingData；当前主逻辑仍通过 tilingData 读取运行期字段。
    (void)RETURN_SOFTMAX_LSE;

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    constexpr BSALayout layout = static_cast<BSALayout>(LAYOUT_T);
    constexpr BSALayout kvLayout = static_cast<BSALayout>(KV_LAYOUT_T);
    // isPa: KV 连续 -> false; PA_ND -> true
    constexpr bool bsaIsPa = true;
    constexpr bool bsaUseDn = BaseApi::IsDn();
    constexpr bool HAS_ATTENTION = (MASK_MODE == 3);
    BSA_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                S2TemplateType::Aligned256, DTemplateType::Aligned128, DTemplateType::Aligned128, HAS_ATTENTION,
                bsaIsPa, bsaUseDn);
}
