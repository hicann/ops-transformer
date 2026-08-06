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
#include "quant_block_sparse_attn_mx_tiling_data.h"
#include "arch35/quant_block_sparse_attn_kernel.h"
#include "arch35/quant_block_sparse_attn_mx_kernel.h"

using namespace AscendC;
using namespace optiling;

// 仿 kv_quant_sparse_attn_sharedkv.cpp 的 kernel 侧调用写法：
//   - 按 g_coreType 选择实/Dummy 的 Cube/Vec block（AIC 跑真 Cube + Dummy Vec，AIV 跑 Dummy Cube + 真 Vec）
//   - 实例化 QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType>
//   - GET_TILING_DATA -> InitBaseAPI -> Process()
// __VA_ARGS__ 为 block 模板参数（严格对齐 CUBE_BLOCK_TRAITS 字段顺序）。

#define QBSA_OP_IMPL_WITH_TILING(tilingDataPtr, ...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QBSABlockCube<__VA_ARGS__>, \
                                      BaseApi::QBSABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QBSABlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::QBSABlockVec<__VA_ARGS__>>::type; \
        BaseApi::QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqlensQ, cuSeqlensKv, \
                       seqUsedQ, seqUsedKv, blockTable, nullptr /* queryPaddingSize */, nullptr /* kvPaddingSize */, \
                       qScale /* deqScaleQ */, kScale /* deqScaleK */, vScale /* deqScaleV */, pScale /* pScale */, \
                       nullptr /* softmaxMax */, nullptr /* softmaxSum */, softmaxLse, attentionOut, \
                       user /* workspace */, tilingDataPtr, &tPipe); \
        op.Process(); \
    } while (0)

#define QBSA_OP_IMPL(...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(QuantBlockSparseAttnTilingData, tilingDataIn, tiling); \
        const QuantBlockSparseAttnTilingData *__restrict tilingData = &tilingDataIn; \
        QBSA_OP_IMPL_WITH_TILING(tilingData, __VA_ARGS__); \
    } while (0)

#define QBSA_MX_OP_IMPL_WITH_TILING(tilingDataPtr, ...) \
    do { \
        /* MXFullQuantMode 读取 QuantBlockSparseAttnMxTilingData，不读取 FP8 tiling struct。 */ \
        BaseApi::QuantBlockSparseAttnMxKernel<__VA_ARGS__> op; \
        op.Init(query, key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqlensQ, cuSeqlensKv, \
                seqUsedQ, seqUsedKv, blockTable, qScale, kScale, vScale, pScale, softmaxLse, attentionOut, user, \
                tilingDataPtr, &tPipe); \
        op.Process(); \
    } while (0)

#define QBSA_MX_OP_IMPL(...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(QuantBlockSparseAttnMxTilingData, tilingDataIn, tiling); \
        const QuantBlockSparseAttnMxTilingData *__restrict tilingData = &tilingDataIn; \
        QBSA_MX_OP_IMPL_WITH_TILING(tilingData, __VA_ARGS__); \
    } while (0)

template <uint32_t QKV_DTYPE, uint32_t LAYOUT_T, uint32_t KV_LAYOUT_T, uint32_t MASK_MODE, bool RETURN_SOFTMAX_LSE,
          uint32_t Config, uint32_t QUANT_MODE>
__global__ __aicore__ void quant_block_sparse_attn(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *qScale, __gm__ uint8_t *kScale,
    __gm__ uint8_t *vScale, __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *attenMask, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // mask 模式 / softmaxLse 使能同时进入 tilingKey 与 tilingData；LSE 已改为编译期模板参数。
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    constexpr QBSALayout layout = static_cast<QBSALayout>(LAYOUT_T);
    constexpr QBSALayout kvLayout = static_cast<QBSALayout>(KV_LAYOUT_T);
    // isPa: KV 连续 -> false; PA_ND -> true
    constexpr bool bsaIsPa = true;
    constexpr bool bsaUseDn = BaseApi::IsDn();
    constexpr bool HAS_ATTENTION = (MASK_MODE == 3);
    if constexpr (QUANT_MODE == MXFullQuantMode) {
        // MX 当前支持 TND + PA BNBD，S2 logical tile 为 512。
        static_assert(Config == Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128,
                      "MXFullQuantMode must use S1=128, S2=512, D=128, DV=128 config");
        QBSA_MX_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                       S2TemplateType::Aligned512, DTemplateType::Aligned128, DTemplateType::Aligned128, HAS_ATTENTION,
                       RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
    } else {
        static_assert(Config == Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128,
                      "FP8QuantMode must use S1=128, S2=256, D=128, DV=128 config");
        QBSA_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                    S2TemplateType::Aligned256, DTemplateType::Aligned128, DTemplateType::Aligned128, HAS_ATTENTION,
                    RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
    }
}
