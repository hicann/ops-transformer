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
 * \file lightning_indexer_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_v2_template_tiling_key.h"
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)) || (__CCE_AICORE__ == 310)
#include "arch35/lightning_indexer_v2_kernel_arch35.h"
#else
#include "arch22/lightning_indexer_v2_kernel_arch22.h"
#endif

// Legacy core-version gate: 310. NPU architecture 9201 identifies Ascend960.
// Keep the 950-only fast implementation out of the 960 compilation branch.
#if (__CCE_AICORE__ == 310) && (!defined(__NPU_ARCH__) || (__NPU_ARCH__ != 9201))
#include "arch35/lightning_indexer_v2_fast_kernel_arch35.h"
#define INVOKE_LI_V2_FAST(...) \
    do { \
        GlobalTensor<uint32_t> fastMetadata; \
        if (tiling_data->fastPathCandidate != 0U && metadata != nullptr) { \
            fastMetadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata)); \
        } \
        if (tiling_data->fastPathCandidate != 0U && metadata != nullptr && \
            fastMetadata.GetValue(optiling::LI_V2_FAST_ENABLE_INDEX) != 0U) { \
            RunLightningIndexerV2FastPath<LIV2Type<__VA_ARGS__>>(q, k, w, sparseIndices, sparseValues, user, \
                                                                 tiling_data, &tPipe, cuSeqlensQ, cuSeqlensK, \
                                                                 blocktable, sequsedK, metadata); \
            return; \
        } \
    } while (0)
#else
#define INVOKE_LI_V2_FAST(...)
#endif

using namespace LIV2Kernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<LIV2Type<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(LIV2TilingData, tiling_data_in, tiling); \
        const LIV2TilingData* __restrict tiling_data = &tiling_data_in; \
        INVOKE_LI_V2_FAST(__VA_ARGS__); \
        op.Init(q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blocktable, outputIdxOffset, \
                metadata, sparseIndices, sparseValues, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)

// candidate (two-level topk) 宏：arch22 分支专用——Init 多传 2 个 candidate 输出。
// arch35(950/960, 含 __CCE_AICORE__==310 与 __NPU_ARCH__==9201) Init 签名无 candidate 参
// （红线 1：arch35 逐字节不动），该分支继续走原宏 INVOKE_LI_NO_KFC_OP_IMPL；arch35 上本宏
// 不展开，两个 candidate 形参恒为空 tensor 指针、不被消费（host 侧 tiling 入口守卫已在
// DAV_3510 上拒绝 candidate 模式）。
#define INVOKE_LI_CANDIDATE_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<LIV2Type<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(LIV2TilingData, tiling_data_in, tiling); \
        const LIV2TilingData* __restrict tiling_data = &tiling_data_in; \
        op.Init(q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blocktable, outputIdxOffset, \
                metadata, sparseIndices, sparseValues, candidateBlockIndicesOut, candidateBlockLength, user, \
                tiling_data, &tPipe); \
        op.Process(); \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void lightning_indexer_v2(
    __gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* w, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensK,
    __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK, __gm__ uint8_t* cmpResidualK, __gm__ uint8_t* blocktable,
    __gm__ uint8_t* outputIdxOffset, __gm__ uint8_t* metadata, __gm__ uint8_t* sparseIndices,
    __gm__ uint8_t* sparseValues, __gm__ uint8_t* candidateBlockIndicesOut, __gm__ uint8_t* candidateBlockLength,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)

{
    TPipe tPipe;
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

    if (ORIG_DTYPE_Q == DT_BF16) {
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)) || (__CCE_AICORE__ == 310)
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerV2Kernel, bfloat16_t, bfloat16_t, int32_t, float, uint32_t,
                                 PAGE_ATTENTION, LI_V2_LAYOUT(LAYOUT_T), LI_V2_LAYOUT(K_LAYOUT_T), true);
#else
        INVOKE_LI_CANDIDATE_OP_IMPL(LightningIndexerV2Kernel, bfloat16_t, bfloat16_t, int32_t, float, uint32_t,
                                    PAGE_ATTENTION, LI_V2_LAYOUT(LAYOUT_T), LI_V2_LAYOUT(K_LAYOUT_T), true);
#endif
    } else if (ORIG_DTYPE_Q == DT_FLOAT16) {
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)) || (__CCE_AICORE__ == 310)
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerV2Kernel, half, half, int32_t, float, uint32_t, PAGE_ATTENTION,
                                 LI_V2_LAYOUT(LAYOUT_T), LI_V2_LAYOUT(K_LAYOUT_T), true);
#else
        INVOKE_LI_CANDIDATE_OP_IMPL(LightningIndexerV2Kernel, half, half, int32_t, float, uint32_t, PAGE_ATTENTION,
                                    LI_V2_LAYOUT(LAYOUT_T), LI_V2_LAYOUT(K_LAYOUT_T), true);
#endif
    }
}
