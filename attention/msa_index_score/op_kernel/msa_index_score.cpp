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
 * \file msa_index_score.cpp
 * \brief MsaIndexScore kernel 入口。
 *
 * 入参顺序与 op_def / aclnn 对齐：
 *   query, key, start_loc, block_table, query_scale, key_scale, atten_mask, actual_seq_qlen, actual_seq_klen, score,
 *   sparse_indices, sparse_values, workspace, tiling
 * atten_mask 仅做 host 校验；device 侧按 sparse_mode 解析因果，不消费该 GM。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "msa_index_score_tiling_key.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/msa_index_score_kernel_arch35.h"
#include "arch35/msa_mxfp4_kernel.h"
#include "arch35/fp8_pipeline/msa_fp8_pipeline_adapter.h"
#else
#include "arch22/msa_index_score_kernel_arch22.h"
#endif

using namespace MsaIndexScoreNs;

#define MSA_INVOKE_KERNEL(ElementQ, IsQuant) \
    do { \
        MsaIndexScoreKernel<ElementQ, IsQuant> op; \
        op.Init(query, key, blockTable, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, sparseIndices, \
                sparseValues, userWs, tilingData); \
        op.Process(); \
    } while (0)

// BSND（M3）走独立实例化，128 路径的 MsaIndexScoreKernel<ElementQ, IsQuant> 保持原样。
#define MSA_INVOKE_KERNEL_BLK16(ElementQ) \
    do { \
        MsaIndexScoreKernel<ElementQ, false, true> op; \
        op.Init(query, key, blockTable, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, sparseIndices, \
                sparseValues, userWs, tilingData); \
        op.Process(); \
    } while (0)

#define MSA_INVOKE_KERNEL_TOPK(ElementQ) \
    do { \
        MsaIndexScoreKernel<ElementQ, false, false, (MSA_A5_USE_C2UB != 0), false, false, true> op; \
        op.Init(query, key, blockTable, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, sparseIndices, \
                sparseValues, userWs, tilingData); \
        op.Process(); \
    } while (0)

#define MSA_INVOKE_KERNEL_COMPRESSED_PAGE(ElementQ) \
    do { \
        MsaIndexScoreKernel<ElementQ, false, true, (MSA_A5_USE_C2UB != 0), false, false, false, false, false, true> \
            op; \
        op.Init(query, key, blockTable, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, sparseIndices, \
                sparseValues, userWs, tilingData); \
        op.Process(); \
    } while (0)

#define MSA_INVOKE_KERNEL_FP8_PIPELINED(ElementQ, Fused, Balanced, Bypass) \
    do { \
        if (!RunNativeFp8Pipeline<ElementQ, Fused, Balanced, Bypass>(query, key, blockTable, actualSeqQlen, \
                                                                     actualSeqKlen, score, userWs, sparseIndices, \
                                                                     sparseValues, tilingData)) { \
            MSA_INVOKE_KERNEL_COMPRESSED_PAGE(ElementQ); \
        } \
    } while (0)

#define MSA_INVOKE_KERNEL_DIRECT_FLOAT(ElementQ) \
    do { \
        MsaIndexScoreKernel<ElementQ, false, true, (MSA_A5_USE_C2UB != 0), false, false, false, false, false, true, \
                            true> \
            op; \
        op.Init(query, key, blockTable, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, sparseIndices, \
                sparseValues, userWs, tilingData); \
        op.Process(); \
    } while (0)

template <uint32_t schMode>
__global__ __aicore__ void msa_index_score(GM_ADDR query, GM_ADDR key, GM_ADDR startLoc, GM_ADDR blockTable,
                                           GM_ADDR queryScale, GM_ADDR keyScale, GM_ADDR attenMask,
                                           GM_ADDR actualSeqQlen, GM_ADDR actualSeqKlen, GM_ADDR score,
                                           GM_ADDR sparseIndices, GM_ADDR sparseValues, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)attenMask; // host 校验；device 按 sparse_mode 解析 rightDownCausal
    // 950 非量化 PA：S 在 UB、无 K scratch，user workspace 可能为 0。不能因此跳过整个 kernel 流程。
    GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        userWs = workspace;
    }
    GET_TILING_DATA_WITH_STRUCT(MsaIndexScoreTilingData, tilingDataIn, tiling);
    const MsaIndexScoreTilingData* __restrict tilingData = &tilingDataIn;

    if constexpr (schMode == MSA_TILING_KEY_BF16) {
        MSA_INVOKE_KERNEL(bfloat16_t, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP16) {
        MSA_INVOKE_KERNEL(half, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP16_INT8) {
        MSA_INVOKE_KERNEL(half, true);
#if (__CCE_AICORE__ == 310)
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN) {
        MSA_INVOKE_KERNEL(fp8_e4m3fn_t, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2) {
        MSA_INVOKE_KERNEL(fp8_e5m2_t, false);
    } else if constexpr (schMode == MSA_TILING_KEY_HIFLOAT8) {
        MSA_INVOKE_KERNEL(hifloat8_t, false);
    } else if constexpr (schMode == MSA_TILING_KEY_BF16_BLK16) {
        MSA_INVOKE_KERNEL_BLK16(bfloat16_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP16_BLK16) {
        MSA_INVOKE_KERNEL_BLK16(half);
    } else if constexpr (schMode == MSA_TILING_KEY_BF16_TOPK) {
        MSA_INVOKE_KERNEL_TOPK(bfloat16_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP16_TOPK) {
        MSA_INVOKE_KERNEL_TOPK(half);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_TOPK) {
        MSA_INVOKE_KERNEL_TOPK(fp8_e5m2_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_TOPK) {
        MSA_INVOKE_KERNEL_TOPK(fp8_e4m3fn_t);
    } else if constexpr (schMode == MSA_TILING_KEY_MXFP4_TOPK) {
        MsaMxFp4Kernel<true> op;
        op.Init(query, key, blockTable, queryScale, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, userWs,
                tilingData, sparseIndices, sparseValues);
        op.ProcessFp4();
    } else if constexpr (schMode == MSA_TILING_KEY_MXFP4) {
        MsaMxFp4Kernel<> op;
        op.Init(query, key, blockTable, queryScale, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, userWs,
                tilingData);
        op.ProcessFp4();
    } else if constexpr (schMode == MSA_TILING_KEY_MXFP4_BLK16) {
        MsaMxFp4Kernel<> op;
        op.Init(query, key, blockTable, queryScale, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, userWs,
                tilingData, sparseIndices, sparseValues);
        op.ProcessBlock16();
    } else if constexpr (schMode == MSA_TILING_KEY_MXFP4_PREFILL) {
        MsaMxFp4Kernel<> op;
        op.Init(query, key, blockTable, queryScale, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, userWs,
                tilingData);
        op.ProcessPrefill();
    } else if constexpr (schMode == MSA_TILING_KEY_MXFP4_LATENCY) {
        MsaMxFp4Kernel<> op;
        op.Init(query, key, blockTable, queryScale, keyScale, actualSeqQlen, actualSeqKlen, startLoc, score, userWs,
                tilingData);
        op.ProcessLatency();
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_BLK16) {
        MSA_INVOKE_KERNEL_BLK16(fp8_e4m3fn_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_COMPRESSED_PAGE) {
        MSA_INVOKE_KERNEL_COMPRESSED_PAGE(fp8_e4m3fn_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_PIPELINED) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, false, false, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_PIPELINED_TOPK) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, true, false, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_BALANCED) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, false, true, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_BALANCED_TOPK) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, true, true, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_PIPELINED_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, false, false, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_PIPELINED_TOPK_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, true, false, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_BALANCED_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, false, true, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E4M3FN_BALANCED_TOPK_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e4m3fn_t, true, true, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_BLK16) {
        MSA_INVOKE_KERNEL_BLK16(fp8_e5m2_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_COMPRESSED_PAGE) {
        MSA_INVOKE_KERNEL_COMPRESSED_PAGE(fp8_e5m2_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_PIPELINED) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, false, false, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_PIPELINED_TOPK) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, true, false, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_BALANCED) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, false, true, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_BALANCED_TOPK) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, true, true, false);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_PIPELINED_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, false, false, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_PIPELINED_TOPK_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, true, false, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_BALANCED_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, false, true, true);
    } else if constexpr (schMode == MSA_TILING_KEY_FP8_E5M2_BALANCED_TOPK_BYPASS) {
        MSA_INVOKE_KERNEL_FP8_PIPELINED(fp8_e5m2_t, true, true, true);
    } else if constexpr (schMode == MSA_TILING_KEY_BF16_COMPRESSED_PAGE) {
        MSA_INVOKE_KERNEL_COMPRESSED_PAGE(bfloat16_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP16_COMPRESSED_PAGE) {
        MSA_INVOKE_KERNEL_COMPRESSED_PAGE(half);
    } else if constexpr (schMode == MSA_TILING_KEY_BF16_DIRECT_FLOAT) {
        MSA_INVOKE_KERNEL_DIRECT_FLOAT(bfloat16_t);
    } else if constexpr (schMode == MSA_TILING_KEY_FP16_DIRECT_FLOAT) {
        MSA_INVOKE_KERNEL_DIRECT_FLOAT(half);
#endif
    }
}
