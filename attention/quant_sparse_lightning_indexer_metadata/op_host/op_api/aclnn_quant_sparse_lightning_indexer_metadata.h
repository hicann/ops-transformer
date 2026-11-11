/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_LIGHTNING_INDEXER_V2_METADATA_H
#define ACLNN_QUANT_LIGHTNING_INDEXER_V2_METADATA_H

#include <string>
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算 QuantSparseLightningIndexer(两级 TopK 第二级 consumer 算子)所需的分核
 *        信息(metadata), 输出固定长度 1024 的 INT32 张量, 作为该算子的 metadata 入参。
 *
 * 支持平台: Atlas A2 / Atlas A3(910B / 910_93, arch22)。
 *
 * @param [in] cuSeqlensQOptional q 各 batch 的前缀和, INT32, shape [B+1]; layout_q 为 TND 时必传。
 * @param [in] cuSeqlensKOptional k 各 batch 的前缀和, INT32, shape [B+1]; layout_k 为 TND 时必传。
 * @param [in] sequsedQOptional   q 每 batch 的有效长度, INT32, shape [B]; 可选。
 * @param [in] sequsedKOptional   k 每 batch 的压缩后有效长度, INT32, shape [B]; layout_k 为
 *                                PA_BBND(分页注意力布局)时必传。
 * @param [in] cmpResidualKOptional k 的压缩残余长度, INT32, shape [B], 取值 [0, cmpRatio);
 *                                仅当 cmpRatio > 1 且 maskMode 为 3(因果掩码) 时必传, 否则必须
 *                                为空。传 nullptr 或空张量表示按 0 处理。
 * @param [in] candidateBlockLengthOptional 预留接口, 仅接受空张量; 传非空张量将返回参数错误。
 * @param [in] numHeadsQ          q 侧头数 N1。
 * @param [in] numHeadsK          k 侧头数 N2, 仅支持 1。
 * @param [in] headDim            头维, 仅支持 128。
 * @param [in] topk               第二级 TopK 的 k 值, 取值 [1, 2048]; 用于分核信息中的
 *                                FD(分核信息驱动的动态分核)门控判定。
 * @param [in] quantMode          量化模式, 仅支持 2(INT8 量化)。
 * @param [in] batchSize          batch 数 B; 必须与 q/k 侧推导的 batch 数一致。
 * @param [in] maxSeqlenQ         q 的最大序列长度; layout_q 为 BSND 时必须等于 q 第 1 轴长度。
 * @param [in] maxSeqlenK         k 的最大序列长度(压缩后口径); layout_k 为 BSND 时必须等于
 *                                k 第 1 轴长度。
 * @param [in] layoutQOptional    q 的布局, "BSND" 或 "TND"; 默认 "BSND"。
 * @param [in] layoutKOptional    k 的布局, 仅支持 "PA_BBND"(按 block_table 分页寻址)。
 * @param [in] maskMode           掩码模式, 0(无掩码) 或 3(rightDownCausal, 因果掩码)。
 * @param [in] cmpRatio           k 的压缩倍数, 取值 (0, 128] 且必须为 2 的幂; 默认 1(不压缩)。
 * @param [in] candidateBlockSize 候选块粒度(每块位置数), 当前仅支持 8。
 * @param [out] metadata          输出分核信息, INT32, shape 固定 [1024]; 非空, 作为
 *                                QuantSparseLightningIndexer 的 metadata 入参原样传入。
 * @param [out] workspaceSize     返回所需 workspace 字节数。
 * @param [out] executor          返回算子执行器上下文。
 * @return ACLNN_SUCCESS 成功; 其它为错误码(参数非法时返回 ACLNN_ERR_PARAM_INVALID)。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantSparseLightningIndexerMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional,
    const aclTensor* candidateBlockLengthOptional, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
    int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, char* layoutQOptional,
    char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t candidateBlockSize, const aclTensor* metadata,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行分核信息计算(第二阶段调用, 需在 GetWorkspaceSize 之后)。
 * @param [in] workspace      workspace 指针(大小取 GetWorkspaceSize 的返回值)。
 * @param [in] workspaceSize  workspace 字节数。
 * @param [in] executor       算子执行器上下文(GetWorkspaceSize 的输出)。
 * @param [in] stream         执行的 ACL stream。
 * @return ACLNN_SUCCESS 成功; 其它为错误码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantSparseLightningIndexerMetadata(void* workspace,
                                                                                            uint64_t workspaceSize,
                                                                                            aclOpExecutor* executor,
                                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_QUANT_LIGHTNING_INDEXER_V2_METADATA_AICPU_H
