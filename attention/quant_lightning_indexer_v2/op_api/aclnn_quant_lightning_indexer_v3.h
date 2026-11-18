/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_LIGHTNING_INDEXER_V3_H
#define ACLNN_QUANT_LIGHTNING_INDEXER_V3_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算QuantLightningIndexerV3所需的workspace大小，并创建执行器。
 * @domain aclnn_ops_infer
 * @param [in] q: Query输入。
 * @param [in] k: Key输入。
 * @param [in] w: 各q token对应的head权重。
 * @param [in] qDescale: Query的反量化系数。
 * @param [in] kDescale: Key的反量化系数。
 * @param [in] cuSeqlensQOptional: 各Batch中q的Sequence Length累加前缀和。
 * @param [in] cuSeqlensKOptional: 各Batch中k的Sequence Length累加前缀和。
 * @param [in] sequsedQOptional: 各Batch中q实际参与运算的Sequence Length。
 * @param [in] sequsedKOptional: 各Batch中k实际参与运算的Sequence Length。
 * @param [in] cmpResidualKOptional: 各Batch中k压缩前Sequence Length除以cmpRatio的余数。
 * @param [in] blockTableOptional: PageAttention中KV存储使用的block映射表。
 * @param [in] outputIdxOffsetOptional: 预留的输出索引偏移量，当前须传入nullptr。
 * @param [in] metadataOptional: 由aclnnQuantLightningIndexerV3Metadata生成的分核信息。
 * @param [in] topk: 每个q token选取的k token数量。
 * @param [in] quantMode: 量化模式。
 * @param [in] maxSeqlenQ: Query的最大序列长度。
 * @param [in] layoutQOptional: Query的排列格式。
 * @param [in] layoutKOptional: Key的排列格式。
 * @param [in] maskMode: 掩码模式，0表示No mask，3表示rightDownCausal。
 * @param [in] cmpRatio: Key的压缩率，1表示不压缩。
 * @param [in] returnValue: 是否输出sparseValuesOut，当前仅支持0，表示不输出。
 * @param [in] candidateTopkBlocks: 候选逻辑块数量，-1表示关闭候选选择，启用时为2048。
 * @param [in] candidateBlockSize: 压缩后Key空间中的候选块大小，与candidateTopkBlocks配对为(-1, -1)或(2048, 8)。
 * @param [out] sparseIndicesOut: TopK位置索引输出，INT32类型。
 * @param [out] sparseValuesOut: TopK索引对应的分数，BFLOAT16类型；returnValue为0时为空Tensor。
 * @param [out] candidateBlockIndicesOut: 候选逻辑块索引输出，INT32类型，块号相对于当前Batch的压缩Key空间。
 * @param [out] candidateBlockLengthOut: 每行实际有效候选块数，INT32类型。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantLightningIndexerV3GetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* w, const aclTensor* qDescale, const aclTensor* kDescale,
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, const aclTensor* blockTableOptional,
    const aclTensor* outputIdxOffsetOptional, const aclTensor* metadataOptional, int64_t topk, int64_t quantMode,
    int64_t maxSeqlenQ, char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t cmpRatio,
    int64_t returnValue, int64_t candidateTopkBlocks, int64_t candidateBlockSize, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, const aclTensor* candidateBlockIndicesOut,
    const aclTensor* candidateBlockLengthOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行QuantLightningIndexerV3计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspaceSize: workspace大小，由aclnnQuantLightningIndexerV3GetWorkspaceSize获取。
 * @param [in] executor: op执行器，由aclnnQuantLightningIndexerV3GetWorkspaceSize获取。
 * @param [in] stream: 指定执行计算任务的Stream。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantLightningIndexerV3(void* workspace, uint64_t workspaceSize,
                                                                                aclOpExecutor* executor,
                                                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_QUANT_LIGHTNING_INDEXER_V3_H
