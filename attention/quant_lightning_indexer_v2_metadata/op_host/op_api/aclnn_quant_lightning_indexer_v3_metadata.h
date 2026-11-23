/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_LIGHTNING_INDEXER_V3_METADATA_H
#define ACLNN_QUANT_LIGHTNING_INDEXER_V3_METADATA_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算QuantLightningIndexerV3Metadata所需的workspace大小，并创建执行器。
 * @domain aclnn_ops_infer
 * @param [in] cuSeqlensQOptional: 各Batch中q的Sequence Length累加前缀和，INT32类型，shape为(B+1,)。
 * @param [in] cuSeqlensKOptional: 各Batch中k的Sequence Length累加前缀和，INT32类型。
 * @param [in] sequsedQOptional: 各Batch中q实际参与运算的Sequence Length，INT32类型，shape为(B,)。
 * @param [in] sequsedKOptional: 各Batch中k实际参与运算的Sequence Length，INT32类型，shape为(B,)。
 * @param [in] cmpResidualKOptional: 各Batch中k压缩前Sequence Length除以cmpRatio的余数，INT32类型，shape为(B,)。
 * @param [in] causalSeqlensKvGlobalOptional: DCP 每请求全局 exclusive 因果上界 C（[B] int32）。
 * @param [in] numHeadsQ: q的head个数。
 * @param [in] numHeadsK: k的head个数。
 * @param [in] headDim: 每个head的维度。
 * @param [in] topk: 每个q token选取的k token数量。
 * @param [in] quantMode: 量化模式。
 * @param [in] batchSize: Batch数量，须与序列输入的Batch数量一致。
 * @param [in] maxSeqlenQ: q的最大Sequence Length，-1表示不限制长度。
 * @param [in] maxSeqlenK: k的最大Sequence Length，-1表示不限制长度。
 * @param [in] layoutQOptional: q的排列格式。
 * @param [in] layoutKOptional: k的排列格式。
 * @param [in] maskMode: 掩码模式，0表示No mask，3表示rightDownCausal。
 * @param [in] cmpRatio: k的压缩率，1表示不压缩。
 * @param [in] candidateTopkBlocks: 候选逻辑块数量，-1表示关闭候选选择，启用时为2048。
 * @param [in] candidateBlockSize: 候选逻辑块大小，与candidateTopkBlocks配对为(-1, -1)或(2048, 8)。
 * @param [in] dcpParams: DCP（Decode Context Parallel，解码上下文并行）参数列表，固定顺序
 *            [cpWorld, cpRank, cpKvCacheInterleaveSize]；传 nullptr 表示非 DCP（等价于 {1, 0, 1}）。
 * @param [out] metadata: 分核信息输出，INT32类型，shape为(1024,)，供aclnnQuantLightningIndexerV3使用。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional,
    const aclTensor* causalSeqlensKvGlobalOptional, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
    int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, char* layoutQOptional,
    char* layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t candidateTopkBlocks, int64_t candidateBlockSize,
    const aclIntArray* dcpParams, const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行QuantLightningIndexerV3Metadata计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspaceSize: workspace大小，由aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize获取。
 * @param [in] executor: op执行器，由aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize获取。
 * @param [in] stream: 指定执行计算任务的Stream。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantLightningIndexerV3Metadata(void* workspace,
                                                                                        uint64_t workspaceSize,
                                                                                        aclOpExecutor* executor,
                                                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_QUANT_LIGHTNING_INDEXER_V3_METADATA_H
