/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_nsa_selected_attention_grad.h
 * \brief
 */

#ifndef OP_API_INC_LIGHTNING_INDEXER_GRAD_H_
#define OP_API_INC_LIGHTNING_INDEXER_GRAD_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLightningIndexerGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(const aclTensor* query, const aclTensor* key, const aclTensor* dy,
                                                      const aclTensor* sparseIndices, const aclTensor* weights,
                                                      const aclTensor* actualSeqQLenOptional,
                                                      const aclTensor* actualSeqKvLenOptional, int64_t headNum,
                                                      char* inputLayout, int64_t sparseMode, int64_t preTokens,
                                                      int64_t nextTokens, bool deterministic, const aclTensor* dqOut,
                                                      const aclTensor* dkOut, const aclTensor* dweightsOut,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLightningIndexerGrad的第二段接口，用于执行计算。
 */
aclnnStatus aclnnLightningIndexerGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LIGHTNING_INDEXER_GRAD_H_
