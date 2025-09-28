/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_nsa_selected_attention_grad.h
 * \brief
 */

#ifndef OP_API_INC_NAS_SELECTED_ATTENTION_GRAD_H_
#define OP_API_INC_NAS_SELECTED_ATTENTION_GRAD_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNsaSelectedAttentionGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnNsaSelectedAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *attentionOut,
    const aclTensor *attentionOutGrad, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *topkIndices, const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclTensor *attenMaskOptional, double scaleValue, int64_t selectedBlockSize, int64_t selectedBlockCount,
    int64_t headNum, char *inputLayout, int64_t sparseMode, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnNsaSelectedAttentionGrad的第二段接口，用于执行计算。
 */
aclnnStatus aclnnNsaSelectedAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_NAS_SELECTED_ATTENTION_GRAD_H_
