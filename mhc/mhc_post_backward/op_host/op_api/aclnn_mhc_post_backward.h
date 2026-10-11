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
 * \file aclnn_mhc_post_backward.h
 * \brief MhcPostBackward ACLNN API
 */

#ifndef OP_API_INC_MHC_POST_Backward_H
#define OP_API_INC_MHC_POST_Backward_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPostBackwardGetWorkspaceSize的第一段接口，计算workspace大小。
 * 功能描述：该算子实现Manifold-Constraint Hyper-Connection的Post部分的反向
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnMhcPostBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* x, const aclTensor* hRes,
                                                 const aclTensor* hOut, const aclTensor* hPost, aclTensor* gradX,
                                                 aclTensor* gradHRes, aclTensor* gradHOut, aclTensor* gradHPost,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMhcPostBackward的第二段接口，用于执行计算。
 */
aclnnStatus aclnnMhcPostBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                 const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_POST_H
