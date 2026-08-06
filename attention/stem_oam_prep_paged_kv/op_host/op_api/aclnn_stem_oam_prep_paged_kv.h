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
 * \file aclnn_stem_oam_prep_paged_kv.h
 * \brief
 */
#ifndef ACLNN_STEM_OAM_PREP_PAGED_KV_H
#define ACLNN_STEM_OAM_PREP_PAGED_KV_H

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnStemOamPrepPagedKv的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnStemOamPrepPagedKv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              const aclrtStream stream);

/**
 * @brief aclnnStemOamPrepPagedKv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnStemOamPrepPagedKvGetWorkspaceSize(
    const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices, const aclIntArray *kvSeqLens,
    const aclTensor *kScaleCacheOptional, const aclTensor *vScaleOptional, double lambdaMag, char *kvLayout,
    int64_t stemBlockSize, int64_t stemStride, const aclTensor *kFlat, const aclTensor *vBias,
    uint64_t *workspaceSize, aclOpExecutor **executor);

#ifdef __cplusplus
}
#endif
#endif // ACLNN_STEM_OAM_PREP_PAGED_KV_H
