/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <string.h>
#include "graph/types.h"
#include "aclnn_quant_lightning_indexer.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
// Dim Num
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

extern aclnnStatus aclnnInnerQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional, int64_t queryQuantMode,
    int64_t keyQuantMode, char *layoutQueryOptional, char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t keyStride0, int64_t keyDequantScaleStride0, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerQuantLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   const aclrtStream stream);

aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional, int64_t queryQuantMode,
    int64_t keyQuantMode, char *layoutQueryOptional, char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Query pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }

    int64_t keyStride0 = -1;
    int64_t keyDequantScaleStride0 = -1;
    if (!IsContiguous(key)) {
        auto keyStride = key->GetViewStrides();
        auto keyShape = key->GetViewShape();
        bool isPaBsnd = (layoutKeyOptional != nullptr && std::string(layoutKeyOptional) == "PA_BSND");
        size_t checkStartIdx = isPaBsnd ? 1 : 0;
        if (keyShape.GetDimNum() == DIM_NUM_FOUR) {
            int64_t expected = 1;
            for (int64_t i = 3; i >= static_cast<int64_t>(checkStartIdx); --i) {
                if (keyStride[i] != expected) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "Key only supports non-contiguous tensor on the 0-axis in PA scenarios. "
                            "axis[%ld] stride=%ld, expected=%ld",
                            i, (long)keyStride[i], (long)expected);
                    return ACLNN_ERR_PARAM_INVALID;
                }
                expected *= keyShape.GetDim(i);
            }
        }
        keyStride0 = keyStride[0];
    }
    if (!IsContiguous(keyDequantScale)) {
        auto scaleStride = keyDequantScale->GetViewStrides();
        auto scaleShape = keyDequantScale->GetViewShape();
        bool isPaBsnd = (layoutKeyOptional != nullptr && std::string(layoutKeyOptional) == "PA_BSND");
        size_t checkStartIdx = isPaBsnd ? 1 : 0;
        if (scaleShape.GetDimNum() == DIM_NUM_THREE) {
            int64_t expected = 1;
            for (int64_t i = 2; i >= static_cast<int64_t>(checkStartIdx); --i) {
                if (scaleStride[i] != expected) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "Key_dequant_scale only supports non-contiguous tensor on the 0-axis in PA scenarios. "
                            "axis[%ld] stride=%ld, expected=%ld",
                            i, static_cast<long>(scaleStride[i]), static_cast<long>(expected));
                    return ACLNN_ERR_PARAM_INVALID;
                }
                expected *= scaleShape.GetDim(i);
            }
        }
        keyDequantScaleStride0 = scaleStride[0];
    }

    return aclnnInnerQuantLightningIndexerGetWorkspaceSize(
        query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
        actualSeqLengthsKeyOptional, blockTableOptional, queryQuantMode, keyQuantMode, layoutQueryOptional,
        layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, keyStride0, keyDequantScaleStride0, out,
        workspaceSize, executor);
}

aclnnStatus aclnnQuantLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    return aclnnInnerQuantLightningIndexer(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
