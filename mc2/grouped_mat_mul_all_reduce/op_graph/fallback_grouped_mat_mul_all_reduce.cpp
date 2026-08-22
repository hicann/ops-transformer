/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback.h"
#include "common/utils/op_mc2.h"
#include "mc2_common_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

enum GMMAllReduceInOutIdx : size_t {
    INPUT_X,
    INPUT_WEIGHT,
    INPUT_BIAS,
    INPUT_GROUP_LIST,
    INPUT_CONTEXT,
    OUTPUT_Y = 0
};

enum GMMAllReduceAttrIdx : size_t {
    K_SPLIT_ITEM,
    K_GROUP,
    K_OP,
    K_COMM_TURN,
};

static graphStatus PrepareInputTensorListGMMAllReduce(OpExecuteContext *host_api_ctx,
                                                      std::vector<const gert::Tensor *> &tensorList, size_t index,
                                                      size_t &num)
{
    while (1) {
        auto inputGe = host_api_ctx->GetDynamicInputTensor(index, num);
        if (inputGe == nullptr) {
            break;
        }
        tensorList.push_back(inputGe);
        num++;
    }
    return GRAPH_SUCCESS;
}

static graphStatus PrepareOutputTensorListGMMAllReduce(OpExecuteContext *host_api_ctx,
                                                       std::vector<const gert::Tensor *> &tensorList, size_t index,
                                                       size_t numGeWeight, int32_t splitItem)
{
    size_t numGeY = 0;
    if (0 == splitItem || 1 == splitItem) {
        // Length of tensorListY equals that of tensorListWeight when split_item = 0 / 1
        numGeY = numGeWeight;
    } else if (2 == splitItem || 3 == splitItem) { // Length of tensorListY equals 1 when split_item = 2 / 3
        numGeY = 1;
    } else {
        OP_LOGE("gmm_all_reduce_fallback", "Invalid value of split_item: %d, which must be one of 0/1/2/3.", splitItem);
        return GRAPH_FAILED;
    }

    for (size_t k = 0; k < numGeY; k++) {
        auto outputGe = host_api_ctx->GetOutputTensor(index + k);
        if (outputGe == nullptr) {
            return GRAPH_FAILED;
        }
        tensorList.push_back(outputGe);
    }
    return GRAPH_SUCCESS;
}

static graphStatus GroupedMatMulAllReduceExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OPS_ERR_IF(host_api_ctx == nullptr, OP_LOGE_WITH_INVALID_INPUT("gmm_all_reduce_fallback", "host_api_ctx"),
               return GRAPH_FAILED);
    std::vector<const gert::Tensor *> geTensorListX;
    size_t numGeX = 0;
    PrepareInputTensorListGMMAllReduce(host_api_ctx, geTensorListX, GMMAllReduceInOutIdx::INPUT_X, numGeX);
    std::vector<const gert::Tensor *> geTensorListWeight;
    size_t numGeWeight = 0;
    PrepareInputTensorListGMMAllReduce(host_api_ctx, geTensorListWeight, GMMAllReduceInOutIdx::INPUT_WEIGHT,
                                       numGeWeight);
    std::vector<const gert::Tensor *> geTensorListBias;
    size_t numGeBias = 0;
    PrepareInputTensorListGMMAllReduce(host_api_ctx, geTensorListBias, GMMAllReduceInOutIdx::INPUT_BIAS, numGeBias);

    auto groupListGe = host_api_ctx->GetOptionalInputTensor(GMMAllReduceInOutIdx::INPUT_GROUP_LIST);
    std::vector<int64_t> shape;
    if (groupListGe != nullptr) {
        auto groupListDataPtr = groupListGe->GetData<int64_t>();
        auto &groupListShape = groupListGe->GetStorageShape();
        if (groupListShape.GetDimNum() > 0) {
            auto groupListLength = groupListShape.GetDim(0);
            for (int64_t i = 0; i < groupListLength; i++) {
                int64_t groupListData = groupListDataPtr[i];
                shape.push_back(groupListData);
            }
        }
    }

    auto attrs = host_api_ctx->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT("gmm_all_reduce_fallback", "attrs"), return GRAPH_FAILED);
    const int64_t *splitItem = attrs->GetAttrPointer<int64_t>(GMMAllReduceAttrIdx::K_SPLIT_ITEM);
    OPS_ERR_IF(splitItem == nullptr, OP_LOGE_WITH_INVALID_INPUT("gmm_all_reduce_fallback", "splitItem"),
               return GRAPH_FAILED);
    const char *group = attrs->GetStr(static_cast<size_t>(GMMAllReduceAttrIdx::K_GROUP));
    OPS_ERR_IF(group == nullptr, OP_LOGE_WITH_INVALID_INPUT("gmm_all_reduce_fallback", "group"),
               return ge::GRAPH_FAILED);
    const char *op = attrs->GetStr(static_cast<size_t>(GMMAllReduceAttrIdx::K_OP));
    OPS_ERR_IF(op == nullptr, OP_LOGE_WITH_INVALID_INPUT("gmm_all_reduce_fallback", "op"), return ge::GRAPH_FAILED);
    const int64_t *comm_turn_ptr = attrs->GetInt(static_cast<size_t>(GMMAllReduceAttrIdx::K_COMM_TURN));
    const int64_t comm_turn = (comm_turn_ptr != nullptr ? *comm_turn_ptr : 0);
    const int64_t stream_mode = 1; // 1  STOP_ON_FAILURE

    std::vector<const gert::Tensor *> geTensorListY;
    PrepareOutputTensorListGMMAllReduce(host_api_ctx, geTensorListY, GMMAllReduceInOutIdx::OUTPUT_Y, numGeWeight,
                                        *splitItem);

    // execute opapi
    auto groupListIntArray = ConvertType(shape);
    auto api_ret = EXEC_OPAPI_CMD(aclnnGroupedMatMulAllReduce, geTensorListX, geTensorListWeight, geTensorListBias,
                                  groupListIntArray, *splitItem, group, op, comm_turn, stream_mode, geTensorListY);
    OPS_ERR_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("gmm_all_reduce_fallback", "api_ret failed:%u", api_ret),
               return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatMulAllReduce)
    .OpExecuteFunc(GroupedMatMulAllReduceExecuteFunc)
    .HostInputs({GMMAllReduceInOutIdx::INPUT_GROUP_LIST});
} // namespace fallback

#ifdef __cplusplus
}
#endif
