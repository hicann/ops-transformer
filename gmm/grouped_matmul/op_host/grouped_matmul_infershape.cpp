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
 * \file grouped_matmul_infershape.cpp
 * \brief
 */
#include <algorithm>
#include "gmm/common/op_host/log_format_util.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "grouped_matmul_infershape_weight_quant_checker.h"
#include "grouped_matmul_infershape_quant_checker.h"
#include "grouped_matmul_infershape_common_util.h"

using namespace ge;
namespace ops {

static std::set<std::string> GmmDavidSupportSoc = {"Ascend950"};

enum class PlatformID : std::uint8_t {
    UNKNOWN,
    ASCEND310P,
    ASCEND910B,
    ASCEND950
};

struct GMMParamsInfo {
    size_t numX;
    size_t numWeight;
    size_t numY;
    int64_t lenGroupList;
    size_t groupNum;
    size_t numScale;
    size_t numOffset;
    size_t numAntiquantScale;
    size_t numAntiquantOffset;
    PlatformID platform;
};

struct GMMSetOutputParams {
    bool isSingleX;
    bool isSingleY;
    size_t xDimM;
    size_t weightDimN;
    int64_t lenGroupList;
    size_t numWeight;
    size_t numX;
};

static inline std::string ToString(const std::int64_t value)
{
    return std::to_string(value);
}

static ge::graphStatus CheckSplitItem(int64_t splitItem)
{
    if (splitItem == GMM_X_Y_SEPARATED || splitItem == GMM_NO_SEPARATED || splitItem == GMM_X_SEPARATED ||
        splitItem == GMM_Y_SEPARATED) {
        return GRAPH_SUCCESS;
    } else {
        return GRAPH_FAILED;
    }
}

static bool IsTensorListNullOrEmpty(const gert::InferShapeContext *context, size_t index)
{
    auto shape = context->GetDynamicInputShape(index, 0);
    if (shape == nullptr) {
        return true;
    }
    if (shape->GetDimNum() == 0 || (shape->GetDimNum() == 1 && shape->GetDim(0) == 0)) {
        if (context->GetDynamicInputShape(index, 1) == nullptr) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckGroupType(const gert::InferShapeContext *context, int64_t groupType)
{
    if (groupType == GMM_NO_SPLIT || groupType == GMM_SPLIT_M || groupType == GMM_SPLIT_K) {
        return GRAPH_SUCCESS;
    } else if (groupType == GMM_SPLIT_N) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groupType", std::to_string(groupType),
                                              "Splitting tensor along the N-axis is not supported yet");
        return GRAPH_FAILED;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "groupType", std::to_string(groupType),
            Ops::Transformer::Gmm::FormatString("GroupType can only be -1/0/2 now", groupType).c_str());
        return GRAPH_FAILED;
    }
}

static ge::graphStatus UpdateShapeYMultiDim(gert::InferShapeContext *context, size_t idxY, const gert::Shape *xShape,
                                            const gert::Shape *weightShape)
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;
    size_t dimY = yShape->GetDimNum();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_X);

    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    if (transposeWPtr != nullptr && *transposeWPtr) {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 2)); // -2: transpose weight
    } else {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 1));
    }
    if (transposeXPtr != nullptr && *transposeXPtr) {
        yShape->SetDim(dimY - 2, xShape->GetDim(xShape->GetDimNum() - 1)); // -2: last two dim of Y
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateShapeY(gert::InferShapeContext *context, size_t idxY, std::vector<int64_t> yDims)
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateMultipleShapeY(gert::InferShapeContext *context, const gert::Tensor *groupListTensor,
                                            size_t weightDimN, bool isXTransposed, size_t xDimM)
{
    auto groupListData = groupListTensor->GetData<int64_t>();
    OP_CHECK_IF(groupListData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList",
                                                         "Failed to obtain necessary data from groupListTensor"),
                return GRAPH_FAILED);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTypePtr);
    const gert::Shape *x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Shape);
    const gert::Shape *weight0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight0Shape);
    // For SPARSEM (groupListType=2), groupList shape is [E, 2], so loop count should be E (first dim)
    // For CUMSUM/COUNT (groupListType=0/1), groupList is 1D, loop count is shape size
    int64_t loopCount = (*groupListTypePtr == GROUP_LIST_SPARSE) ? groupListTensor->GetStorageShape().GetDim(0) :
                                                                   groupListTensor->GetShapeSize();
    int64_t preOffset = 0;
    for (int idx = 0; idx < loopCount; ++idx) {
        const gert::Shape *weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idx);
        if (weightShape == nullptr) {
            weightShape = weight0Shape;
        }
        if (isXTransposed) {
            const gert::Shape *xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
            if (xShape == nullptr) {
                xShape = x0Shape;
            }
            std::vector<int64_t> yDims = {xShape->GetDim(xDimM), weightShape->GetDim(weightDimN)};
            OP_CHECK_IF(
                UpdateShapeY(context, GMM_INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "y",
                    Ops::Transformer::Gmm::FormatString("Failed to update shape of y[%d] when x is transposed", idx)
                        .c_str()),
                return GRAPH_FAILED);
        } else {
            std::vector<int64_t> yDims;
            if (*groupListTypePtr == 0) {
                yDims = {groupListData[idx] - preOffset, weightShape->GetDim(weightDimN)};
                preOffset = groupListData[idx];
            } else if (*groupListTypePtr == 1) {
                yDims = {groupListData[idx], weightShape->GetDim(weightDimN)};
            } else if (*groupListTypePtr == GROUP_LIST_SPARSE) {
                // SPARSEM: groupList shape is [E, 2], second column (idx*2+1) is token count per group
                yDims = {groupListData[idx * GROUP_LIST_SPARSE + GROUP_LIST_SPARSE_OFFSET],
                         weightShape->GetDim(weightDimN)};
            }
            OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            context->GetNodeName(), "y",
                            Ops::Transformer::Gmm::FormatString(
                                "Failed to update shape of y[%d] with groupListType %ld", idx, *groupListTypePtr)
                                .c_str()),
                        return GRAPH_FAILED);
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus MultiInMultiOutWithoutGroupList(gert::InferShapeContext *context)
{
    size_t idx = 0;
    size_t idw = 0;
    const gert::Shape *w0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
    while (true) {
        const gert::Shape *xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
        if (xShape == nullptr) {
            break;
        }
        ++idx;
        const gert::Shape *wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idw);
        if (wShape) {
            ++idw;
        } else {
            wShape = w0Shape;
        }
        OP_CHECK_IF(UpdateShapeYMultiDim(context, GMM_INDEX_OUT_Y + idx - 1, xShape, wShape) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "y",
                        Ops::Transformer::Gmm::FormatString(
                            "Failed to update shape of y[%zu] in multi-in-multi-out case without groupList", idx - 1)
                            .c_str()),
                    return GRAPH_FAILED);
    }
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_TYPE);
    bool success = true;
    if (w0Shape->GetDimNum() == 2) { // 2 two-dim weight tensor
        if (groupTypePtr != nullptr && *groupTypePtr == 2) {
            success = true;
        } else {
            success = idx == idw;
        }
    } else {
        success = static_cast<int64_t>(idx) == w0Shape->GetDim(0);
    }
    OP_CHECK_IF(!success,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "weight",
                    Ops::Transformer::Gmm::FormatString(
                        "x tensorList's length[%zu] != weight tensor's first dim[%ld] and length[%zu]", idx,
                        w0Shape->GetDim(0), idw)
                        .c_str()),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus MultiWeightMultiOutWithoutGroupList(gert::InferShapeContext *context)
{
    size_t idx = 0;
    const gert::Shape *x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Shape);
    while (true) {
        const gert::Shape *wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idx);
        if (!wShape) {
            break;
        }
        ++idx;
        OP_CHECK_IF(
            UpdateShapeYMultiDim(context, GMM_INDEX_OUT_Y + idx - 1, x0Shape, wShape) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "y",
                Ops::Transformer::Gmm::FormatString(
                    "Failed to update shape of y[%zu] in multi-weight-multi-out case without groupList", idx - 1)
                    .c_str()),
            return GRAPH_FAILED);
    }

    return GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus GetAttrsValue(T context, GMMAttrs &gmmAttrs)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_SPLIT_ITEM);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitItemPtr);
    gmmAttrs.splitItem = *splitItemPtr;
    OP_LOGI(context->GetNodeName(), "Attr splitItem = %ld", gmmAttrs.splitItem);

    const int64_t *dtypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
    gmmAttrs.outputDtype = *dtypePtr;
    OP_LOGI(context->GetNodeName(), "Attr dtype = %ld", gmmAttrs.outputDtype);

    const auto tuningConfigPtr =
        attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(GMM_INDEX_ATTR_TUNING_CONFIG);
    gmmAttrs.tuningConfig =
        (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0) ? tuningConfigPtr->GetData()[0] : 0;
    OP_LOGI(context->GetNodeName(), "Attr tuningConfig = %ld", gmmAttrs.tuningConfig);

    const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupTypePtr);
    gmmAttrs.groupType = *groupTypePtr;
    OP_LOGI(context->GetNodeName(), "Attr groupType = %ld", gmmAttrs.groupType);

    const bool *transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeWPtr);
    gmmAttrs.transposeWeight = *transposeWPtr;
    OP_LOGI(context->GetNodeName(), "Attr isWeightTransposed = %d", gmmAttrs.transposeWeight);

    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeXPtr);
    gmmAttrs.transposeX = *transposeXPtr;
    OP_LOGI(context->GetNodeName(), "Attr isXTransposed = %d", gmmAttrs.transposeX);

    const int64_t *activeType = attrs->GetInt(GMM_INDEX_ATTR_ACT_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeType);
    gmmAttrs.activeType = *activeType;
    OP_LOGI(context->GetNodeName(), "Attr activeType = %ld", gmmAttrs.activeType);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(gert::InferShapeContext *context, GMMAttrs &gmmAttrs)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(
        CheckSplitItem(gmmAttrs.splitItem) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "splitItem", std::to_string(gmmAttrs.splitItem),
                                              "Invalid splitItem, which can only be one of 0/1/2/3"),
        return GRAPH_FAILED);
    OP_CHECK_IF(CheckGroupType(context, gmmAttrs.groupType) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groupType",
                                                      std::to_string(gmmAttrs.groupType), "Invalid groupType"),
                return GRAPH_FAILED);
    const int64_t *activeType = attrs->GetInt(GMM_INDEX_ATTR_ACT_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeType);
    OP_CHECK_IF(
        *activeType < 0 || *activeType >= static_cast<int64_t>(GMMActType::END_ACT_TYPE_ENUM),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "activeType", std::to_string(gmmAttrs.activeType),
                                              "activeType must be no less than 0 and smaller than 6"),
        return GRAPH_FAILED);
    OP_CHECK_IF(*activeType == static_cast<int64_t>(GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "activeType", std::to_string(*activeType),
                                                      "Activation function does not support GELU_ERR_FUNC now"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetNumOfInputs(const gert::InferShapeContext *context, size_t &numX, size_t &numWeight,
                                      int64_t &lenGroupList)
{
    ge::graphStatus res = GRAPH_SUCCESS;
    const gert::Shape *shape = nullptr;
    while (true) {
        shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, numX);
        if (shape == nullptr) { // last shape
            break;
        }
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            if (shape->GetDim(i) < 0) { // shape dim cannot be smaller than 0
                res = GRAPH_FAILED;
                break;
            }
        }
        ++numX;
    }
    OP_LOGI(context->GetNodeName(), "numX = %lu", numX);

    while (true) {
        shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, numWeight);
        if (shape == nullptr) { // last shape
            break;
        }
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            if (shape->GetDim(i) < 0) { // shape dim cannot be smaller than 0
                res = GRAPH_FAILED;
                break;
            }
        }
        ++numWeight;
    }
    OP_LOGI(context->GetNodeName(), "numWeight = %lu", numWeight);

    const gert::Tensor *groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
    if (groupListTensor != nullptr) {
        lenGroupList = groupListTensor->GetStorageShape().GetDim(0); // groupListType 2 shape is [e, 2]
        if (lenGroupList < 0) {                                      // lenGroupList cannot be smaller than 0
            res = GRAPH_FAILED;
        }
    }
    OP_LOGI(context->GetNodeName(), "lenGroupList = %ld", lenGroupList);

    return res;
}

static int64_t GetDim0(const gert::InferShapeContext *context, bool isXTransposed, size_t numX, size_t xDimM)
{
    int64_t dim0 = 0;
    if (isXTransposed) {
        const gert::Shape *x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
        dim0 = (x0Shape == nullptr ? 0 : x0Shape->GetDim(xDimM));
    } else {
        for (size_t idx = 0; idx < numX; ++idx) {
            const gert::Shape *xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
            int64_t tmpDim0 = (xShape == nullptr ? 0 : xShape->GetDim(0));
            if (tmpDim0 >= 0) {
                dim0 += tmpDim0;
            } else {
                return tmpDim0;
            }
        }
    }

    return dim0;
}

static bool inline IsNonEmpty(const gert::Shape *shape)
{
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

struct GMMWeightAxisInfo {
    size_t k;
    size_t n;
};

static bool IsS8S4PseudoQuant(const gert::InferShapeContext *context)
{
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    auto scaleDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_SCALE, 0);
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    return xDesc != nullptr && weightDesc != nullptr && scaleDesc != nullptr && IsNonEmpty(scaleShape) &&
           xDesc->GetDataType() == DT_INT8 && weightDesc->GetDataType() == DT_INT4 &&
           scaleDesc->GetDataType() == DT_UINT64;
}

static bool IsS8S4SpecialWeightFormat(const gert::InferShapeContext *context)
{
    if (!IsS8S4PseudoQuant(context)) {
        return false;
    }
    const auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const auto tuningConfigPtr =
        attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(GMM_INDEX_ATTR_TUNING_CONFIG);
    if (tuningConfigPtr == nullptr || tuningConfigPtr->GetSize() <= 1) {
        return false;
    }
    const auto tuningConfig = tuningConfigPtr->GetData();
    return tuningConfig[1] == 1;
}

static GMMWeightAxisInfo GetWeightAxisInfo(const gert::InferShapeContext *context, size_t weightDimNum,
                                           bool transposeWeight)
{
    const bool logicalTransposeWeight = transposeWeight || IsS8S4SpecialWeightFormat(context);
    return {weightDimNum - (logicalTransposeWeight ? 1UL : 2UL), weightDimNum - (logicalTransposeWeight ? 2UL : 1UL)};
}

static ge::graphStatus IsGmmAntiQuantEmpty(gert::InferShapeContext *context)
{
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_SCALE),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "antiquantScale",
                                                         "antiquantScale is not null or empty"),
                return GRAPH_FAILED);
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_OFFSET),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "antiquantOffset",
                                                         "antiquantOffset is not null or empty"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus IsGmmQuantEmpty(gert::InferShapeContext *context)
{
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_SCALE),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale", "scale is not null or empty"),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        !IsTensorListNullOrEmpty(context, GMM_INDEX_IN_OFFSET),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "offset", "offset is not null or empty"),
        return GRAPH_FAILED);
    const gert::Shape *pertokenQuantScale0Shape = context->GetOptionalInputShape(GMM_INDEX_IN_PERTOKEN_SCALE);
    OP_CHECK_IF(IsNonEmpty(pertokenQuantScale0Shape),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "perTokenScale",
                                                         "pertokenQuant scale is not null or empty"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckNonQuant(gert::InferShapeContext *context)
{
    OP_CHECK_IF(IsGmmQuantEmpty(context) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                         "Detected nonquant, but quant inputs is not empty"),
                return GRAPH_FAILED);
    OP_CHECK_IF(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                         "Detected nonquant, but antiquant inputs is not empty"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetGroupSize(const gert::InferShapeContext *context, GMMParamsInfo &paramsInfo)
{
    size_t groupNum = 1;
    // all case allows 1024 in infershape, specific check in tiling
    size_t maxGroupNum = GMM_MAX_GROUP_LIST_SIZE_TENSOR;
    if (paramsInfo.numX > 1UL) {
        groupNum = paramsInfo.numX;
    } else if (paramsInfo.numWeight > 1UL) {
        groupNum = paramsInfo.numWeight;
    } else if (paramsInfo.numY > 1UL) {
        groupNum = paramsInfo.numY;
    } else if (paramsInfo.lenGroupList > 0) {
        groupNum = static_cast<size_t>(paramsInfo.lenGroupList);
    }
    OP_CHECK_IF(
        groupNum > maxGroupNum,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "groupNum", std::to_string(groupNum),
            Ops::Transformer::Gmm::FormatString("groupNum must be less than or equal to %zu", maxGroupNum).c_str()),
        return GRAPH_FAILED);
    paramsInfo.groupNum = groupNum;
    return GRAPH_SUCCESS;
}

static graphStatus CheckDimNumAndPerGroupNum(const gert::InferShapeContext *context, bool isAntiquantInt4,
                                             const std::tuple<size_t, size_t, int64_t> &dimData,
                                             const gert::Shape *tensorShape, const std::string &tensorType)
{
    size_t tensorDimNum = std::get<0>(dimData);
    size_t expectedDimNum = std::get<1>(dimData);   // 1: the sceond element
    int64_t weightKDimValue = std::get<2>(dimData); // 2: the third element
    if (isAntiquantInt4) {
        if (tensorDimNum == expectedDimNum) {
            int64_t perGroupNum = tensorShape->GetDim(tensorDimNum - 2); // 2: the last 2-th index
            OP_CHECK_IF(!(perGroupNum > 0 && weightKDimValue % perGroupNum == 0),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "perGroupNum", std::to_string(perGroupNum),
                            Ops::Transformer::Gmm::FormatString("perGroupNum must be greater than 0 and divide K[%ld]",
                                                                weightKDimValue)
                                .c_str()),
                        return GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                tensorDimNum != expectedDimNum - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), tensorType.c_str(), std::to_string(tensorDimNum),
                    Ops::Transformer::Gmm::FormatString("%s rank must be %zu (perchannel) or %zu (pergroup) in A16W4",
                                                        tensorType.c_str(), expectedDimNum - 1, expectedDimNum)
                        .c_str()),
                return GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(
            tensorDimNum != expectedDimNum - 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), tensorType.c_str(), std::to_string(tensorDimNum),
                Ops::Transformer::Gmm::FormatString("%s rank must be %zu", tensorType.c_str(), expectedDimNum - 1)
                    .c_str()),
            return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalTensorList(gert::InferShapeContext *context, const std::string tensorType,
                                               const GMMParamsInfo &paramsInfo, const GMMAttrs &gmmAttrs,
                                               size_t nodeIdx)
{
    // check bias，scale, antiquant scale or antiquant offset's size，tensor dimension and shape.
    const size_t &groupNum = paramsInfo.groupNum;
    size_t tensorSize = 0;
    while (context->GetDynamicInputShape(nodeIdx, tensorSize) != nullptr) {
        ++tensorSize;
    }
    uint64_t weightGroupedSize = static_cast<uint64_t>(paramsInfo.numWeight);
    const int64_t &groupType = gmmAttrs.groupType;
    auto shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    const auto weightAxis = GetWeightAxisInfo(context, shape->GetDimNum(), gmmAttrs.transposeWeight);
    uint64_t weightNDimIdx = weightAxis.n;
    auto tensor0Shape = context->GetDynamicInputShape(nodeIdx, 0);
    // tensorList size should equals with weight's size
    OP_CHECK_IF(tensorSize != weightGroupedSize,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), tensorType + ", weight",
                    Ops::Transformer::Gmm::FormatString("%zu, %zu", tensorSize, weightGroupedSize).c_str(),
                    tensorType + " tensor count must equal weight tensor count."),
                return GRAPH_FAILED);
    bool isSingleWeight = (weightGroupedSize == 1 && groupType != GMM_NO_SPLIT);
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    bool isAntiquantInt4 = (w0Desc->GetDataType() == DT_INT4 && tensorType.find("antiquant") != std::string::npos);
    if (isSingleWeight) { // In this case, nodeIdx should have only single tensor, its dim should be 2.
        OP_CHECK_IF(IsTensorListNullOrEmpty(context, nodeIdx),
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), tensorType.c_str()), return GRAPH_FAILED);
        size_t tensorDimNum = tensor0Shape->GetDimNum();
        int64_t k = shape->GetDim(weightAxis.k);
        // 3: shape is (E,G,N),G is the perGroupNum
        OP_CHECK_IF(CheckDimNumAndPerGroupNum(context, isAntiquantInt4, {tensorDimNum, 3, k}, tensor0Shape,
                                              tensorType) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                             "CheckDimNumAndPerGroupNum failed"),
                    return GRAPH_FAILED);
        OP_CHECK_IF(static_cast<size_t>(tensor0Shape->GetDim(0)) != groupNum,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), tensorType.c_str(), std::to_string(tensor0Shape->GetDim(0)),
                        Ops::Transformer::Gmm::FormatString("%s batch size should equal group count %zu",
                                                            tensorType.c_str(), groupNum)
                            .c_str()),
                    return GRAPH_FAILED);
        // tensor's N axis size should equal with weight's N axis.
        int64_t weightNDimValue = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0)->GetDim(weightNDimIdx);
        int64_t tensorNDimValue = tensor0Shape->GetDim(tensorDimNum - 1);
        OP_CHECK_IF(
            tensorNDimValue != weightNDimValue,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), tensorType.c_str(), std::to_string(tensorNDimValue),
                Ops::Transformer::Gmm::FormatString("N dimension should equal weight N dimension %ld", weightNDimValue)
                    .c_str()),
            return GRAPH_FAILED);
    } else {
        for (uint64_t i = 0; i < groupNum; i++) {
            auto tensorShape = context->GetDynamicInputShape(nodeIdx, i);
            OP_CHECK_IF(tensorShape == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(
                            context->GetNodeName(),
                            Ops::Transformer::Gmm::FormatString("%s[%lu]", tensorType.c_str(), i).c_str()),
                        return GRAPH_FAILED);
            // check each of tensor's dim to be 1
            size_t tensorDimNum = tensorShape->GetDimNum();
            auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
            int64_t k = wShape->GetDim(wShape->GetDimNum() - (gmmAttrs.transposeWeight ? 1 : 2)); // 2: axis index
            // 2: shape is (G,N), G is the perGroupNum
            OP_CHECK_IF(CheckDimNumAndPerGroupNum(context, isAntiquantInt4, {tensorDimNum, 2, k}, tensorShape,
                                                  tensorType) != GRAPH_SUCCESS,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                                 "CheckDimNumAndPerGroupNum failed"),
                        return GRAPH_FAILED);
            int64_t weightNDimValue = wShape->GetDim(weightNDimIdx);
            int64_t tensorNDimValue = tensorShape->GetDim(tensorDimNum - 1);
            OP_CHECK_IF(
                tensorNDimValue != weightNDimValue,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), tensorType.c_str(), std::to_string(tensorNDimValue),
                    Ops::Transformer::Gmm::FormatString("N dimension of %s[%lu] should equal weight N dimension %ld",
                                                        tensorType.c_str(), i, weightNDimValue)
                        .c_str()),
                return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckPerTokenScale(const gert::InferShapeContext *context, const GMMParamsInfo &paramsInfo)
{
    // check pertoken scale's size, tensor dimension and shape
    const size_t &xGroupedSize = paramsInfo.numX;
    const size_t &weightGroupedSize = paramsInfo.numWeight;
    const size_t &yGroupedSize = paramsInfo.numY;
    uint64_t xMDimIdx = 0;
    // check pertoken scale's size to be equal with x's
    if ((xGroupedSize == 1UL) && (yGroupedSize == 1UL)) {
        auto perTokenScale0Shape = context->GetOptionalInputShape(GMM_INDEX_IN_PERTOKEN_SCALE);
        OP_CHECK_IF(perTokenScale0Shape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "perTokenScaleOptional"), return GRAPH_FAILED);
        // tensor dimension of pertoken_scale should be 1.
        size_t tensorDimNum = perTokenScale0Shape->GetDimNum();
        OP_CHECK_IF(tensorDimNum != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "perTokenScaleOptional", std::to_string(tensorDimNum),
                        "perTokenScaleOptional rank must be 1 when x is a single tensor"),
                    return GRAPH_FAILED);
        // check pertoken_scale's tensor shape size to be equal with M axis size of x.
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
        int64_t xMDimValue = xShape->GetDim(xMDimIdx);
        int64_t tensorMDimValue = perTokenScale0Shape->GetDim(tensorDimNum - 1);
        OP_CHECK_IF(tensorMDimValue != xMDimValue,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "perTokenScaleOptional", std::to_string(tensorMDimValue),
                        Ops::Transformer::Gmm::FormatString("perTokenScale M dimension should equal x M dimension %ld",
                                                            xMDimValue)
                            .c_str()),
                    return GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            context->GetNodeName(), "x, weight, y",
            Ops::Transformer::Gmm::FormatString("%zu, %zu, %zu", xGroupedSize, weightGroupedSize, yGroupedSize).c_str(),
            "Per-token quantization requires x, weight, and y to each be single tensors");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupedMatmulQuant(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                               const GMMParamsInfo &paramsInfo)
{
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "Ascend310P",
                                                      "quant cases are not supported on Ascend310P"),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        gmmAttrs.groupType == GMM_SPLIT_K,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groupType", std::to_string(gmmAttrs.groupType),
                                              "quant cases do not support split along the K axis"),
        return GRAPH_FAILED);
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_OFFSET),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "offset",
                                                         "offset must be nullptr in quant, but now is not nullptr"),
                return GRAPH_FAILED);
    if (gmmAttrs.outputDtype != GMM_OUT_DTYPE_INT32) { // output dtype is int32, this scene does not need scale
        OP_CHECK_IF(IsTensorListNullOrEmpty(context, GMM_INDEX_IN_SCALE),
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "scale"), return GRAPH_FAILED);
        OP_CHECK_IF(
            CheckOptionalTensorList(context, "scale", paramsInfo, gmmAttrs, GMM_INDEX_IN_SCALE) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale", "Invalid scale"),
            return GRAPH_FAILED);
    }
    bool isPerTokenQuant = context->GetOptionalInputShape(GMM_INDEX_IN_PERTOKEN_SCALE) != nullptr;
    if (isPerTokenQuant) {
        OP_CHECK_IF(CheckPerTokenScale(context, paramsInfo) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "perTokenScale",
                                                             "Check perTokenScale failed"),
                    return GRAPH_FAILED);
    }
    OP_CHECK_IF(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                         "Detected quant, but antiquant inputs is not empty"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
static bool isA8W4AsymmetricQuant(const gert::InferShapeContext *context)
{
    auto offsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_OFFSET, 0);
    if (offsetShape == nullptr) {
        return false;
    }
    size_t offsetDimNum = offsetShape->GetDimNum();
    if (offsetDimNum != GMM_A8W4_OFFSET_DIM_NUM) {
        return false;
    }
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    if (weightShape == nullptr) {
        return false;
    }
    if (weightShape->GetDimNum() < GMM_MIN_WEIGHT_DIM) {
        return false;
    }
    const auto weightAxis = GetWeightAxisInfo(context, weightShape->GetDimNum(), false);
    if (offsetShape->GetDim(0) == weightShape->GetDim(0) && offsetShape->GetDim(1) == 1 &&
        offsetShape->GetDim(GMM_A8W4_OFFSET_DIM_NUM - 1) == weightShape->GetDim(weightAxis.n)) {
        return true;
    }
    return false;
}
static ge::graphStatus CheckA8W4AsymQuantParams(gert::InferShapeContext *context, const GMMParamsInfo &paramsInfo)
{
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "Ascend310P",
                                                      "quant cases are not supported on Ascend310P"),
                return GRAPH_FAILED);
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    size_t biasDimNum = biasShape->GetDimNum();
    size_t scaleDimNum = scaleShape->GetDimNum();
    int64_t e = weightShape->GetDim(0);
    const auto weightAxis = GetWeightAxisInfo(context, weightShape->GetDimNum(), false);
    int64_t n = weightShape->GetDim(weightAxis.n);
    OP_CHECK_IF(
        IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "antiquant inputs is not empty"),
        return GRAPH_FAILED);
    OP_CHECK_IF(biasDimNum != GMM_A8W4_BIAS_DIM_NUM || biasShape->GetDim(0) != e || biasShape->GetDim(1) != n,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "bias", Ops::Base::ToString(*biasShape).c_str(),
                    Ops::Transformer::Gmm::FormatString("bias shape should be (e,n). e=%ld, n=%ld", e, n).c_str()),
                return GRAPH_FAILED);
    auto isScaleInvalid = !(scaleDimNum == GMM_A8W4_OFFSET_DIM_NUM && scaleShape->GetDim(0) == e &&
                            scaleShape->GetDim(1) == 1 && scaleShape->GetDim(GMM_A8W4_OFFSET_DIM_NUM - 1) == n);
    OP_CHECK_IF(isScaleInvalid,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "scale", Ops::Base::ToString(*scaleShape).c_str(),
                    Ops::Transformer::Gmm::FormatString("scale shape should be (e,1,n). e=%ld, n=%ld", e, n).c_str()),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static int64_t GetPergroupSize(const GMMAttrs &gmmAttrs, bool isSingleWeight, const gert::Shape *wShape,
                               const gert::Shape *shape)
{
    int64_t pergroupSize = 0;
    size_t shapeDimNum = shape->GetDimNum();
    if (isSingleWeight) { // antiquant param shape (E, N), (E, G, N)
        if (shapeDimNum > GMM_SEPARATED_WEIGHT_DIM) {
            int64_t k = gmmAttrs.transposeWeight ? wShape->GetDim(2) : wShape->GetDim(1); // 2: the k axis index
            pergroupSize = k / shape->GetDim(shapeDimNum - 2);                            // 2: the last 2-th index
        }
    } else { //  antiquant param shape (N), (G, N)
        if (shapeDimNum > 1UL) {
            int64_t k = gmmAttrs.transposeWeight ? wShape->GetDim(1) : wShape->GetDim(0);
            pergroupSize = k / shape->GetDim(shapeDimNum - 2); // 2: the last 2-th index
        }
    }
    return pergroupSize;
}

static ge::graphStatus CheckGroupedMatmulAntiQuantGroupSize(const gert::InferShapeContext *context,
                                                            const GMMAttrs &gmmAttrs, const GMMParamsInfo &paramsInfo,
                                                            bool hasAntiquantOffset)
{
    auto antiquantScale0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto dimNum = antiquantScale0Shape->GetDimNum();
    bool isSingleWeight = ((paramsInfo.numWeight == 1UL) && (gmmAttrs.groupType != GMM_NO_SPLIT));
    int64_t pergroupSize = GetPergroupSize(gmmAttrs, isSingleWeight,
                                           context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0), antiquantScale0Shape);
    OP_CHECK_IF(gmmAttrs.transposeWeight && pergroupSize % 2 != 0, // 2: a factor
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "pergroupSize", std::to_string(pergroupSize),
                    "pergroupSize should be even when weight is transposed in A16W4-pergroup case"),
                return GRAPH_FAILED);
    for (size_t i = 0;; ++i) {
        auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, i);
        if (antiquantScaleShape == nullptr) {
            break;
        }
        size_t antiquantScaleDimNum = antiquantScaleShape->GetDimNum();
        OP_CHECK_IF(
            antiquantScaleDimNum != dimNum,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "antiquantScale", std::to_string(antiquantScaleDimNum),
                Ops::Transformer::Gmm::FormatString("antiquantScale[%zu] rank should equal %zu", i, dimNum).c_str()),
            return GRAPH_FAILED);
        auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
        int64_t pergroupSizeOfScale = GetPergroupSize(gmmAttrs, isSingleWeight, wShape, antiquantScaleShape);
        OP_CHECK_IF(
            pergroupSizeOfScale != pergroupSize,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "antiquantScale", std::to_string(pergroupSizeOfScale),
                Ops::Transformer::Gmm::FormatString("antiquantScale[%zu] pergroup size should be %ld", i, pergroupSize)
                    .c_str()),
            return GRAPH_FAILED);
        if (hasAntiquantOffset) {
            auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, i);
            size_t antiquantOffsetDimNum = antiquantOffsetShape->GetDimNum();
            OP_CHECK_IF(antiquantOffsetDimNum != dimNum,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            context->GetNodeName(), "antiquantOffset", std::to_string(antiquantOffsetDimNum),
                            Ops::Transformer::Gmm::FormatString("antiquantOffset[%zu] rank should equal %zu", i, dimNum)
                                .c_str()),
                        return GRAPH_FAILED);
            int64_t pergroupSizeOfOffset = GetPergroupSize(gmmAttrs, isSingleWeight, wShape, antiquantOffsetShape);
            OP_CHECK_IF(pergroupSizeOfOffset != pergroupSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "antiquantOffset", std::to_string(pergroupSizeOfOffset),
                            Ops::Transformer::Gmm::FormatString("antiquantOffset[%zu] pergroup size should be %ld", i,
                                                                pergroupSize)
                                .c_str()),
                        return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupedMatmulAntiQuantForShape(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                                           const GMMParamsInfo &paramsInfo)
{
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platform", "Ascend310P",
                                                      "antiquant cases are not supported on Ascend310P"),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        gmmAttrs.groupType == GMM_SPLIT_K,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groupType", std::to_string(gmmAttrs.groupType),
                                              "antiquant cases do not support split along the K axis"),
        return GRAPH_FAILED);
    OP_CHECK_IF(IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_SCALE),
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "antiquantScale"), return GRAPH_FAILED);
    // check antiquantScale and antiquantOffset's tensor shape
    OP_CHECK_IF(
        CheckOptionalTensorList(context, "antiquantScale", paramsInfo, gmmAttrs, GMM_INDEX_IN_ANTIQUANT_SCALE) !=
            GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "antiquantScale", "Invalid antiquantScale"),
        return GRAPH_FAILED);
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    bool hasAntiquantOffset = !IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_OFFSET);
    OP_CHECK_IF(w0Desc->GetDataType() != DT_INT4 && !hasAntiquantOffset,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "antiquantOffset"), return GRAPH_FAILED);
    if (hasAntiquantOffset) {
        OP_CHECK_IF(CheckOptionalTensorList(context, "antiquantOffset", paramsInfo, gmmAttrs,
                                            GMM_INDEX_IN_ANTIQUANT_OFFSET) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "antiquantOffset",
                                                             "Invalid antiquantOffset"),
                    return GRAPH_FAILED);
    }
    // check perGroupSize
    if (w0Desc->GetDataType() == DT_INT4) {
        OP_CHECK_IF(
            CheckGroupedMatmulAntiQuantGroupSize(context, gmmAttrs, paramsInfo, hasAntiquantOffset) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "Invalid antiquant group size"),
            return GRAPH_FAILED);
    }
    OP_CHECK_IF(IsGmmQuantEmpty(context) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                         "Detected antiquant, but quant inputs is not empty"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckQuantParams(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                        GMMParamsInfo &paramsInfo)
{
    auto x0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Desc);
    DataType xDtype = x0Desc->GetDataType();
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    DataType weightDtype = w0Desc->GetDataType();
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {
        if (!isA8W4AsymmetricQuant(context)) {
            return GRAPH_SUCCESS;
        }
        return CheckA8W4AsymQuantParams(context, paramsInfo);
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16 || xDtype == DataType::DT_FLOAT) &&
        xDtype == weightDtype) {
        // nonquant
        return CheckNonQuant(context);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        // quant
        return CheckGroupedMatmulQuant(context, gmmAttrs, paramsInfo);
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16) &&
        (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
        // antiquant
        return CheckGroupedMatmulAntiQuantForShape(context, gmmAttrs, paramsInfo);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus CheckFunctionParamsForShape(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                                   GMMParamsInfo &paramsInfo)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        paramsInfo.platform = PlatformID::UNKNOWN;
        OP_LOGW(context->GetNodeName(), "Cannot get platform info!");
        return GRAPH_SUCCESS;
    } else {
        paramsInfo.platform = (optionalInfo.soc_version.find("310P") != std::string::npos) ? PlatformID::ASCEND310P :
                              (optionalInfo.soc_version.find("950") != std::string::npos)  ? PlatformID::ASCEND950 :
                                                                                             PlatformID::ASCEND910B;
    }
    OP_CHECK_IF(CheckQuantParams(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "CheckQuantParams failed"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
static ge::graphStatus CheckDimNumAndGroupListNoSplitAndFormat(const gert::InferShapeContext *context,
                                                               uint64_t tensorListLength, const size_t numWeight)
{
    // when groupList is not empty, check its size equal with the length of x.
    auto groupTensorOptionalShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    if (groupTensorOptionalShape != nullptr) {
        OP_CHECK_IF(groupTensorOptionalShape->GetDim(0) != static_cast<int64_t>(tensorListLength),
                    OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "groupList",
                                                 std::to_string(groupTensorOptionalShape->GetDim(0)),
                                                 std::to_string(tensorListLength)),
                    return GRAPH_FAILED);
    }
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    // check dimension
    for (size_t i = 0; i < tensorListLength; ++i) {
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        OP_CHECK_IF(xShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(),
                                               Ops::Transformer::Gmm::FormatString("x[%zu]", i).c_str()),
                    return GRAPH_FAILED);
        if (numWeight > 1) {
            wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
            size_t weightDimNum = wShape->GetDimNum();
            OP_CHECK_IF(
                weightDimNum != GMM_SEPARATED_WEIGHT_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "weight", std::to_string(weightDimNum),
                    Ops::Transformer::Gmm::FormatString("weight[%lu] rank should be 2 when weight is separated", i)
                        .c_str()),
                return GRAPH_FAILED);
        }
        size_t xDimNum = xShape->GetDimNum();
        OP_CHECK_IF(xDimNum > GMM_MAX_FM_DIM || xDimNum < GMM_MIN_FM_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(xDimNum),
                        Ops::Transformer::Gmm::FormatString("x[%lu] rank should be in [2, 6]", i).c_str()),
                    return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus TensorType2NodeId(const std::vector<std::string> &tensorType, std::vector<int64_t> &nodeIdx)
{
    if (nodeIdx.size() > tensorType.size()) {
        return GRAPH_FAILED;
    }
    for (size_t i(0); i < nodeIdx.size(); ++i) {
        if (tensorType[i] == "x") {
            nodeIdx[i] = GMM_INDEX_IN_X;
        } else if (tensorType[i] == "weight") {
            nodeIdx[i] = GMM_INDEX_IN_WEIGHT;
        } else if (tensorType[i] == "y") {
            nodeIdx[i] = GMM_INDEX_OUT_Y;
        } else {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckDimNum(gert::InferShapeContext *context, uint64_t tensorListLength,
                                   const size_t expectedDimNum, const std::string tensorType)
{
    int64_t nodeIdx = 0;
    if (tensorType == "x") {
        nodeIdx = static_cast<int64_t>(GMM_INDEX_IN_X);
    } else if (tensorType == "weight") {
        nodeIdx = static_cast<int64_t>(GMM_INDEX_IN_WEIGHT);
    } else if (tensorType == "y") {
        nodeIdx = static_cast<int64_t>(GMM_INDEX_OUT_Y);
    } else {
        return GRAPH_FAILED;
    }
    const gert::Shape *shape;
    for (size_t i = 0; i < tensorListLength; ++i) {
        if (tensorType == "y") {
            shape = context->GetOutputShape(nodeIdx + i);
        } else {
            shape = context->GetDynamicInputShape(nodeIdx, i);
        }
        OP_CHECK_IF(
            shape == nullptr,
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(),
                                       Ops::Transformer::Gmm::FormatString("%s[%zu]", tensorType.c_str(), i).c_str()),
            return GRAPH_FAILED);
        size_t dimNum = shape->GetDimNum();
        OP_CHECK_IF(
            dimNum != expectedDimNum,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), tensorType.c_str(), std::to_string(dimNum),
                Ops::Transformer::Gmm::FormatString("%s[%lu] rank should be %lu", tensorType.c_str(), i, expectedDimNum)
                    .c_str()),
            return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightShapeInnerAxisEven(const gert::InferShapeContext *context, const size_t weightSize,
                                                     const int64_t innerAxisDimId)
{
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    DataType wDtype = w0Desc->GetDataType();
    if (wDtype == DataType::DT_INT4) {
        for (size_t i = 0; i < weightSize; ++i) {
            auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
            int64_t n = wShape->GetDim(innerAxisDimId);
            OP_CHECK_IF(n % 2 != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "weight N", std::to_string(n),
                            Ops::Transformer::Gmm::FormatString(
                                "w[%zu] dim %ld value should be even when weight is int4 dtype", i, innerAxisDimId)
                                .c_str()),
                        return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus IsxSizeEqualWithWeightKAxis(const gert::InferShapeContext *context,
                                                   const GMMParamsInfo &paramsInfo, const gert::Shape *wShape,
                                                   size_t &wKDimIdx, size_t &wNDimIdx)
{
    if (paramsInfo.numWeight == 1 && wShape->GetDimNum() > 2) { // 2: separated tensor's dim
        wKDimIdx += 1UL;
        wNDimIdx += 1UL;
        OP_CHECK_IF(paramsInfo.numX != static_cast<size_t>(wShape->GetDim(0)),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "weight",
                        Ops::Transformer::Gmm::FormatString(
                            "When x and y are separated, and weight is not separated, size of x "
                            "%zu should equal to the first dim of weight tensor %ld",
                            paramsInfo.numX, wShape->GetDim(0))
                            .c_str()),
                    return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckCaseNoSplit(gert::InferShapeContext *context, bool transposeWeight,
                                        const GMMParamsInfo &paramsInfo)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    const size_t &xSize = paramsInfo.numX;
    const size_t &weightSize = paramsInfo.numWeight;
    // check group num
    OP_CHECK_IF(xSize != paramsInfo.numY,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x, y",
                    Ops::Transformer::Gmm::FormatString("%zu, %zu", xSize, paramsInfo.numY).c_str(),
                    "When y is separated, x and y tensor-list sizes should be equal"),
                return GRAPH_FAILED);
    OP_CHECK_IF(weightSize != 1 && xSize != weightSize,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x, weight",
                    Ops::Transformer::Gmm::FormatString("%zu, %zu", xSize, weightSize).c_str(),
                    "When x and weight are separated, tensor-list sizes should be equal"),
                return GRAPH_FAILED);
    // check dimension
    OP_CHECK_IF(
        CheckDimNumAndGroupListNoSplitAndFormat(context, xSize, weightSize) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList",
                                                 "Dim num or format of tensor in tensor lists or grouplist is invalid"),
        return GRAPH_FAILED);
    // check shape
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    size_t wKDimIdx = transposeWeight ? 1UL : 0UL;
    size_t wNDimIdx = transposeWeight ? 0UL : 1UL;
    OP_CHECK_IF(
        IsxSizeEqualWithWeightKAxis(context, paramsInfo, wShape, wKDimIdx, wNDimIdx) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "IsxSizeEqualWithWeightKAxis failed"),
        return GRAPH_FAILED);
    int64_t weightKDimValue = wShape->GetDim(wKDimIdx);
    int64_t weightNDimValue = wShape->GetDim(wNDimIdx);
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    DataType wDtype = w0Desc->GetDataType();
    // 2: an even factor
    OP_CHECK_IF(
        wDtype == DataType::DT_INT4 && weightNDimValue % 2 != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "weight N", std::to_string(weightNDimValue),
            Ops::Transformer::Gmm::FormatString("w[0] dim %lu value should be even when weight is int4 dtype", wNDimIdx)
                .c_str()),
        return GRAPH_FAILED);
    for (size_t i = 0; i < xSize; i++) {
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        size_t xDimNum = xShape->GetDimNum();
        // check inner axis of x, which should not be larger than 65535
        int64_t xKDimValue = xShape->GetDim(xDimNum - 1); // x always is not transposed
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(xKDimValue > GMM_MAX_INNER_AXIS,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "x K", std::to_string(xKDimValue),
                            Ops::Transformer::Gmm::FormatString("x[%lu] K should be less or equal to %ld", i,
                                                                GMM_MAX_INNER_AXIS)
                                .c_str()),
                        return GRAPH_FAILED);
        }
        if (weightSize > 1UL) {
            wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            weightKDimValue = wShape->GetDim(wKDimIdx);
            weightNDimValue = wShape->GetDim(wNDimIdx);
            // 2: an even factor
            OP_CHECK_IF(
                i > 0 && wDtype == DataType::DT_INT4 && weightNDimValue % 2 != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "weight N", std::to_string(weightNDimValue),
                    Ops::Transformer::Gmm::FormatString("w[%lu] N value should be even when weight is int4 dtype", i)
                        .c_str()),
                return GRAPH_FAILED);
        }
        OP_CHECK_IF(xKDimValue != weightKDimValue,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context->GetNodeName(), "x K, weight K",
                        Ops::Transformer::Gmm::FormatString("%ld, %ld", xKDimValue, weightKDimValue).c_str(),
                        Ops::Transformer::Gmm::FormatString("x[%lu] K should equal weight[%lu] K", i, i).c_str()),
                    return GRAPH_FAILED);
        // if weight is not transposed, check N aisx; otherwise, check K axis, which can be skiped
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(!transposeWeight && weightNDimValue > GMM_MAX_INNER_AXIS,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "weight N", std::to_string(weightNDimValue),
                            Ops::Transformer::Gmm::FormatString("w[%zu] N should be less or equal to %ld", i,
                                                                GMM_MAX_INNER_AXIS)
                                .c_str()),
                        return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckInnerAxisOfTensorList(const gert::InferShapeContext *context, size_t nodeId,
                                                  int64_t innerAxisDimId, size_t checkNum, const char *tensorType)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    for (size_t i = 0; i < checkNum; i++) {
        auto shape = context->GetDynamicInputShape(nodeId, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t innerAxisValue = shape->GetDim(innerAxisDimId);
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(
                innerAxisValue > GMM_MAX_INNER_AXIS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), Ops::Transformer::Gmm::FormatString("%s[%zu]", tensorType, i).c_str(),
                    std::to_string(innerAxisValue),
                    Ops::Transformer::Gmm::FormatString("dim %ld should be less or equal to %ld", innerAxisDimId,
                                                        static_cast<int64_t>(GMM_MAX_INNER_AXIS))
                        .c_str()),
                return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeSameLengthTensorList(gert::InferShapeContext *context,
                                                      const std::vector<size_t> &dimIds, const int64_t innerAxisDimId,
                                                      const std::vector<std::string> tensorType, uint64_t groupNum)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    std::vector<int64_t> nodeIdx = {0, 0};
    OP_CHECK_IF(TensorType2NodeId(tensorType, nodeIdx) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "TensorType2NodeId failed"),
                return GRAPH_FAILED);
    // check two tensorlist's size to be the same, and tensors to have consistant dimension.
    const gert::Shape *shape;
    for (uint64_t i = 0; i < groupNum; i++) {
        shape = context->GetDynamicInputShape(nodeIdx[0], i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t dimValue1 = shape->GetDim(dimIds[0]);
        // tensorType[2] indicates whether check tensorList0's inner axis(innerAxisDimId)
        if (tensorType[2] == "true" && innerAxisDimId > -1) {
            auto shape0 = context->GetDynamicInputShape(nodeIdx[0], i);
            OP_CHECK_NULL_WITH_CONTEXT(context, shape0);
            int64_t innerAxisValue = shape0->GetDim(innerAxisDimId);
            if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) &&
                innerAxisValue > GMM_MAX_INNER_AXIS) {
                OP_LOGW(context->GetNodeName(),
                        "Dim %lu value of %s[%lu] should be less than or equal to %ld,"
                        "but now is %ld.",
                        dimIds[0], tensorType[0].c_str(), i, GMM_MAX_INNER_AXIS, innerAxisValue);
            }
        }
        if (tensorType[1] == "y") {
            shape = context->GetOutputShape(nodeIdx[1] + i);
        } else {
            shape = context->GetDynamicInputShape(nodeIdx[1], i);
        }
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t dimValue2 = shape->GetDim(dimIds[1]);
        if (dimValue1 != dimValue2) {
            OP_LOGW(context->GetNodeName(),
                    "Dim %lu value of %s[%lu] should be equal to dim %lu value of %s[%lu],"
                    "but now is %ld and %ld respectively.",
                    dimIds[0], tensorType[0].c_str(), i, dimIds[1], tensorType[1].c_str(), i, dimValue1, dimValue2);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeDiffLengthTensorList(gert::InferShapeContext *context,
                                                      const std::vector<size_t> &dimIds, const int64_t innerAxisdimId,
                                                      const std::vector<std::string> tensorType, uint64_t groupNum)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    std::vector<int64_t> nodeIdx = {0, 0};
    OP_CHECK_IF(TensorType2NodeId(tensorType, nodeIdx) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "TensorType2NodeId failed"),
                return GRAPH_FAILED);
    // check each tensor's selected dimension size in a multi-tensor tensorlist's to equal with
    // the tensor selected dimension in single-tensor tensorlist.
    // the selected axis is not the split-axis.
    const gert::Shape *singleTensor0;
    if (tensorType[1] == "y") {
        singleTensor0 = context->GetOutputShape(nodeIdx[1]);
    } else {
        singleTensor0 = context->GetDynamicInputShape(nodeIdx[1], 0);
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, singleTensor0);
    int64_t dimValueSingle = singleTensor0->GetDim(dimIds[1]);
    // tensorType[2] indicates whether check single tensorList's inner axis(innerAxisDimId)
    if (tensorType[2] == "true" && innerAxisdimId > -1) {
        int64_t dimValue = singleTensor0->GetDim(innerAxisdimId);
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(dimValue > GMM_MAX_INNER_AXIS,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), tensorType[1].c_str(), std::to_string(dimValue),
                            Ops::Transformer::Gmm::FormatString("Dim %ld should be less or equal to %ld",
                                                                innerAxisdimId, GMM_MAX_INNER_AXIS)
                                .c_str()),
                        return GRAPH_FAILED);
        }
    }
    const gert::Shape *longTensor;
    for (uint64_t i = 0; i < groupNum; i++) {
        if (tensorType[0] == "y") {
            longTensor = context->GetOutputShape(nodeIdx[0] + i);
        } else {
            longTensor = context->GetDynamicInputShape(nodeIdx[0], i);
        }
        OP_CHECK_NULL_WITH_CONTEXT(context, longTensor);
        int64_t dimValueLong = longTensor->GetDim(dimIds[0]);
        OP_CHECK_IF(dimValueLong != dimValueSingle,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context->GetNodeName(), tensorType[0] + ", " + tensorType[1],
                        Ops::Transformer::Gmm::FormatString("%ld, %ld", dimValueLong, dimValueSingle).c_str(),
                        Ops::Transformer::Gmm::FormatString(
                            "Dimensions %lu of %s[%lu] and %lu of %s[0] should be equal", dimIds[0],
                            tensorType[0].c_str(), i, dimIds[1], tensorType[1].c_str())
                            .c_str()),
                    return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupListCommonTensor(const gert::InferShapeContext *context,
                                                  const bool isRequiredGroupList, const int64_t groupNum)
{
    auto groupTensorOptionalShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    bool isNull = groupTensorOptionalShape == nullptr;
    OP_CHECK_IF(isNull && isRequiredGroupList, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "groupList"),
                return GRAPH_FAILED);
    if (isNull) {
        return GRAPH_SUCCESS;
    }
    int64_t groupListSize = groupTensorOptionalShape->GetDim(0);
    OP_CHECK_IF(groupListSize > GMM_MAX_GROUP_LIST_SIZE_TENSOR,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "groupList length", std::to_string(groupListSize),
                    Ops::Transformer::Gmm::FormatString("groupList length must be less than or equal to %ld",
                                                        GMM_MAX_GROUP_LIST_SIZE_TENSOR)
                        .c_str()),
                return GRAPH_FAILED);
    OP_CHECK_IF(!((groupListSize == groupNum && groupNum > 1) || groupNum == 1),
                OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "groupList", std::to_string(groupListSize),
                                             std::to_string(groupNum)),
                return GRAPH_FAILED);
    auto groupListDesc = context->GetOptionalInputDesc(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListDesc);
    OP_CHECK_IF(groupListDesc->GetDataType() != DataType::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "groupList",
                    TypeUtils::DataTypeToAscendString(groupListDesc->GetDataType()).GetString(),
                    "Only int64 is supported for groupList"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSingleXSingleWeightSingleY(gert::InferShapeContext *context, bool transposeWeight,
                                                        const GMMParamsInfo &paramsInfo)
{
    std::vector<std::string> tenorXAndWeight{"x", "weight", "true"};
    // check dimension
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numX, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "Dim num or format of tensor in tensor list x is invalid"),
                return GRAPH_FAILED);
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numWeight, GMM_SPLIT_M_SINGLE_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "weight", "Dim num or format of tensor in tensor list weight is invalid"),
                return GRAPH_FAILED);
    // check shape, x(m,k), weight(b,k,n), y(m,n)
    int64_t innerAxisDimId = 1; // x always is not transposed, check K axis
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    const bool specialWeightFormat = IsS8S4SpecialWeightFormat(context);
    const auto weightAxis = GetWeightAxisInfo(context, weightShape->GetDimNum(), transposeWeight);
    size_t kAxisOfWeight = weightAxis.k;
    OP_CHECK_IF(CheckShapeSameLengthTensorList(context, {1, kAxisOfWeight}, innerAxisDimId, tenorXAndWeight,
                                               paramsInfo.numX) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "k dim value of x and weight is not matched"),
                return GRAPH_FAILED);
    innerAxisDimId = specialWeightFormat ?
                         static_cast<int64_t>(weightAxis.n) :
                         (!transposeWeight ? 2 : -1); // 非转置时 weight 内轴 N 位于第 3 维（索引 2），转置时置 -1 跳过
    OP_CHECK_IF(
        CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_WEIGHT, innerAxisDimId, paramsInfo.numWeight, "weight") !=
            GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "weight",
            Ops::Transformer::Gmm::FormatString("inner axis size of weight is larger than %ld", GMM_MAX_INNER_AXIS)
                .c_str()),
        return GRAPH_FAILED);
    const int64_t weightInnerAxis = specialWeightFormat ? static_cast<int64_t>(weightAxis.n) : 2;
    OP_CHECK_IF(CheckWeightShapeInnerAxisEven(context, paramsInfo.numWeight, weightInnerAxis) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "weight's N axis size should be even when it is int4 dtype"),
                return GRAPH_FAILED);
    // check groupList
    OP_CHECK_IF(CheckGroupListCommonTensor(
                    context, true, context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0)->GetDim(0)) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList", "Invalid groupList"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSingleXSeparatedWeightSingleY(gert::InferShapeContext *context, bool transposeWeight,
                                                           const GMMParamsInfo &paramsInfo)
{
    std::vector<std::string> tenorWeightAndX{"weight", "x", "true"};
    // check dimension
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numX, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "Dim num or format of tensor in tensor list x is invalid"),
                return GRAPH_FAILED);
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numWeight, GMM_SEPARATED_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "weight", "Dim num or format of tensor in tensor list weight is invalid"),
                return GRAPH_FAILED);
    // check shape, x(m,k), weight(k,n), y(m,n)
    int64_t innerAxisDimId = 1; // x always is not transposed, check K axis
    size_t kAxisOfWeight = transposeWeight ? 1UL : 0UL;
    OP_CHECK_IF(CheckShapeDiffLengthTensorList(context, {kAxisOfWeight, 1}, innerAxisDimId, tenorWeightAndX,
                                               paramsInfo.numWeight) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "k dim value of x and weight is not matched"),
                return GRAPH_FAILED);
    innerAxisDimId =
        !transposeWeight ? 1 : -1; // if w is not transposed, check N asix; otherwise, check k axis, which can be skiped
    OP_CHECK_IF(
        CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_WEIGHT, innerAxisDimId, 1, "weight") != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "weight",
            Ops::Transformer::Gmm::FormatString("inner axis size of weight is larger than %ld", GMM_MAX_INNER_AXIS)
                .c_str()),
        return GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightShapeInnerAxisEven(context, paramsInfo.numWeight, 1) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "weight's N axis size should be even when it is int4 dtype"),
                return GRAPH_FAILED);
    // check groupList
    OP_CHECK_IF(CheckGroupListCommonTensor(context, true, paramsInfo.numWeight) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList", "Invalid groupList"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSeparatedXSeparatedWeightSingleY(gert::InferShapeContext *context, bool transposeWeight,
                                                              const GMMParamsInfo &paramsInfo)
{
    const size_t &xSize = paramsInfo.numX;
    const size_t &weightSize = paramsInfo.numWeight;
    std::vector<std::string> tenorWeightAndX{"weight", "x", "true"};
    OP_CHECK_IF(
        xSize != weightSize,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "weight",
            Ops::Transformer::Gmm::FormatString(
                "When x and weight are separated, size of x %lu should equal to size of weight %lu", xSize, weightSize)
                .c_str()),
        return GRAPH_FAILED);
    // check dimension
    OP_CHECK_IF(CheckDimNum(context, xSize, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "Dim num or format of tensor in tensor list x is invalid"),
                return GRAPH_FAILED);
    OP_CHECK_IF(CheckDimNum(context, weightSize, GMM_SEPARATED_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "weight", "Dim num or format of tensor in tensor list weight is invalid"),
                return GRAPH_FAILED);
    // check shape, x(m,k), weight(k,n), y(m,n)
    int64_t innerAxisDimId = 1; // originalShape's inner axis of weight
    size_t kAxisOfWeight = transposeWeight ? 1UL : 0UL;
    OP_CHECK_IF(CheckShapeSameLengthTensorList(context, {kAxisOfWeight, 1}, innerAxisDimId, tenorWeightAndX,
                                               weightSize) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "k dim value of x and weight is not matched"),
                return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 1 : -1; // if w is not transposed, N asix has been checked, need to check x's
                                                // inner axis(K, when x is always not transposed)
    OP_CHECK_IF(
        CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_X, innerAxisDimId, 1, "x") != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "x",
            Ops::Transformer::Gmm::FormatString("inner axis size of x is larger than %ld", GMM_MAX_INNER_AXIS).c_str()),
        return GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightShapeInnerAxisEven(context, weightSize, 1) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "weight's N axis size should be even when it is int4 dtype"),
                return GRAPH_FAILED);
    // check groupList
    OP_CHECK_IF(CheckGroupListCommonTensor(context, false, xSize) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList", "Invalid groupList"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckCaseSplitM(gert::InferShapeContext *context, bool transposeWeight,
                                       const GMMParamsInfo &paramsInfo)
{
    const size_t &xSize = paramsInfo.numX;
    const size_t &weightSize = paramsInfo.numWeight;
    const size_t &ySize = paramsInfo.numY;
    if ((xSize == 1UL) && (weightSize == 1UL) && (ySize == 1UL)) {
        OP_CHECK_IF(SplitMSingleXSingleWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                             "Split m, single x, single weight, single y case failed"),
                    return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xSize == 1UL) && (weightSize > 1UL) && (ySize == 1UL)) {
        OP_CHECK_IF(weightSize != paramsInfo.groupNum,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "weight",
                        Ops::Transformer::Gmm::FormatString("weight Size [%zu] does not equal with groupNum %zu",
                                                            weightSize, paramsInfo.groupNum)
                            .c_str()),
                    return GRAPH_FAILED);
        OP_CHECK_IF(SplitMSingleXSeparatedWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "weight", "Split m, single x, separated weight, single y case failed"),
                    return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xSize == 1UL) && (weightSize > 1UL) && (ySize > 1UL)) {
        const gert::Tensor *groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
        OP_CHECK_IF(groupListTensor == nullptr || groupListTensor->GetData<int64_t>() == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList",
                                                             "Failed to obtain necessary data from groupListTensor. "
                                                             "When grouplist is an invalid tensor, split m, single x, "
                                                             "separated weight, separated y cases do not support"),
                    return GRAPH_FAILED);
        return GRAPH_SUCCESS; // skip the check
    }
    if ((xSize > 1UL) && (weightSize > 1UL) && (ySize == 1UL)) {
        OP_CHECK_IF(weightSize != paramsInfo.groupNum,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "weight",
                        Ops::Transformer::Gmm::FormatString("weight Size [%zu] does not equal with groupNum %zu",
                                                            weightSize, paramsInfo.groupNum)
                            .c_str()),
                    return GRAPH_FAILED);
        OP_CHECK_IF(
            SplitMSeparatedXSeparatedWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                     "Split m, separated x, separated weight, single y case failed"),
            return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
        context->GetNodeName(), "groupType",
        Ops::Transformer::Gmm::FormatString(
            "When groupType is 0, current case with x %zu, weight %zu, y %zu is not supported", xSize, weightSize,
            ySize)
            .c_str());
    return GRAPH_FAILED;
}

static ge::graphStatus CheckCaseSplitK(gert::InferShapeContext *context, bool transposeX, bool transposeWeight,
                                       const GMMParamsInfo &paramsInfo)
{
    std::vector<std::string> tenorXAndWeight{"x", "weight", "true"};
    const size_t &xSize = paramsInfo.numX;
    const size_t &weightSize = paramsInfo.numWeight;
    const size_t &ySize = paramsInfo.numY;
    if (xSize == 1UL) {
        if (paramsInfo.platform == PlatformID::ASCEND950) {
            return GRAPH_SUCCESS;
        }
        OP_CHECK_IF(!transposeX,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "groupType",
                        "When groupType is 2 and x is not separated, tensor in x should be transposed"),
                    return GRAPH_FAILED);
        // check dimension
        OP_CHECK_IF(CheckDimNum(context, xSize, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                             "Dim num or format of tensor in tensor list x is invalid"),
                    return GRAPH_FAILED);
        OP_CHECK_IF(
            CheckDimNum(context, weightSize, GMM_SPLIT_K_SINGLE_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                     "Dim num or format of tensor in tensor list weight is invalid"),
            return GRAPH_FAILED);
        // check shape, x(m,k), weight(k,n), y(b,m,n)
        int64_t innerAxisDimId = 1; // x always is transposed, and the inner axis is always the last axis, M axis.
        size_t kAxisOfWeight = transposeWeight ? 1UL : 0UL;
        if ((weightSize == 1UL) && (ySize == 1UL)) {
            OP_CHECK_IF(CheckShapeSameLengthTensorList(context, {0, kAxisOfWeight}, innerAxisDimId, tenorXAndWeight,
                                                       xSize) != GRAPH_SUCCESS,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                                 "k dim value of x and weight is not matched"),
                        return GRAPH_FAILED);
            innerAxisDimId = 1; // w always is not transposed, and the inner axis is always the last axis, N axis.
            // check groupList
            OP_CHECK_IF(
                CheckGroupListCommonTensor(context, true, 1) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupList", "Invalid groupList"),
                return GRAPH_FAILED);
        }
        OP_CHECK_IF(
            CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_WEIGHT, innerAxisDimId, weightSize, "weight") !=
                GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "weight",
                Ops::Transformer::Gmm::FormatString("inner axis size of weight is larger than %ld", GMM_MAX_INNER_AXIS)
                    .c_str()),
            return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
        context->GetNodeName(), "groupType",
        Ops::Transformer::Gmm::FormatString("When groupType is 2, only support case with unseparated x, weight and y, "
                                            "but now x size is %lu, weight size is %lu, y size is %lu",
                                            xSize, weightSize, ySize)
            .c_str());
    return GRAPH_FAILED;
}

static ge::graphStatus CheckParamDifferentGroupType(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                                    const GMMParamsInfo &paramsInfo)
{
    OP_CHECK_IF(paramsInfo.platform == PlatformID::UNKNOWN,
                OP_LOGW(context->GetNodeName(), "Cannot get platform info!"), return GRAPH_SUCCESS);
    const int64_t &groupType = gmmAttrs.groupType;
    const bool &transposeX = gmmAttrs.transposeX;
    const bool &transposeWeight = gmmAttrs.transposeWeight;
    OP_CHECK_IF(transposeX && transposeWeight,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "transposeX, transposeWeight",
                    Ops::Transformer::Gmm::FormatString("%d, %d", static_cast<int>(transposeX),
                                                        static_cast<int>(transposeWeight))
                        .c_str(),
                    "x and weight can not be transposed at the same time"),
                return GRAPH_FAILED);
    auto groupTensorOptionalShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTypePtr);
    size_t validGroupTensorDimNum = (*groupListTypePtr == 2L) ? 2UL : 1UL; // 2: split M sparse, group list shape [e, 2]
    OP_CHECK_IF(
        groupTensorOptionalShape != nullptr &&
            (groupTensorOptionalShape->GetDimNum() > validGroupTensorDimNum || groupTensorOptionalShape->GetDim(0) < 1),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "groupList",
            Ops::Transformer::Gmm::FormatString(
                "If groupList is a tensor, its dim num must be 1, or 2 when groupListType is 2, and the "
                "size of the first dimension must be greater than 0. Current values: groupListType=%ld, "
                "dim num=%zu, dim0=%ld",
                *groupListTypePtr, groupTensorOptionalShape->GetDimNum(), groupTensorOptionalShape->GetDim(0))
                .c_str()),
        return GRAPH_FAILED);
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P && !(groupType == GMM_SPLIT_M && paramsInfo.numX == 1 &&
                                                                   paramsInfo.numWeight == 1 && paramsInfo.numY == 1),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "groupType, x, weight, y",
                    Ops::Transformer::Gmm::FormatString("%ld, %zu, %zu, %zu", groupType, paramsInfo.numX,
                                                        paramsInfo.numWeight, paramsInfo.numY)
                        .c_str(),
                    "When on ASCEND310P, it only supports split m, single x, single weight, single y"),
                return GRAPH_FAILED);

    if (groupType == GMM_NO_SPLIT) {
        OP_CHECK_IF(
            transposeX,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "transposeX", "true",
                                                  "When x, weight and y are all separated, x can not be transposed"),
            return GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseNoSplit(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "Invalid inputs"),
                    return GRAPH_FAILED);
    } else if (groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(transposeX,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "transposeX", "true",
                                                          "When groupType is 0, x can not be transposed"),
                    return GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseSplitM(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "Invalid inputs"),
                    return GRAPH_FAILED);
    } else if (groupType == GMM_SPLIT_K) {
        OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_BIAS),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "bias",
                                                             "When groupType is 2, bias must be empty"),
                    return GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseSplitK(context, transposeX, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "Invalid inputs"),
                    return GRAPH_FAILED);
    }
    if (!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_BIAS)) {
        OP_CHECK_IF(CheckOptionalTensorList(context, "bias", paramsInfo, gmmAttrs, GMM_INDEX_IN_BIAS) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "bias", "Invalid bias"),
                    return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus XNotSingleYSeparated(gert::InferShapeContext *context, size_t weightDimN, bool isXTransposed,
                                            size_t xDimM)
{
    const gert::Tensor *groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
    if (groupListTensor != nullptr) {
        OP_CHECK_IF(
            UpdateMultipleShapeY(context, groupListTensor, weightDimN, isXTransposed, xDimM) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "y", "Failed to update shape of y when x is not single and y is separated"),
            return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(MultiInMultiOutWithoutGroupList(context) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "input", "Failed to process multi-in-multi-out case without GroupList"),
                    return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus XSingleYSeparated(gert::InferShapeContext *context, size_t weightDimN, bool isXTransposed,
                                         size_t xDimM)
{
    const gert::Tensor *groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_IF(groupListTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "groupList"),
                return GRAPH_FAILED);
    OP_CHECK_IF(UpdateMultipleShapeY(context, groupListTensor, weightDimN, isXTransposed, xDimM) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "y", "Failed to update shape of y when x is single and y is separated"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GMMSetOutputShape(gert::InferShapeContext *context, GMMAttrs &gmmAttrs,
                                         const GMMSetOutputParams &outputParams, const gert::Shape *x0Shape,
                                         const gert::Shape *w0Shape)
{
    bool isSingleX = outputParams.isSingleX;
    bool isSingleY = outputParams.isSingleY;
    size_t xDimM = outputParams.xDimM;
    size_t weightDimN = outputParams.weightDimN;
    size_t numX = outputParams.numX;
    size_t numWeight = outputParams.numWeight;
    int64_t lenGroupList = outputParams.lenGroupList;
    // X单 Y多
    if (isSingleX && !isSingleY) {
        if (gmmAttrs.groupType != GMM_SPLIT_K) {
            OP_CHECK_IF(
                XSingleYSeparated(context, weightDimN, gmmAttrs.transposeX, xDimM) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "y", "Failed to update shape of y in the single-x, separated-y case"),
                return GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                MultiWeightMultiOutWithoutGroupList(context) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "input", "Failed to process multi-weight-multi-out case without GroupList"),
                return GRAPH_FAILED);
        }
        // X单 Y单
    } else if (isSingleX && isSingleY) {
        OP_CHECK_IF(gmmAttrs.groupType != GMM_SPLIT_M && gmmAttrs.groupType != GMM_SPLIT_K,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "x",
                        "When x is single tensor, input tensors can only be split along M or K axis"),
                    return GRAPH_FAILED);
        std::vector<int64_t> yDims = {x0Shape->GetDim(xDimM), w0Shape->GetDim(weightDimN)};
        if (gmmAttrs.groupType == GMM_SPLIT_K) {
            yDims.insert(yDims.begin(), numWeight == 1 ? lenGroupList : numWeight);
        }
        OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "y", "Failed to update shape of y in the single-x, single-y case"),
                    return GRAPH_FAILED);
    }
    // X多 Y多
    else if (!isSingleX && !isSingleY) {
        OP_CHECK_IF(
            XNotSingleYSeparated(context, weightDimN, gmmAttrs.transposeX, xDimM) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "y", "Failed to update shape of y in the separated-x, separated-y case"),
            return GRAPH_FAILED);
    }
    // X多 Y单
    else if (!isSingleX && isSingleY) {
        std::vector<int64_t> yDims = {GetDim0(context, gmmAttrs.transposeX, numX, xDimM), w0Shape->GetDim(weightDimN)};
        OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "y", "Failed to update shape of y in the separated-x, single-y case"),
                    return GRAPH_FAILED);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferShape4DavidWeightQuantGMM(gert::InferShapeContext *context)
{
    GroupedMatmulWeightQuantChecker davidWeightQuantGMMChecker;
    GroupedMatmulCommonUtil utilForDavidWeightQuantGMM;
    OP_CHECK_IF(GetAttrsValue(context, utilForDavidWeightQuantGMM.attrsInfo) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetAttrsValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.GetXAndWeightDimValue(context, utilForDavidWeightQuantGMM.attrsInfo) !=
                    GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetXAndWeightDimValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.CheckShape(context, utilForDavidWeightQuantGMM) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShape failed"), return GRAPH_FAILED);
    OP_CHECK_IF(
        davidWeightQuantGMMChecker.InferOutShape(context, utilForDavidWeightQuantGMM.attrsInfo) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "InferOutShape failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus InferShape4DavidQuantGMM(gert::InferShapeContext *context)
{
    GroupedMatmulQuantChecker davidQuantGMMChecker;
    GroupedMatmulCommonUtil utilForDavidQuantGMM;
    OP_CHECK_IF(GetAttrsValue(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetAttrsValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.GetXAndWeightDimValue(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetXAndWeightDimValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.GetGroupNumValue(context) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetGroupNumValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.CheckShape(context, utilForDavidQuantGMM) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShape failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.InferOutShape(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "InferOutShape failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

template <typename T>
static graphStatus IsDavidWeightQuantGMMByShape(T context)
{
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    DataType xDtype = xDesc->GetDataType();
    DataType weightDtype = weightDesc->GetDataType();
    return GetSizeByDataType(xDtype) != GetSizeByDataType(weightDtype) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

template <typename T>
static graphStatus IsDavidQuantGMMByShape(T context)
{
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    auto scaleDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    DataType xDtype = xDesc->GetDataType();
    DataType weightDtype = weightDesc->GetDataType();
    if (xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_INT4 || xDtype == ge::DT_FLOAT4_E1M2) {
        return GRAPH_SUCCESS;
    }
    return (GetSizeByDataType(xDtype) == 1 && GetSizeByDataType(weightDtype) == 1) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

static ge::graphStatus TryDavidInferShape(gert::InferShapeContext *context, bool &isDavidCase)
{
    isDavidCase = false;
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret != GRAPH_SUCCESS || GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) == 0 ||
        IsS8S4PseudoQuant(context)) {
        return GRAPH_FAILED; // not handled by David path
    }
    if (IsDavidQuantGMMByShape(context) == GRAPH_SUCCESS) {
        isDavidCase = true;
        OP_CHECK_IF(InferShape4DavidQuantGMM(context) != GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Check params failed"), return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    } else if (IsDavidWeightQuantGMMByShape(context) == GRAPH_SUCCESS) {
        isDavidCase = true;
        OP_CHECK_IF(InferShape4DavidWeightQuantGMM(context) != GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Check params failed"), return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED; // not a David quant case
}

static ge::graphStatus ParseAttrsAndCountInputs(gert::InferShapeContext *context, GMMAttrs &gmmAttrs, size_t &numX,
                                                size_t &numWeight, int64_t &lenGroupList)
{
    numX = 0;
    numWeight = 0;
    lenGroupList = 0;
    if (GetNumOfInputs(context, numX, numWeight, lenGroupList) != GRAPH_SUCCESS) {
        OP_CHECK_IF(CheckDimNum(context, numX, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                             "Dim num of tensor in tensorList x is invalid"),
                    return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    size_t numY = context->GetComputeNodeOutputNum();
    GMMParamsInfo paramsInfo{numX, numWeight, numY, lenGroupList, 0, 0, 0, 0, 0, PlatformID::UNKNOWN};
    OP_CHECK_IF(GetGroupSize(context, paramsInfo) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "check groupNum failed"),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        CheckFunctionParamsForShape(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "CheckFunctionParamsForShape failed"),
        return GRAPH_FAILED);
    OP_CHECK_IF(CheckParamDifferentGroupType(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "groupType",
                                                         "CheckParamDifferentGroupType failed"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus ComputeAndSetOutputShape(gert::InferShapeContext *context, GMMAttrs &gmmAttrs, size_t numX,
                                                size_t numWeight, int64_t lenGroupList)
{
    const gert::Shape *x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Shape);
    size_t xDimNum = x0Shape->GetDimNum();
    const gert::Shape *w0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
    size_t weightDimNum = w0Shape->GetDimNum();
    size_t numY = context->GetComputeNodeOutputNum();
    bool isSingleX = (numX == 1UL) && (gmmAttrs.groupType != GMM_NO_SPLIT);
    bool isSingleY = (numY == 1UL) && (gmmAttrs.groupType != GMM_NO_SPLIT);
    size_t xDimM = gmmAttrs.transposeX ? xDimNum - 1UL : xDimNum - 2UL;
    size_t weightDimN = GetWeightAxisInfo(context, weightDimNum, gmmAttrs.transposeWeight).n;

    GMMSetOutputParams outputParams;
    outputParams.isSingleX = isSingleX;
    outputParams.isSingleY = isSingleY;
    outputParams.xDimM = xDimM;
    outputParams.numX = numX;
    outputParams.weightDimN = weightDimN;
    outputParams.lenGroupList = lenGroupList;
    outputParams.numWeight = numWeight;
    OP_CHECK_IF(GMMSetOutputShape(context, gmmAttrs, outputParams, x0Shape, w0Shape) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "GMMSetOutputShape failed"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GroupedMatmul(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    bool isDavidCase = false;
    ge::graphStatus davidRet = TryDavidInferShape(context, isDavidCase);
    if (isDavidCase) {
        return davidRet;
    }
    GMMAttrs gmmAttrs{GMM_X_Y_SEPARATED, 0, GMM_NO_SPLIT, false, false, 0, 0};
    OP_CHECK_IF(GetAttrsValue(context, gmmAttrs) != GRAPH_SUCCESS || CheckAttrs(context, gmmAttrs) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "Failed to get attrs"),
                return GRAPH_FAILED);
    size_t numX = 0;
    size_t numWeight = 0;
    int64_t lenGroupList = 0;
    OP_CHECK_IF(
        ParseAttrsAndCountInputs(context, gmmAttrs, numX, numWeight, lenGroupList) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "ParseAttrsAndCountInputs failed"),
        return GRAPH_FAILED);
    return ComputeAndSetOutputShape(context, gmmAttrs, numX, numWeight, lenGroupList);
}

// =========================================================================================
// =========================================================================================
static graphStatus CheckTensorListDataType(const gert::InferDataTypeContext *context, uint32_t index,
                                           const DataType dtype, const char *tensorType)
{
    size_t inIdx = 0;
    while (true) {
        auto iDtype = context->GetDynamicInputDataType(index, inIdx);
        if (iDtype == DT_UNDEFINED) {
            break;
        }
        OP_CHECK_IF(
            iDtype != dtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), Ops::Transformer::Gmm::FormatString("%s[%zu]", tensorType, inIdx).c_str(),
                TypeUtils::DataTypeToAscendString(iDtype).GetString(),
                Ops::Transformer::Gmm::FormatString("%s dtype should match %s", tensorType,
                                                    TypeUtils::DataTypeToAscendString(dtype).GetString())
                    .c_str()),
            return GRAPH_FAILED);
        ++inIdx;
    }
    return GRAPH_SUCCESS;
}

static graphStatus CheckMatmulDataType(gert::InferDataTypeContext *context, const DataType xDtype,
                                       const DataType weightDtype, const DataType biasDtype)
{
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_X, xDtype, "x") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "x",
                    Ops::Transformer::Gmm::FormatString("x dtype does not match with required dtype[%s]",
                                                        TypeUtils::DataTypeToAscendString(xDtype).GetString())
                        .c_str()),
                return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_WEIGHT, weightDtype, "weight") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "weight",
                    Ops::Transformer::Gmm::FormatString("weight dtype does not match with required dtype[%s]",
                                                        TypeUtils::DataTypeToAscendString(weightDtype).GetString())
                        .c_str()),
                return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, biasDtype, "bias") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "bias",
                    Ops::Transformer::Gmm::FormatString("bias dtype does not match with required dtype[%s]",
                                                        TypeUtils::DataTypeToAscendString(biasDtype).GetString())
                        .c_str()),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckNonQuantMatmulParams(fe::PlatformInfo &platformInfo, gert::InferDataTypeContext *context,
                                             const DataType xDtype, const DataType weightDtype)
{
    DataType biasDtype = xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT : xDtype;
    if (GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
        if (biasDtype != DT_UNDEFINED) {
            OP_CHECK_IF(std::find(BIAS_DTYPE_SUPPORT_LIST.begin(), BIAS_DTYPE_SUPPORT_LIST.end(), biasDtype) ==
                            BIAS_DTYPE_SUPPORT_LIST.end(),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context->GetNodeName(), "bias", TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                            "non quant case bias only supports dtype float16, bfloat16 and float32"),
                        return GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype) != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, weight",
                    Ops::Transformer::Gmm::FormatString("%s, %s", TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                                                        TypeUtils::DataTypeToAscendString(weightDtype).GetString())
                        .c_str(),
                    "case with x dtype and weight dtype is not supported"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckFunctionQuantParams(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_X, DataType::DT_INT8, "x") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "x dtype does not match with required dtype[INT8]"),
                return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_WEIGHT, DataType::DT_INT8, "weight") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "weight dtype does not match with required dtype[INT8]"),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, DataType::DT_INT32, "bias") != GRAPH_SUCCESS) &&
            (CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, DataType::DT_BF16, "bias") != GRAPH_SUCCESS),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "bias",
            TypeUtils::DataTypeToAscendString(context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0)).GetString(),
            "bias dtype must be INT32 or BF16"),
        return GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    if (*outputDtype == GMM_OUT_DTYPE_INT32) { // output dtype is int32, this scene does not need scale
        return GRAPH_SUCCESS;
    }
    auto scale0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    // Now we cannot make sure if is pertoken quant case, so scale/offset dtype check is remained to the InferShape
    // stage.
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_SCALE, scale0Dtype, "scale") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale",
                                                         "dtypes of scales in the tensorList should all be the same"),
                return GRAPH_FAILED);
    auto offset0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_OFFSET, 0);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_OFFSET, offset0Dtype, "offset") != GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "offset",
                                                         "dtypes of offsets in the tensorList should all be the same"),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckGroupedMatmulAntiQuantForDtype(gert::InferDataTypeContext *context)
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    OP_CHECK_IF(
        CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_SCALE, xDtype, "antiquantScale") != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "antiquantScale",
            Ops::Transformer::Gmm::FormatString("antiquantScale dtype does not match with x dtype[%s]",
                                                TypeUtils::DataTypeToAscendString(xDtype).GetString())
                .c_str()),
        return GRAPH_FAILED);
    OP_CHECK_IF(
        CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, xDtype, "antiquantOffset") != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "antiquantOffset",
            Ops::Transformer::Gmm::FormatString("antiquantOffset dtype does not match with x dtype[%s]",
                                                TypeUtils::DataTypeToAscendString(xDtype).GetString())
                .c_str()),
        return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckFunctionParamsForDtype(gert::InferDataTypeContext *context)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    graphStatus ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    PlatformID platform = PlatformID::UNKNOWN;
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGW(context->GetNodeName(), "Cannot get platform info.");
        return GRAPH_SUCCESS;
    } else {
        platform = (optionalInfo.soc_version.find("310P") != std::string::npos) ? PlatformID::ASCEND310P :
                   (optionalInfo.soc_version.find("950") != std::string::npos)  ? PlatformID::ASCEND950 :
                                                                                  PlatformID::ASCEND910B;
    }
    DataType xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    DataType weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    if (platform == PlatformID::ASCEND310P) {
        bool isAllInputFP16 = xDtype == DataType::DT_FLOAT16 && weightDtype == DataType::DT_FLOAT16;
        OP_CHECK_IF(
            !isAllInputFP16,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), "x, weight",
                Ops::Transformer::Gmm::FormatString("%s, %s", TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                                                    TypeUtils::DataTypeToAscendString(weightDtype).GetString())
                    .c_str(),
                "Only float16 is supported on Ascend310P platforms"),
            return GRAPH_FAILED);
        auto biasDtype = context->GetOptionalInputDataType(GMM_INDEX_IN_BIAS);
        OP_CHECK_IF(biasDtype != ge::DT_UNDEFINED && biasDtype != DataType::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "bias",
                                                          TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                                                          "only bias float16 is supported on Ascend310P platforms"),
                    return GRAPH_FAILED);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {
        return GRAPH_SUCCESS;
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16 || xDtype == DataType::DT_FLOAT) &&
        xDtype == weightDtype) { // nonquant
        return CheckNonQuantMatmulParams(platformInfo, context, xDtype, weightDtype);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        // quant
        OP_CHECK_IF(CheckFunctionQuantParams(context) != GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                             "CheckFunctionQuantParams failed"),
                    return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16) &&
        (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
        // antiquant
        DataType biasDtype = xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT : DataType::DT_FLOAT16;
        OP_CHECK_IF(
            CheckMatmulDataType(context, xDtype, weightDtype, biasDtype) != GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), "x, weight",
                Ops::Transformer::Gmm::FormatString("%s, %s", TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                                                    TypeUtils::DataTypeToAscendString(weightDtype).GetString())
                    .c_str(),
                "case with x dtype and weight dtype is not supported"),
            return GRAPH_FAILED);
        return CheckGroupedMatmulAntiQuantForDtype(context);
    }
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        context->GetNodeName(), "x, weight",
        Ops::Transformer::Gmm::FormatString("%s, %s", TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                                            TypeUtils::DataTypeToAscendString(weightDtype).GetString())
            .c_str(),
        "GMM: there is no matching xDtype and weightDtype pattern");
    return GRAPH_FAILED;
}

static graphStatus CheckQuantParamsDtype(const gert::InferDataTypeContext *context, const int64_t outputDtype,
                                         const DataType yDtype)
{
    size_t i = 0;
    auto scale0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_IF(scale0Dtype == ge::DT_UNDEFINED,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "scale",
                                                      TypeUtils::DataTypeToAscendString(scale0Dtype).GetString(),
                                                      "scale dtype is undefined"),
                return GRAPH_FAILED);
    auto perTokenScale0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    bool isPerTokenQuant = perTokenScale0Dtype != ge::DT_UNDEFINED;
    if (isPerTokenQuant) {
        bool isOutputBF16 = scale0Dtype == DataType::DT_BF16 && outputDtype == 1;
        bool isOutputFloat16 = scale0Dtype == DataType::DT_FLOAT && outputDtype == 0;
        OP_CHECK_IF(
            !isOutputBF16 && !isOutputFloat16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "scale", TypeUtils::DataTypeToAscendString(scale0Dtype).GetString(),
                Ops::Transformer::Gmm::FormatString("per-token quant only supports scale dtype BF16 with output "
                                                    "dtype BF16, or scale dtype FP32 with output dtype FP16, but "
                                                    "output dtype is %s",
                                                    TypeUtils::DataTypeToAscendString(yDtype).GetString())
                    .c_str()),
            return GRAPH_FAILED);
    } else {
        bool isOutputInt8 = scale0Dtype == DataType::DT_UINT64 && outputDtype == -1;
        bool isOutputBF16 = scale0Dtype == DataType::DT_BF16 && outputDtype == 1;
        bool isOutputFP16 = scale0Dtype == DataType::DT_FLOAT && outputDtype == 0;
        OP_CHECK_IF(
            !isOutputInt8 && !isOutputBF16 && !isOutputFP16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "scale", TypeUtils::DataTypeToAscendString(scale0Dtype).GetString(),
                Ops::Transformer::Gmm::FormatString("per-channel quant only supports scale dtype UINT64 with "
                                                    "output dtype INT8, BF16 with output dtype BF16, or FP32 with "
                                                    "output dtype FP16, but output dtype is %s",
                                                    TypeUtils::DataTypeToAscendString(yDtype).GetString())
                    .c_str()),
            return GRAPH_FAILED);
    }
    if (isPerTokenQuant) {
        OP_CHECK_IF(
            perTokenScale0Dtype != DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "perTokenScale",
                                                  TypeUtils::DataTypeToAscendString(perTokenScale0Dtype).GetString(),
                                                  "perTokenScale dtype must be float32"),
            return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4DavidWeightQuantGMM(gert::InferDataTypeContext *context)
{
    GroupedMatmulWeightQuantChecker davidWeightQuantGMMChecker;
    OP_CHECK_IF(davidWeightQuantGMMChecker.CheckDtype(context) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckDtype failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.InferOutDtype(context) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetYDtype failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4DavidQuantGMM(gert::InferDataTypeContext *context)
{
    GroupedMatmulQuantChecker davidQuantGMMChecker;
    GroupedMatmulCommonUtil utilForDavidQuantGMM;
    OP_CHECK_IF(GetAttrsValue(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetAttrsValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.CheckDtype(context, utilForDavidQuantGMM) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckDtype failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.InferOutDtype(context) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetYDtype failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmul(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        if (IsDavidQuantGMMByShape(context) == GRAPH_SUCCESS) {
            OP_CHECK_IF(InferDtype4DavidQuantGMM(context) != GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "InferDtype4DavidQuantGMM failed"), return GRAPH_FAILED);
            return GRAPH_SUCCESS;
        } else if (IsDavidWeightQuantGMMByShape(context) == GRAPH_SUCCESS) {
            OP_CHECK_IF(InferDtype4DavidWeightQuantGMM(context) != GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "InferDtype4DavidWeightQuantGMM failed"), return GRAPH_FAILED);
            return GRAPH_SUCCESS;
        }
    }
    OP_CHECK_IF(
        CheckFunctionParamsForDtype(context) != GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input", "CheckFunctionParamsForDtype failed"),
        return GRAPH_FAILED);

    auto x0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weight0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    size_t numY = context->GetComputeNodeOutputNum();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    bool isQuantCase = x0Dtype == ge::DT_INT8 && weight0Dtype == ge::DT_INT8;
    bool isA8W4 = x0Dtype == ge::DT_INT8 && weight0Dtype == ge::DT_INT4;
    const int64_t *outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    DataType yDtype = x0Dtype;
    if (isQuantCase && outputDtype != nullptr) {
        auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
        OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", std::to_string(*outputDtype),
                                                          "value of attr dtype only supports -1/0/1/2"),
                    return GRAPH_FAILED);
        yDtype = it->second;
        if (*outputDtype != GMM_OUT_DTYPE_INT32) { // output dtype is int32, this scene does not need scale
            OP_CHECK_IF(CheckQuantParamsDtype(context, *outputDtype, yDtype) != GRAPH_SUCCESS,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "input",
                                                                 "Check quant params data type failed"),
                        return GRAPH_FAILED);
        }
    }
    if (isA8W4 && outputDtype != nullptr) {
        auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
        OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", std::to_string(*outputDtype),
                                                          "value of attr dtype only supports -1/0/1/2"),
                    return GRAPH_FAILED);
        yDtype = it->second;
    }
    for (size_t k = 0; k < numY; k++) {
        context->SetOutputDataType(GMM_INDEX_OUT_Y + k, yDtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmul).InferShape(InferShape4GroupedMatmul).InferDataType(InferDataType4GroupedMatmul);
} // namespace ops
