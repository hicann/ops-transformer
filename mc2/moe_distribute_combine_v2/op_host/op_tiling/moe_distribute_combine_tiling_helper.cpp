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
 * \file moe_distribute_combine_tiling_helper.cc
 * \brief
 */

#include "moe_distribute_combine_tiling_helper.h"
#include "mc2_log.h"

using namespace ge;

namespace optiling {
static bool CheckRequiredInputDim2D(const gert::TilingContext *context, const char *nodeName, uint32_t inputIndex,
                                    const char *tensorName)
{
    const gert::StorageShape *shape = context->GetInputShape(inputIndex);
    OP_TILING_CHECK(shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, tensorName), return false);
    OP_TILING_CHECK(shape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, tensorName, std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                        ("The shape dim of " + std::string(tensorName) + " must be 2D.").c_str()),
                    return false);
    OP_LOGD(nodeName, "%s dim0 = %ld", tensorName, shape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "%s dim1 = %ld", tensorName, shape->GetStorageShape().GetDim(1));
    return true;
}

static bool CheckOptionalSharedExpertXDim(const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    if (sharedExpertX == nullptr) {
        return true;
    }
    auto attrs = context->GetAttrs();
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OP_TILING_CHECK(*sharedExpertRankNumPtr != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertX", "present",
                                              "should be None when sharedExpertRankNum is non-zero"),
                    return false);
    OP_TILING_CHECK(((sharedExpertX->GetStorageShape().GetDimNum() != TWO_DIMS) &&
                     (sharedExpertX->GetStorageShape().GetDimNum() != THREE_DIMS)),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "sharedExpertX", std::to_string(sharedExpertX->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of sharedExpertX must be within the range {2D, 3D}"),
                    return false);
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckInputTensorDim(const gert::TilingContext *context,
                                                                  const char *nodeName)
{
    if (!CheckRequiredInputDim2D(context, nodeName, EXPAND_X_INDEX, "expandX")) {
        return false;
    }
    if (!CheckRequiredInputDim2D(context, nodeName, EXPERT_IDS_INDEX, "expertIds")) {
        return false;
    }

    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdx"), return false);
    OP_TILING_CHECK(
        expandIdxStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expandIdx", std::to_string(expandIdxStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expandIdx must be 1D."),
        return false);
    OP_LOGD(nodeName, "expandIdx dim0 = %ld", expandIdxStorageShape->GetStorageShape().GetDim(0));

    return CheckOptionalSharedExpertXDim(context, nodeName);
}

inline bool MoeDistributeCombineTilingHelper::CheckInputSendCountsTensorDim(const gert::TilingContext *context,
                                                                            const char *nodeName)
{
    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCounts"),
                    return false);
    OP_TILING_CHECK(
        epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "epSendCounts", std::to_string(epSendCountsStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of epSendCounts must be 1D."),
        return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckInputExpertScalesTensorDim(const gert::TilingContext *context,
                                                                              const char *nodeName)
{
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScales"),
                    return false);
    OP_TILING_CHECK(
        expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expertScale", std::to_string(expertScalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expertScales must be 2D."),
        return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckOutputTensorDim(const gert::TilingContext *context,
                                                                   const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x", std::to_string(xStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of x must be 2D."),
                    return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorDim(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(
        !CheckInputTensorDim(context, nodeName) || !CheckInputSendCountsTensorDim(context, nodeName) ||
            !CheckInputExpertScalesTensorDim(context, nodeName) || !CheckOutputTensorDim(context, nodeName),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "inputs/outputs", "", "Input/output param shape is invalid."),
        return false);

    // x_active_mask当前不支持传入
    // A3/A5 v1接口的x_active_mask不支持传入
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape != nullptr,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x_active_mask", "present", "should be None"),
                        return false);
    }

    OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x(output)", "", "Output param shape is invalid."),
                    return false);

    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckActiveMask(const gert::TilingContext *context, const char *nodeName)
{
    // Check Dim/DType/Format
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskStorageShape"),
                    return false);
    const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        xActiveMaskDimNums != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "xActiveMask", std::to_string(xActiveMaskDimNums).c_str(),
                                                 "The shape dim of xActiveMask must be 1D."),
        return false);
    auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc"), return false);
    OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "xActiveMask",
                                                          Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str(),
                                                          "The dtype of xActiveMask must be bool."),
                    return false);
    auto xActiveMaskFmt = static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        xActiveMaskFmt == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "xActiveMask", Ops::Base::ToString(xActiveMaskFmt).c_str(), "FRACTAL_NZ"),
        return false);

    return true;
}

// 校验数据类型
static bool CheckRoutingTensorDataTypes(const gert::TilingContext *context, const char *nodeName)
{
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdsDesc"), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expertIds",
                                                          Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of expertIds must be int32."),
                    return false);
    auto expandIdxDesc = context->GetInputDesc(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc"), return false);
    OP_TILING_CHECK((expandIdxDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expandIdx",
                                                          Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str(),
                                                          "The dtype of expandIdx must be int32."),
                    return false);
    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCountsDesc"),
                    return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "epSendCounts",
                                                          Ops::Base::ToString(epSendCountsDesc->GetDataType()).c_str(),
                                                          "The dtype of epSendCounts must be int32."),
                    return false);
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorDataType(const gert::TilingContext *context, const char *nodeName)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expandX",
                                                          Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                                          "The dtype of expandX must be bf16 or float16."),
                    return false);
    if (!CheckRoutingTensorDataTypes(context, nodeName)) {
        return false;
    }
    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScalesDesc"),
                    return false);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expertScales",
                                                          Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str(),
                                                          "The dtype of expertScales must be float."),
                    return false);
    auto sharedExpertXDesc = context->GetOptionalInputDesc(SHARED_EXPERT_X_INDEX);
    if (sharedExpertXDesc != nullptr) {
        OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "sharedExpertX", Ops::Base::ToString(sharedExpertXDesc->GetDataType()).c_str(),
                            "The dtype of sharedExpertX must be the same as that of expandX."),
                        return false);
    }
    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != expandXDesc->GetDataType()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be the same as that of expandX."),
        return false);
    return true;
}

static bool CheckTensorFormatFractalNz(const gert::CompileTimeTensorDesc *desc, const char *nodeName,
                                       const char *descName, const char *tensorName)
{
    OP_TILING_CHECK(desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, descName), return false);
    auto format = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    OP_TILING_CHECK(format == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, tensorName, Ops::Base::ToString(format).c_str(), "FRACTAL_NZ"),
                    return false);
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorFormat(const gert::TilingContext *context, const char *nodeName)
{
    if (!CheckTensorFormatFractalNz(context->GetInputDesc(EXPAND_X_INDEX), nodeName, "expandxDesc", "expandX")) {
        return false;
    }
    if (!CheckTensorFormatFractalNz(context->GetInputDesc(EXPERT_IDS_INDEX), nodeName, "expertIdsDesc", "expertIds")) {
        return false;
    }
    if (!CheckTensorFormatFractalNz(context->GetInputDesc(EXPAND_IDX_INDEX), nodeName, "expandIdxDesc", "expandIdx")) {
        return false;
    }
    if (!CheckTensorFormatFractalNz(context->GetInputDesc(EP_SEND_COUNTS_INDEX), nodeName, "epSendCountsDesc",
                                    "epSendCounts")) {
        return false;
    }
    if (!CheckTensorFormatFractalNz(context->GetInputDesc(EXPERT_SCALES_INDEX), nodeName, "expertScalesDesc",
                                    "expertScales")) {
        return false;
    }
    return CheckTensorFormatFractalNz(context->GetOutputDesc(OUTPUT_X_INDEX), nodeName, "xDesc", "x");
}

ge::graphStatus MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombine(gert::TilingContext *context,
                                                                                  const char *nodeName)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "tensor dims", "", "param shape is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "tensor", "", "param dataType is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "tensor", "", "param Format is invalid"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombineA5(gert::TilingContext *context,
                                                                                    const char *nodeName,
                                                                                    const bool isTokenMask)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "tensor dims", "", "param shape is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "tensor", "", "param dataType is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "tensor", "", "param Format is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查ActiveMask信息
    if ((OpVersionManager::GetInstance().GetVersion() != OP_VERSION_1) && isTokenMask) {
        OP_TILING_CHECK(!CheckActiveMask(context, nodeName),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "xActiveMask", "", "xActiveMask is invalid."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
