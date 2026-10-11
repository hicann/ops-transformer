/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_tiling_base.cpp
 * \brief
 */
#include <cmath>
#include "moe_init_routing_v2_grad_tiling.h"
using namespace AscendC;
namespace optiling {
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "platformInfo", "nullptr", "Failed to get platform info."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion = ascendcPlatform.GetSocVersion();

    auto compileInfoPtr = reinterpret_cast<const MoeInitRoutingV2GradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "compileInfo", "nullptr",
                                                      "Compile info should not be null."),
                return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    aicoreParams_.numBlocks = aivNum;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckDtypeValidity()
{
    const std::vector<ge::DataType> DATA_TYPE_SUPPORT = {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16,
                                                         ge::DataType::DT_BF16};
    // 获取输入dtype
    auto gradExpandedXDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradExpandedXDesc);
    inDtype = gradExpandedXDesc->GetDataType();
    auto expandedRowIdxDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxDesc);
    auto rowIdxDtype = expandedRowIdxDesc->GetDataType();
    auto gradXDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    auto outDtype = gradXDesc->GetDataType();
    if (inDtype != outDtype) {
        std::string dtypeMsg = Ops::Base::ToString(inDtype) + " and " + Ops::Base::ToString(outDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "grad_expanded_x and grad_x", dtypeMsg.c_str(),
                                               "The dtype of grad_expanded_x and grad_x must be the same.");
        return ge::GRAPH_FAILED;
    }
    if (rowIdxDtype != ge::DataType::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(rowIdxDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expanded_row_idx", dtypeStr.c_str(), "INT32");
        return ge::GRAPH_FAILED;
    }
    auto it = std::find(DATA_TYPE_SUPPORT.begin(), DATA_TYPE_SUPPORT.end(), inDtype);
    if (it == DATA_TYPE_SUPPORT.end()) {
        std::string dtypeStr = Ops::Base::ToString(inDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_expanded_x", dtypeStr.c_str(),
                                  "FLOAT, FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckShapeAllPositive(const gert::Shape& shape, std::string name)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), (name + " dim " + std::to_string(i)).c_str(),
                        std::to_string(shape.GetDim(i)).c_str(), "Shape dim should be positive."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckShapeValidity(const gert::Shape& xShape,
                                                                        const gert::Shape& rowIdxShape,
                                                                        const gert::Shape& gradXShape)
{
    if (CheckShapeAllPositive(rowIdxShape, "rowIdxShape") != ge::GRAPH_SUCCESS ||
        CheckShapeAllPositive(gradXShape, "gradXShape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode == 0 && activeNum == 0) {
        if (CheckShapeAllPositive(xShape, "xShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 索引含-1场景，xShape可以有0
        for (size_t i = 0; i < xShape.GetDimNum(); i++) {
            OP_CHECK_IF(xShape.GetDim(i) < 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(),
                                                              ("grad_expanded_x dim " + std::to_string(i)).c_str(),
                                                              std::to_string(xShape.GetDim(i)).c_str(),
                                                              "Shape dim should be greater than or equal to 0."),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::CheckParamsValidity(const gert::Shape& xShape,
                                                                         const gert::Shape& rowIdxShape,
                                                                         const gert::Shape& gradXShape) const
{
    // attr属性校验
    if (dropPadMode != 0 && dropPadMode != 1) {
        std::string dropPadModeStr = std::to_string(dropPadMode);
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "drop_pad_mode", dropPadModeStr.c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }

    if (topK <= 0) {
        std::string topKStr = std::to_string(topK);
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "top_k", topKStr.c_str(), "greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (activeNum < 0) {
        std::string activeNumStr = std::to_string(activeNum);
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "active_num", activeNumStr.c_str(),
                                  "greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }

    // shape校验
    size_t xDimNnum = xShape.GetDimNum();
    if (xDimNnum != DIM_TWO && xDimNnum != DIM_THREE) {
        std::string dimNumStr = std::to_string(xDimNnum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_expanded_x", dimNumStr.c_str(), "2D or 3D");
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode == 1 && xDimNnum != DIM_THREE) {
        std::string dimNumStr = std::to_string(xDimNnum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_expanded_x", dimNumStr.c_str(), "3D");
        return ge::GRAPH_FAILED;
    }

    if (dropPadMode == 0) {
        if (activeNum == 0 && xShape.GetDim(0) != rowIdxShape.GetDim(0)) {
            std::string shapeMsg = Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(rowIdxShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_expanded_x and expanded_row_idx",
                                                   shapeMsg.c_str(),
                                                   "Dim 0 size should be the same under dropless mode.");
            return ge::GRAPH_FAILED;
        }

        if (activeNum > 0 && xShape.GetDim(0) != activeNum) {
            std::string shapeStr = Ops::Base::ToString(xShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "grad_expanded_x", shapeStr.c_str(),
                "Dim 0 size of grad_expanded_x should be equal to active_num under active scene.");
            return ge::GRAPH_FAILED;
        }
    }

    size_t outDimNum = gradXShape.GetDimNum();
    if (outDimNum != DIM_TWO) {
        std::string dimNumStr = std::to_string(outDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_x", dimNumStr.c_str(), "2D");
        return ge::GRAPH_FAILED;
    }

    int64_t hiddenSizeLocal =
        (dropPadMode == 0) ? xShape.GetDim(1) : xShape.GetDim(2); // 2: drop/pad 场景，尾轴在第三维
    if (gradXShape.GetDim(1) != hiddenSizeLocal) {
        std::string shapeMsg = Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(gradXShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_expanded_x and grad_x", shapeMsg.c_str(),
                                               "Tail dim size of input and output should be the same.");
        return ge::GRAPH_FAILED;
    }

    if (gradXShape.GetDim(0) != rowIdxShape.GetDim(0) / topK) {
        std::string shapeMsg = Ops::Base::ToString(rowIdxShape) + " and " + Ops::Base::ToString(gradXShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "expanded_row_idx and grad_x", shapeMsg.c_str(),
            "Dim 0 of grad_x should be equal to expanded_row_idx dim 0 divided by top_k.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();

    // 获取输入shape
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    auto rowIdxShapePtr = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowIdxShapePtr);
    const gert::Shape rowIdxShape = rowIdxShapePtr->GetStorageShape();
    auto gradXShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShapePtr);
    const gert::Shape gradXShape = gradXShapePtr->GetStorageShape();

    // 获取输入属性
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* topKPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topKPtr);
    topK = *topKPtr;
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(1); // 1: drop_pad_mode attr
    dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(2); // 2: active_num attr
    activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;

    ge::graphStatus res = CheckDtypeValidity();
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    // 参数校验
    res = CheckParamsValidity(xShape, rowIdxShape, gradXShape);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    // shape校验
    if (CheckShapeValidity(xShape, rowIdxShape, gradXShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置tilingData基本参数
    int64_t BSK = rowIdxShape.GetDim(0); // A: B*S*K
    E = (dropPadMode == 0) ? 0 : xShape.GetDim(0);
    C = (dropPadMode == 0) ? 0 : xShape.GetDim(1);
    hiddenSize = (dropPadMode == 0) ? xShape.GetDim(1) : xShape.GetDim(2); // 2 for H, when shape is {E, C, H}
    if (BSK % topK != 0) {
        std::string actualMsg = "expanded_row_idx dim 0: " + std::to_string(BSK) + ", top_k: " + std::to_string(topK);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "expanded_row_idx dim 0 and top_k",
                                              actualMsg.c_str(),
                                              "expanded_row_idx dim 0 should be divisible by top_k.");
        return ge::GRAPH_FAILED;
    }
    N = BSK / topK;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2GradTilingBaseClass::GetWorkspaceSize()
{
    // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
    workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMoeInitRoutingV2Grad(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMoeInitRoutingV2Grad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForMoeInitRountingV2Grad enter.");

    auto compileInfo = context->GetCompiledInfo<MoeInitRoutingV2GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->aivNum <= 0) {
        std::string aivNumStr = std::to_string(compileInfo->aivNum);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "aivNum", aivNumStr.c_str(),
                                              "Failed to get valid core num.");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    if (compileInfo->ubSize <= 0) {
        std::string ubSizeStr = std::to_string(compileInfo->ubSize);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", ubSizeStr.c_str(),
                                              "Failed to get valid ub size.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingV2Grad)
    .Tiling(TilingForMoeInitRoutingV2Grad)
    .TilingParse<MoeInitRoutingV2GradCompileInfo>(TilingPrepareForMoeInitRoutingV2Grad);
} // namespace optiling
