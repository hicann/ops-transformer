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
 * \file mc2_common_infershape.cpp
 * \brief
 */

#include "op_host/mc2_common_infershape.h"
#include "mc2_log.h"

using namespace ge;
namespace ops {
namespace {
void ReportInferTypeMetadata(const char* opName, const char* detail)
{
    const std::string reason =
        std::string(detail) + "; actual=null, expected=materialized inference metadata. Rebuild the inference context";
    OP_LOGE(opName, "%s", reason.c_str());
}

void ReportMc2InferShapeFailure(const char* opName, const std::string& reason)
{
    const char* name = opName == nullptr ? "Mc2InferShape" : opName;
    OP_LOGE(name, "%s", reason.c_str());
}
} // namespace

static ge::graphStatus CheckMatrixInputShapes(const gert::InferShapeContext* context, CommParas& commParas)
{
    commParas.x1MatrixShape = context->GetInputShape(0);
    OP_CHECK_IF(commParas.x1MatrixShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x1"),
                return ge::GRAPH_FAILED);
    commParas.x2MatrixShape = context->GetInputShape(1);
    OP_CHECK_IF(commParas.x2MatrixShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x2"),
                return ge::GRAPH_FAILED);
    if (commParas.x1MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x1",
                                     std::to_string(commParas.x1MatrixShape->GetDimNum()) + "D", "2D");
        return ge::GRAPH_FAILED;
    }
    if (commParas.x2MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x2",
                                     std::to_string(commParas.x2MatrixShape->GetDimNum()) + "D", "2D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ResolveRankSize(const gert::InferShapeContext* context, const char* groupStr,
                                       const int64_t* rankSizeAttr, int64_t& rankSize)
{
    if (*rankSizeAttr <= 0) {
#if defined(ENABLE_BUILT_IN)
        uint32_t rankNum = 0;
        if (Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(groupStr, &rankNum) != HCCL_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Get rank size failed, group [%s], rankSize [%u]", groupStr,
                                   rankNum);
            return ge::GRAPH_FAILED;
        }
        if (rankNum == 0) {
            ReportMc2InferShapeFailure(
                context->GetNodeName(),
                "Communication group " + std::string(groupStr) +
                    " query succeeded with rank count=0; positive rank metadata is required for shape inference");
            return ge::GRAPH_FAILED;
        }
        rankSize = static_cast<int64_t>(rankNum);
#else
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "rank_size", std::to_string(*rankSizeAttr).c_str(),
            "rank_size must be positive when built-in communication-group lookup is unavailable");
        return ge::GRAPH_FAILED;
#endif
    } else {
        rankSize = *rankSizeAttr;
    }
    return ge::GRAPH_SUCCESS;
}

static void FillMatmulDims(CommParas& commParas, bool isTransA, bool isTransB)
{
    commParas.dimM = !isTransA ? commParas.x1MatrixShape->GetDim(0) : commParas.x1MatrixShape->GetDim(1);
    commParas.dimKX1 = !isTransA ? commParas.x1MatrixShape->GetDim(1) : commParas.x1MatrixShape->GetDim(0);
    commParas.dimKX2 = !isTransB ? commParas.x2MatrixShape->GetDim(0) : commParas.x2MatrixShape->GetDim(1);
    commParas.dimN = !isTransB ? commParas.x2MatrixShape->GetDim(1) : commParas.x2MatrixShape->GetDim(0);
}

static ge::graphStatus CheckKDimMatch(const gert::InferShapeContext* context, const CommParas& commParas)
{
    if (commParas.dimKX1 != commParas.dimKX2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x1, x2",
            (Ops::Base::ToString(*commParas.x1MatrixShape) + ", " + Ops::Base::ToString(*commParas.x2MatrixShape))
                .c_str(),
            ("K dimensions must match after transposition, actual x1.K=" + std::to_string(commParas.dimKX1) +
             ", x2.K=" + std::to_string(commParas.dimKX2))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// infershape 公共函数
ge::graphStatus CommonParamCheck(const gert::InferShapeContext* context, const size_t isTransAIndex,
                                 const size_t isTransBIndex, CommParas& commParas)
{
    if (CheckMatrixInputShapes(context, commParas) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                ReportMc2InferShapeFailure(context->GetNodeName(), "CommonParamCheck: attribute container is absent"),
                return ge::GRAPH_FAILED);
    const bool* isTransA = attrs->GetAttrPointer<bool>(isTransAIndex);
    const bool* isTransB = attrs->GetAttrPointer<bool>(isTransBIndex);
    const int64_t* rankSizeAttr = attrs->GetAttrPointer<int64_t>(RANK_SIZE);
    OP_CHECK_IF(rankSizeAttr == nullptr,
                ReportMc2InferShapeFailure(
                    context->GetNodeName(),
                    "CommonParamCheck: RuntimeAttrs is missing the rank_size slot, including its registered default"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(isTransA == nullptr,
                ReportMc2InferShapeFailure(
                    context->GetNodeName(),
                    "CommonParamCheck: RuntimeAttrs is missing the is_trans_a slot, including its registered default"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(isTransB == nullptr,
                ReportMc2InferShapeFailure(
                    context->GetNodeName(),
                    "CommonParamCheck: RuntimeAttrs is missing the is_trans_b slot, including its registered default"),
                return ge::GRAPH_FAILED);

    const char* groupStr = attrs->GetAttrPointer<char>(GROUP);
    if (groupStr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "group");
        return ge::GRAPH_FAILED;
    }
    if (ResolveRankSize(context, groupStr, rankSizeAttr, commParas.rankSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillMatmulDims(commParas, *isTransA, *isTransB);
    OP_LOGI(context->GetNodeName(),
            "group = %s isTransA %d isTransB %d x1.M = [%ld] x1.K = [%ld]"
            " x2.K = [%ld] x2.N = [%ld] rankSize = [%ld].",
            groupStr, (*isTransA), (*isTransB), commParas.x1MatrixShape->GetDim(0), commParas.x1MatrixShape->GetDim(1),
            commParas.x2MatrixShape->GetDim(0), commParas.x2MatrixShape->GetDim(1), commParas.rankSize);
    return CheckKDimMatch(context, commParas);
}

ge::graphStatus AllGatherMatmulInferYShape(gert::InferShapeContext* context, CommParas& commParas)
{
    OP_CHECK_IF(CommonParamCheck(context, AG_IS_TRANS_A, AG_IS_TRANS_B, commParas) != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CommonParamCheck execute failed."),
                return GRAPH_FAILED);
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    // 不支持k = 0
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x1",
                                              Ops::Base::ToString(*commParas.x1MatrixShape).c_str(),
                                              "K dimension must be nonzero after applying is_trans_a");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr,
                ReportMc2InferShapeFailure(context->GetNodeName(), "output y shape holder is absent at index 0"),
                return ge::GRAPH_FAILED);
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    yShape->SetDim(0, commParas.dimM * commParas.rankSize);
    yShape->SetDim(1, commParas.dimN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInferGatherOutShape(gert::InferShapeContext* context, const CommParas& commParas,
                                                   const size_t gatherIndex)
{
    if (context->GetAttrs() == nullptr) {
        ReportMc2InferShapeFailure(context->GetNodeName(), "InferGatherOutShape: attribute container is absent");
        return ge::GRAPH_FAILED;
    }
    const bool* isGatherOut = context->GetAttrs()->GetAttrPointer<bool>(gatherIndex);
    OP_CHECK_IF(
        isGatherOut == nullptr,
        ReportMc2InferShapeFailure(
            context->GetNodeName(),
            "InferGatherOutShape: RuntimeAttrs is missing the is_gather_out slot, including its registered default"),
        return ge::GRAPH_FAILED);
    gert::Shape* gatherOutShape = context->GetOutputShape(1);
    OP_CHECK_IF(
        gatherOutShape == nullptr,
        ReportMc2InferShapeFailure(context->GetNodeName(), "output gather_out shape holder is absent at index 1"),
        return ge::GRAPH_FAILED);
    if (*isGatherOut) {
        gatherOutShape->SetDimNum(SUPPORT_DIM_SIZE);
        gatherOutShape->SetDim(0, commParas.dimM * commParas.rankSize);
        gatherOutShape->SetDim(1, commParas.dimKX1);
    } else {
        gatherOutShape->SetDimNum(1);
        gatherOutShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulCommonInferShape(gert::InferShapeContext* context, const size_t gatherIndex)
{
    CommParas commParas;
    OP_CHECK_IF(AllGatherMatmulInferYShape(context, commParas) != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InferShapeAllGatherMatmul inferYshape execute failed."),
                return GRAPH_FAILED);

    OP_CHECK_IF(
        AllGatherMatmulInferGatherOutShape(context, commParas, gatherIndex) != GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InferShapeAllGatherMatmul inferGatherOutShape execute failed."),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferMatmulReduceScatterCommon(gert::InferShapeContext* context)
{
    CommParas commParas;
    OP_CHECK_IF(CommonParamCheck(context, RS_IS_TRANS_A, RS_IS_TRANS_B, commParas) != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CommonParamCheck execute failed."),
                return GRAPH_FAILED);
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x1",
                                              Ops::Base::ToString(*commParas.x1MatrixShape).c_str(),
                                              "K dimension must be nonzero after applying is_trans_a");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr,
                ReportMc2InferShapeFailure(context->GetNodeName(), "output y shape holder is absent at index 0"),
                return ge::GRAPH_FAILED);
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    yShape->SetDim(0, commParas.dimM / commParas.rankSize);
    yShape->SetDim(1, commParas.dimN);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferMatmulOutputDataType(gert::InferDataTypeContext* context, size_t inputIndex,
                                          size_t yDtypeAttrIndex, size_t outputIndex)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                ReportInferTypeMetadata(context->GetNodeName(),
                                        "InferMatmulOutputDataType: GetAttrs returned no RuntimeAttrs holder"),
                return ge::GRAPH_FAILED);
    const auto inputDtype = context->GetInputDataType(inputIndex);
    if ((inputDtype == ge::DataType::DT_FLOAT16) || (inputDtype == ge::DataType::DT_BF16)) {
        context->SetOutputDataType(outputIndex, inputDtype);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t* yDtype = attrs->GetInt(yDtypeAttrIndex);
    if (yDtype == nullptr || *yDtype == static_cast<int64_t>(ge::DataType::DT_UNDEFINED)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "y_dtype", yDtype == nullptr ? "null" : "DT_UNDEFINED",
            "Set y_dtype to a defined data type when the input is not FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(outputIndex, static_cast<ge::DataType>(*yDtype));
    return ge::GRAPH_SUCCESS;
}
} // namespace ops
