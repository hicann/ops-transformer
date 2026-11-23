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
 * \file moe_distribute_dispatch_infer.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include <string>
#include <vector>
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "op_host/util/op_const_def.h"

using namespace ge;
namespace ops {
namespace {
void ReportInferTypeMetadata(const char* opName, const char* detail)
{
    const std::string reason =
        std::string(detail) + "; actual=null, expected=materialized inference metadata. Rebuild the inference context";
    OP_LOGE(opName, "%s: %s", opName, reason.c_str());
}

void ReportDispatchInferMetadata(const char* opName, const char* stage, const char* detail)
{
    const char* name = opName == nullptr ? "MoeDistributeDispatch" : opName;
    const std::string reason =
        std::string(stage) + ": " + detail + "; rebuild the inference context with the required metadata";
    OP_LOGE(name, "%s: %s", name, reason.c_str());
}
} // namespace

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t RANK_NUM_PER_NODE = 8;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDX_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_SCALES_IDX_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_IDX_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX = 5;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_SCALES = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 10;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 11;

static constexpr uint32_t SOC_VERSION_SIZE = 32;

bool IsPlatform910B(const char* nodeName)
{
    char versionValVersion[SOC_VERSION_SIZE];
    // rtGetSocSpec获取成功返回值是0，获取失败返回非0
    if (rtGetSocSpec("version", "Short_SoC_version", versionValVersion, SOC_VERSION_SIZE) != RT_ERROR_NONE) {
        OPS_LOG_E(nodeName, "Cannot get Short_SoC_version info in infershape!");
        return false;
    }
    static std::set<std::string> supportedSoc = {"Ascend910B"};
    OPS_LOG_D(nodeName, "Get Short_SoC_version %s", versionValVersion);
    return (supportedSoc.count(versionValVersion) > 0);
}

static ge::graphStatus InferGlobalBsAndExpandIdxShape(gert::InferShapeContext* context, int64_t& k, int64_t& h,
                                                      int64_t& globalBsReal, int64_t epWorldSize)
{
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    const gert::Shape* expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDX_INDEX);
    OP_CHECK_IF(expertIdsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expert_ids"),
                return ge::GRAPH_FAILED);

    gert::Shape* expandIdxShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_IDX_INDEX);
    OP_CHECK_IF(
        expandIdxShape == nullptr,
        ReportDispatchInferMetadata(context->GetNodeName(), __func__, "output expand_idx shape holder is absent"),
        return ge::GRAPH_FAILED);

    const auto attrs = context->GetAttrs();
    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OP_CHECK_IF(globalBs == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__,
                                            "defaulted attribute global_bs is absent from RuntimeAttrs"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *globalBs < 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "globalBs", std::to_string(*globalBs).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape->GetDimNum() != 2U,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "xShape",
                                             std::to_string(xShape->GetDimNum()).c_str(), "= 2"),
                return ge::GRAPH_FAILED);
    int64_t bs = xShape->GetDim(0);
    h = xShape->GetDim(1);
    globalBsReal = (*globalBs == 0) ? (bs * epWorldSize) : *globalBs;

    OP_CHECK_IF(expertIdsShape->GetDimNum() != 2U,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expertIdsShape",
                                             std::to_string(expertIdsShape->GetDimNum()).c_str(), "= 2"),
                return ge::GRAPH_FAILED);
    k = expertIdsShape->GetDim(1);

    OP_CHECK_IF(expertIdsShape->GetDim(0) != bs,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "xShape/expertIdsShape",
                    (std::string("batch size dim of expertIdsShape: ") + std::to_string(expertIdsShape->GetDim(0)) +
                     std::string(", batch size of xShape: ") + std::to_string(bs))
                        .c_str(),
                    "batch size dim of xShape and expertIdsShape should be equal"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (bs <= 0) || (h <= 0) || (k <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "xShape/expertIdsShape",
                                              (std::string("[") + std::to_string(bs) + ", " + std::to_string(h) +
                                               "], [" + std::to_string(bs) + ", " + std::to_string(k) + "]")
                                                  .c_str(),
                                              "all dims must be > 0"),
        return ge::GRAPH_FAILED);

    expandIdxShape->SetDimNum(DIM_ONE);
    expandIdxShape->SetDim(0U, bs * k);
    OP_LOGD(context->GetNodeName(), "expandIdxShape shape is :%s after infershape.",
            Ops::Base::ToString(*expandIdxShape).c_str());

    return ge::GRAPH_SUCCESS;
}

struct ExpandShapeOutputs {
    gert::Shape* expandX;
    gert::Shape* dynamicScales;
    gert::Shape* expandScales;
};

static ge::graphStatus GetExpandShapeOutputs(gert::InferShapeContext* context, ExpandShapeOutputs& outputs)
{
    outputs.expandX = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OP_CHECK_IF(outputs.expandX == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__, "output expand_x shape holder is absent"),
                return ge::GRAPH_FAILED);
    outputs.dynamicScales = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_CHECK_IF(
        outputs.dynamicScales == nullptr,
        ReportDispatchInferMetadata(context->GetNodeName(), __func__, "output dynamic_scales shape holder is absent"),
        return ge::GRAPH_FAILED);
    outputs.expandScales = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_SCALES);
    OP_CHECK_IF(
        outputs.expandScales == nullptr,
        ReportDispatchInferMetadata(context->GetNodeName(), __func__, "output expand_scales shape holder is absent"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct ExpandShapeAttrs {
    const gert::Shape* expertScales;
    const int64_t* moeExpertNum;
    const int64_t* sharedExpertRankNum;
    const int64_t* epRankId;
    const int64_t* expertShardType;
};

static ge::graphStatus GetExpandShapeAttrs(gert::InferShapeContext* context, int64_t epWorldSize,
                                           ExpandShapeAttrs& attrsOut)
{
    const auto attrs = context->GetAttrs();
    attrsOut.expertScales = context->GetOptionalInputShape(DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX);
    attrsOut.moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    attrsOut.sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    attrsOut.epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    attrsOut.expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OP_CHECK_IF(
        attrsOut.epRankId == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "ep_rank_id", "required attribute is absent"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrsOut.moeExpertNum == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "moe_expert_num",
                                                         "required attribute is absent"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrsOut.expertShardType == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__,
                                            "defaulted attribute expert_shard_type is absent from RuntimeAttrs"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrsOut.sharedExpertRankNum == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__,
                                            "defaulted attribute shared_expert_rank_num is absent from RuntimeAttrs"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((*attrsOut.sharedExpertRankNum < 0) || (*attrsOut.sharedExpertRankNum >= epWorldSize),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "sharedExpertRankNum",
                                          std::to_string(*attrsOut.sharedExpertRankNum).c_str(),
                                          (std::string("[0, ") + std::to_string(epWorldSize) + ")").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (*attrsOut.epRankId < 0) || (*attrsOut.epRankId >= epWorldSize),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "epRankId", std::to_string(*attrsOut.epRankId).c_str(),
                                  (std::string("[0, ") + std::to_string(epWorldSize) + ")").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*attrsOut.expertShardType != 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "expertShardType",
                                          std::to_string(*attrsOut.expertShardType).c_str(), "= 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetExpandShapeOutputs(gert::InferShapeContext* context, const ExpandShapeOutputs& outputs, int64_t h,
                                  int64_t realA, int64_t a, bool hasExpertScales)
{
    outputs.expandX->SetDimNum(DIM_TWO);
    outputs.expandX->SetDim(0U, realA);
    outputs.expandX->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "expandXShape is:%s after infershape.",
            Ops::Base::ToString(*outputs.expandX).c_str());
    outputs.dynamicScales->SetDimNum(DIM_ONE);
    outputs.dynamicScales->SetDim(0U, realA);
    OP_LOGD(context->GetNodeName(), "dynamicScalesShape shape is:%s after infershape.",
            Ops::Base::ToString(*outputs.dynamicScales).c_str());
    outputs.expandScales->SetDimNum(DIM_ONE);
    outputs.expandScales->SetDim(0U, hasExpertScales ? a : 0);
    OP_LOGD(context->GetNodeName(), "expandScalesShape shape is:%s after infershape.",
            Ops::Base::ToString(*outputs.expandScales).c_str());
}

static ge::graphStatus InferExpandXAndScalesShape(gert::InferShapeContext* context, int64_t k, int64_t h,
                                                  int64_t& localExpertNum, int64_t globalBsReal, int64_t epWorldSize)
{
    ExpandShapeOutputs outputs{};
    OP_CHECK_IF(GetExpandShapeOutputs(context, outputs) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GetExpandShapeOutputs failed."),
                return ge::GRAPH_FAILED);
    ExpandShapeAttrs attrs{};
    OP_CHECK_IF(GetExpandShapeAttrs(context, epWorldSize, attrs) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GetExpandShapeAttrs failed."), return ge::GRAPH_FAILED);
    int64_t moeRankNum = epWorldSize - *attrs.sharedExpertRankNum;
    OP_CHECK_IF(
        moeRankNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "moeRankNum", std::to_string(moeRankNum).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    int64_t localMoeExpertNum = *attrs.moeExpertNum / moeRankNum;
    bool isSharedExpert = (*attrs.epRankId < *attrs.sharedExpertRankNum);
    localExpertNum = isSharedExpert ? 1 : localMoeExpertNum;
    int64_t a =
        isSharedExpert ? (globalBsReal / *attrs.sharedExpertRankNum) : (globalBsReal * std::min(localExpertNum, k));
    SetExpandShapeOutputs(context, outputs, h, a, a, attrs.expertScales != nullptr);
    return ge::GRAPH_SUCCESS;
}

struct DispatchInferOutputs {
    gert::Shape* expertTokenNums;
    gert::Shape* epRecvCounts;
    gert::Shape* tpRecvCounts;
};

static ge::graphStatus GetDispatchInferOutputs(gert::InferShapeContext* context, DispatchInferOutputs& outputs)
{
    outputs.expertTokenNums = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_CHECK_IF(outputs.expertTokenNums == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__,
                                            "output expert_token_nums shape holder is absent"),
                return ge::GRAPH_FAILED);
    outputs.epRecvCounts = context->GetOutputShape(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_CHECK_IF(
        outputs.epRecvCounts == nullptr,
        ReportDispatchInferMetadata(context->GetNodeName(), __func__, "output ep_recv_counts shape holder is absent"),
        return ge::GRAPH_FAILED);
    outputs.tpRecvCounts = context->GetOutputShape(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_CHECK_IF(
        outputs.tpRecvCounts == nullptr,
        ReportDispatchInferMetadata(context->GetNodeName(), __func__, "output tp_recv_counts shape holder is absent"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetDispatchInferOutputs(gert::InferShapeContext* context, const DispatchInferOutputs& outputs, int64_t k,
                                    int64_t localExpertNum, int64_t globalBsReal, int64_t epWorldSize,
                                    int64_t tpWorldSize)
{
    outputs.expertTokenNums->SetDimNum(DIM_ONE);
    outputs.expertTokenNums->SetDim(0U, localExpertNum);
    OP_LOGD(context->GetNodeName(), "expertTokenNumsShape shape is :%s after infershape.",
            Ops::Base::ToString(*outputs.expertTokenNums).c_str());
    outputs.epRecvCounts->SetDimNum(DIM_ONE);
    if (IsPlatform910B(context->GetNodeName())) {
        // 910B reserves 2 * k * globalBsReal elements for receive-count capacity,
        // allowing ranks with different batch sizes to share the receive buffer.
        int64_t shapeDim = epWorldSize * localExpertNum + globalBsReal * 2 * k * epWorldSize / RANK_NUM_PER_NODE;
        outputs.epRecvCounts->SetDim(0U, shapeDim);
    } else {
        outputs.epRecvCounts->SetDim(0U, epWorldSize * localExpertNum);
    }
    OP_LOGD(context->GetNodeName(), "epRecvCountShape shape is :%s after infershape.",
            Ops::Base::ToString(*outputs.epRecvCounts).c_str());
    outputs.tpRecvCounts->SetDimNum(DIM_ONE);
    outputs.tpRecvCounts->SetDim(0U, tpWorldSize);
    OP_LOGD(context->GetNodeName(), "tpRecvCountShape shape is :%s after infershape.",
            Ops::Base::ToString(*outputs.tpRecvCounts).c_str());
}

static ge::graphStatus InferShapeMoeDistributeDispatch(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatch.");
    DispatchInferOutputs outputs{};
    OP_CHECK_IF(GetDispatchInferOutputs(context, outputs) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GetDispatchInferOutputs failed."),
                return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__, "RuntimeAttrs holder is absent"),
                return ge::GRAPH_FAILED);
    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OP_CHECK_IF(epWorldSize == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "ep_world_size",
                                                         "required attribute is absent"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *epWorldSize <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "epWorldSize", std::to_string(*epWorldSize).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    const auto tpWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX);
    OP_CHECK_IF(tpWorldSize == nullptr,
                ReportDispatchInferMetadata(context->GetNodeName(), __func__,
                                            "defaulted attribute tp_world_size is absent from RuntimeAttrs"),
                return ge::GRAPH_FAILED);
    int64_t k;
    int64_t h;
    int64_t localExpertNum;
    int64_t globalBsReal;
    OP_CHECK_IF(InferGlobalBsAndExpandIdxShape(context, k, h, globalBsReal, *epWorldSize) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InferGlobalBsAndExpandIdxShape failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        InferExpandXAndScalesShape(context, k, h, localExpertNum, globalBsReal, *epWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InferExpandXAndScalesShape failed."), return ge::GRAPH_FAILED);
    SetDispatchInferOutputs(context, outputs, k, localExpertNum, globalBsReal, *epWorldSize, *tpWorldSize);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}

// InferShape and InferDataType are registered together for this operator; keep both callbacks in this TU
// to preserve the existing host registration layout.
static ge::graphStatus InferDataTypeMoeDistributeDispatch(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatch.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportInferTypeMetadata(context->GetNodeName(), "InferDataType: GetAttrs returned no RuntimeAttrs holder"),
        return ge::GRAPH_FAILED);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OP_CHECK_IF(quantMode == nullptr,
                ReportInferTypeMetadata(context->GetNodeName(),
                                        "InferDataType: GetAttrPointer for defaulted quant_mode returned no value"),
                return ge::GRAPH_FAILED);
    const auto scalesType = context->GetOptionalInputDataType(DISPATCH_INPUT_SCALES_IDX_INDEX);
    bool quantFlag = (scalesType != ge::DT_UNDEFINED) ? true : false;
    OP_LOGD(context->GetNodeName(), "quantFlag is %d.", quantFlag);
    if (quantFlag || (*quantMode != 0)) {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, ge::DT_INT8);
    } else {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, xDtype);
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT64);
    context->SetOutputDataType(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeDispatch)
    .InferShape(InferShapeMoeDistributeDispatch)
    .InferDataType(InferDataTypeMoeDistributeDispatch);
} // namespace ops
