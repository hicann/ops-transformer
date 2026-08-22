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
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
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

bool IsPlatform910B(const char *nodeName)
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

static ge::graphStatus InferExpertIdsShape(gert::InferShapeContext *context, int64_t &k, int64_t &h,
                                           int64_t &globalBsReal, int64_t epWorldSize)
{
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);

    gert::Shape *expandIdxShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdxShape);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);
    const auto tpRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRankId);
    int64_t bs = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    h = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);
    int64_t bsTmp = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(0);
    k = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(1);
    globalBsReal = (*globalBs == 0) ? (bs * epWorldSize) : *globalBs;
    OP_CHECK_IF(
        globalBsReal < 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "globalBsReal", std::to_string(globalBsReal).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (bs <= 0) || (h <= 0) || (bsTmp <= 0) || (k <= 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "xShape/expertIdsShape",
                                              (std::string("[") + std::to_string(bs) + ", " + std::to_string(h) +
                                               "], [" + std::to_string(bsTmp) + ", " + std::to_string(k) + "]")
                                                  .c_str(),
                                              "all dims must be > 0"),
        return ge::GRAPH_FAILED);

    expandIdxShape->SetDimNum(DIM_ONE);
    expandIdxShape->SetDim(0U, bs * k);
    OP_LOGD(context->GetNodeName(), "expandIdxShape shape is :%s after infershape.",
            Ops::Base::ToString(*expandIdxShape).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferExpandXAndScalesShape(gert::InferShapeContext *context, int64_t k, int64_t h,
                                                  int64_t &localExpertNum, int64_t globalBsReal, int64_t epWorldSize)
{
    gert::Shape *expandXShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    gert::Shape *dynamicScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicScalesShape);
    gert::Shape *expandScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_SCALES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandScalesShape);
    const gert::Shape *expertScalesShape = context->GetOptionalInputShape(DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);
    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);
    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);
    const auto expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);
    OP_CHECK_IF((*sharedExpertRankNum < 0) || (*sharedExpertRankNum >= epWorldSize),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "sharedExpertRankNum",
                                          std::to_string(*sharedExpertRankNum).c_str(),
                                          (std::string("[0, ") + std::to_string(epWorldSize) + ")").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((*epRankId < 0) || (*epRankId >= epWorldSize),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "epRankId", std::to_string(*epRankId).c_str(),
                                          (std::string("[0, ") + std::to_string(epWorldSize) + ")").c_str()),
                return ge::GRAPH_FAILED);
    int64_t moeRankNum = epWorldSize - *sharedExpertRankNum;
    OP_CHECK_IF(
        moeRankNum <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "moeRankNum", std::to_string(moeRankNum).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    int64_t localMoeExpertNum = *moeExpertNum / moeRankNum;
    bool isSharedExpert = (*expertShardType == 0) ? (*epRankId < *sharedExpertRankNum) :
                                                    (*epRankId >= (epWorldSize - *sharedExpertRankNum));
    localExpertNum = isSharedExpert ? 1 : localMoeExpertNum;
    int64_t a = isSharedExpert ? (globalBsReal / *sharedExpertRankNum) : (globalBsReal * std::min(localExpertNum, k));
    auto realA = a;
    expandXShape->SetDimNum(DIM_TWO);
    expandXShape->SetDim(0U, realA);
    expandXShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "expandXShape is:%s after infershape.", Ops::Base::ToString(*expandXShape).c_str());
    dynamicScalesShape->SetDimNum(DIM_ONE);
    dynamicScalesShape->SetDim(0U, realA);
    OP_LOGD(context->GetNodeName(), "dynamicScalesShape shape is:%s after infershape.",
            Ops::Base::ToString(*dynamicScalesShape).c_str());
    expandScalesShape->SetDimNum(DIM_ONE);
    expandScalesShape->SetDim(0U, 0);
    if (expertScalesShape != nullptr) {
        expandScalesShape->SetDim(0U, a);
    }
    OP_LOGD(context->GetNodeName(), "expandScalesShape shape is:%s after infershape.",
            Ops::Base::ToString(*expandScalesShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMoeDistributeDispatch(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatch.");
    gert::Shape *expertTokenNumsShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);
    gert::Shape *epRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRecvCountShape);
    gert::Shape *tpRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRecvCountShape);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);
    const auto tpWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpWorldSize);
    int64_t k;
    int64_t h;
    int64_t localExpertNum;
    int64_t globalBsReal;
    OP_CHECK_IF(InferExpertIdsShape(context, k, h, globalBsReal, *epWorldSize) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InferExpertIdsShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        InferExpandXAndScalesShape(context, k, h, localExpertNum, globalBsReal, *epWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "InferExpandXAndScalesShape failed."), return ge::GRAPH_FAILED);
    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, localExpertNum);
    OP_LOGD(context->GetNodeName(), "expertTokenNumsShape shape is :%s after infershape.",
            Ops::Base::ToString(*expertTokenNumsShape).c_str());
    epRecvCountShape->SetDimNum(DIM_ONE);
    if (IsPlatform910B(context->GetNodeName())) {
        // 2：globalbs * 2kn memory size, to support different bs in ranks
        int64_t shapeDim = *epWorldSize * localExpertNum + globalBsReal * 2 * k * (*epWorldSize) / RANK_NUM_PER_NODE;
        epRecvCountShape->SetDim(0U, shapeDim);
    } else {
        epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum);
    }
    OP_LOGD(context->GetNodeName(), "epRecvCountShape shape is :%s after infershape.",
            Ops::Base::ToString(*epRecvCountShape).c_str());
    tpRecvCountShape->SetDimNum(DIM_ONE);
    tpRecvCountShape->SetDim(0U, *tpWorldSize);
    OP_LOGD(context->GetNodeName(), "tpRecvCountShape shape is :%s after infershape.",
            Ops::Base::ToString(*tpRecvCountShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeDispatch(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatch.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
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
