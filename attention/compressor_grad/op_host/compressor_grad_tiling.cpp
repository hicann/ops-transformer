/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file compressor_grad_tiling.cpp
 * \brief
 */

#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "compressor_grad_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

void ConvertRequiredParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    compressorGradContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorGradContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    compressorGradContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.dCmpKv.desc = context.GetRequiredInputDesc(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.dCmpKv.shape = context.GetRequiredInputShape(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.softmaxScore.desc = context.GetRequiredInputDesc(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.softmaxScore.shape = context.GetRequiredInputShape(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.kv.desc = context.GetRequiredInputDesc(KV_INPUT_INDEX);
    compressorGradContext.kv.shape = context.GetRequiredInputShape(KV_INPUT_INDEX);

    compressorGradContext.dX.desc = context.GetOutputDesc(D_X_OUTPUT_INDEX);
    compressorGradContext.dX.shape = context.GetOutputShape(D_X_OUTPUT_INDEX);
    compressorGradContext.dWkv.desc = context.GetOutputDesc(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWkv.shape = context.GetOutputShape(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWgate.desc = context.GetOutputDesc(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dWgate.shape = context.GetOutputShape(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dApe.desc = context.GetOutputDesc(D_APE_OUTPUT_INDEX);
    compressorGradContext.dApe.shape = context.GetOutputShape(D_APE_OUTPUT_INDEX);

    compressorGradContext.dtype = compressorGradContext.x.desc->GetDataType();
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_2) {
        compressorGradContext.layout = LayoutType::LAYOUT_TH;
    }
}

void ConvertOptionalParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus ConvertContext(gert::TilingContext &context, CompressorGradContext &compressorGradContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "opName", "got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    compressorGradContext.opName = context.GetNodeName();
    compressorGradContext.opType = context.GetNodeType();
    ConvertRequiredParams(context, compressorGradContext);
    ConvertOptionalParams(context, compressorGradContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "attrs", "got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.coff = attrs->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    compressorGradContext.cmpRatio = attrs->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);

    OP_CHECK_IF(
        context.GetWorkspaceSizes(1) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "workSpaceSize", "got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    compressorGradContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetBaseInfo(CompressorGradContext &compressorGradContext,
                            CompressorGradBaseParams &compressorGradBaseParams, uint32_t aicNum)
{
    if (compressorGradContext.x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradBaseParams.batchSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        compressorGradBaseParams.seqSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
        compressorGradBaseParams.hiddenSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_2);
        compressorGradBaseParams.tokenSize = compressorGradBaseParams.batchSize * compressorGradBaseParams.seqSize;
    } else {
        compressorGradBaseParams.batchSize =
            compressorGradContext.cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) - 1;
        compressorGradBaseParams.tokenSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        compressorGradBaseParams.hiddenSize =
            compressorGradContext.x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    }

    uint8_t coff =
        compressorGradContext.coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*compressorGradContext.coff);
    compressorGradBaseParams.headDim =
        compressorGradContext.wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) / coff;
    compressorGradBaseParams.featureDim =
        compressorGradContext.wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    compressorGradBaseParams.cmpRatio = static_cast<uint32_t>(*compressorGradContext.cmpRatio);
    compressorGradBaseParams.nSize = 2; // 预留（当前未参与 tiling 决策）
    compressorGradBaseParams.usedCoreNum = aicNum;
    OP_LOGI(compressorGradContext.opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u",
            compressorGradBaseParams.batchSize, compressorGradBaseParams.tokenSize, compressorGradBaseParams.cmpRatio,
            coff);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcWorkSpace(CompressorGradContext &compressorGradContext,
                              CompressorGradBaseParams &compressorGradBaseParams,
                              CompressorGradWorkspaceParams &compressorGradWorkspaceParams, size_t &workspaceSize,
                              size_t libapiSize, uint32_t aicNum)
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
    constexpr uint32_t M_BASE_SIZE = 128;

    uint8_t coff =
        compressorGradContext.coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*compressorGradContext.coff);
    uint32_t cmpRatio = compressorGradBaseParams.cmpRatio;
    uint32_t headDim = compressorGradBaseParams.headDim;
    uint32_t hiddenSize = compressorGradBaseParams.hiddenSize;
    uint32_t cubeCoreNum = aicNum;
    uint32_t groupSize = headDim / 128; // 与 kernel groupSize = headDim // D_BASE_SIZE 一致
    uint32_t groupNum = cubeCoreNum / groupSize;
    uint32_t cmpSize = coff * cmpRatio * headDim;
    uint32_t totalHeadDim = coff * headDim;
    uint32_t coffCoef = 2 / coff;
    uint32_t dealScNum = 128 / cmpRatio;
    uint32_t groupDealScNum = dealScNum * coffCoef;
    uint32_t groupRowStride = groupDealScNum * cmpRatio + (coff - 1) * cmpRatio; // 与 kernel 一致
    uint32_t dbRatio = 2;

    // 与 kernel 的 workspace 指针链逐分区对齐（元素数，FP32）:
    //   ape / dWkv / dWgate 单缓冲；dX / x / dXCache 按 dbRatio=2 双缓冲
    uint64_t apeWorkSpaceSize = static_cast<uint64_t>(groupNum) * cmpSize * coffCoef;
    uint64_t dXWorkSpaceSize = static_cast<uint64_t>(dbRatio) * cubeCoreNum * (M_BASE_SIZE * 2) * hiddenSize;
    uint64_t dWeightWorkSpaceSize = static_cast<uint64_t>(groupNum) * totalHeadDim * hiddenSize;
    // dWkv / dWGate 各占一份 dWeightWorkSpaceSize
    uint64_t xWorkSpaceSize = static_cast<uint64_t>(dbRatio) * groupNum * groupRowStride * hiddenSize * groupSize;
    uint64_t dXCacheWorkSpaceSize = static_cast<uint64_t>(dbRatio) * cmpRatio * hiddenSize;

    workspaceSize = libapiSize;
    workspaceSize +=
        (apeWorkSpaceSize + dXWorkSpaceSize + dWeightWorkSpaceSize * 2 + xWorkSpaceSize + dXCacheWorkSpaceSize) *
        MM1_RES_ELEM_SIZE;

    if (compressorGradContext.workSpaces) {
        compressorGradContext.workSpaces[0] = workspaceSize;
    }

    OP_LOGI(compressorGradContext.opName,
            "Tiling info: workspaceSize = %zu (ape=%llu dx=%llu dw=%llu x=%llu dxcache=%llu)", workspaceSize,
            apeWorkSpaceSize, dXWorkSpaceSize, dWeightWorkSpaceSize * 2, xWorkSpaceSize, dXCacheWorkSpaceSize);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string to_string(const T &value)
{
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

ge::graphStatus GenTilingKey(CompressorGradContext &compressorGradContext)
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;

    auto xDtype = compressorGradContext.x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    uint8_t coff =
        compressorGradContext.coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*compressorGradContext.coff);
    // 通过 ASCENDC 宏编码 tilingKey（force-include 的 codegen 生成头
    // CompressorGradTilingKey_tilingkey.h 声明位布局，与 PyPTO 一致：
    // Coff(2bit) | Layout(1bit) | DataType(2bit)，UINT 值自动映射为索引）
    compressorGradContext.tilingKey = GET_TPL_TILING_KEY(coff, layout, dtype);

    OP_LOGI(compressorGradContext.opName, "CompressorGrad dtype:%hhu layout:%hhu  coff:%hhu", dtype, layout, coff);
    OP_LOGI(compressorGradContext.opName, "CompressorGrad tilingKey:%lu", compressorGradContext.tilingKey);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                           const std::string subName, const char *opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, to_string(actualValue),
                                          subName + " only supports " + oss.str());
}

template <typename T>
ge::graphStatus CheckFeatureValueSupport(const T *featureValue, const std::vector<T> &expectFeatureValList,
                                         const std::string &name, const char *opName)
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value", opName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportInterval(const T *attrValue, const uint32_t minVal, const uint32_t maxVal,
                                              const std::string &name, const char *opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    std::string attr_value = "attr value";
    if (*attrValue < minVal || *attrValue > maxVal) {
        std::ostringstream oss;
        oss << "[" << minVal << ", " << maxVal << "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, std::to_string(*attrValue),
                                              "attr value only supports value in range " + oss.str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportList(const T *attrValue, const std::vector<T> &expectAttrValList,
                                          const std::string &name, const char *opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value", opName);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

namespace compressor_grad_tiling {
std::string LayoutTypeToStr(LayoutType layout)
{
    switch (layout) {
        case LayoutType::LAYOUT_BSH:
            return "BSH";
        case LayoutType::LAYOUT_TH:
            return "TH";
        default:
            return "UNKNOWN_LAYOUT";
    }
}
} // namespace compressor_grad_tiling

ge::graphStatus CheckDimNumInLayoutSupport(CompressorGradContext &compressorGradContext, const std::string &layout,
                                           const gert::StorageShape *shape, const std::string &name)
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    compressorGradContext.opName, name, std::to_string(shape->GetStorageShape().GetDimNum()),
                    "when layout is " + layout + ", dimension should be " + std::to_string(dimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                          const std::string &name, const char *opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE(opName, name, DataTypeToSerialString(actualDtype), oss.str());
}

ge::graphStatus CheckDtypeSupport(CompressorGradContext &compressorGradContext, const gert::CompileTimeTensorDesc *desc,
                                  const std::string &name)
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, name,
                                                     "datatype support list should be specify in DTYPE_SUPPORT_MAP"),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name, compressorGradContext.opName),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("CompressorGrad", "datatype", std::to_string(static_cast<int32_t>(type)),
                                              "not support");
        return "UNDEFINED";
    }
}

ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name, const char *opName)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, name,
                                                         "dim number support list should be specify in DIM_NUM_MAP"),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension", opName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaX(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.x.shape, X_NAME, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumInLayoutSupport(compressorGradContext,
                                       compressor_grad_tiling::LayoutTypeToStr(compressorGradContext.layout),
                                       compressorGradContext.x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaWkv(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.wkv.shape, WKV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaWgate(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.wgate.shape, WGATE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDCmpKv(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS !=
            CheckDtypeSupport(compressorGradContext, compressorGradContext.dCmpKv.desc, D_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.dCmpKv.shape, D_CMP_KV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaSoftmaxScore(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS !=
            CheckDtypeSupport(compressorGradContext, compressorGradContext.softmaxScore.desc, SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(compressorGradContext.softmaxScore.shape, SOFTMAX_SCORE_NAME,
                                                compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaKV(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.kv.desc, KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.kv.shape, KV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaCuSeqlens(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS !=
            CheckDtypeSupport(compressorGradContext, compressorGradContext.cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.cuSeqlens.shape, CU_SEQLENS_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaSeqused(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS !=
            CheckDtypeSupport(compressorGradContext, compressorGradContext.seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.seqUsed.shape, SEQUSED_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaStartPos(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS !=
            CheckDtypeSupport(compressorGradContext, compressorGradContext.startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.startPos.shape, START_POS_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDX(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.dX.desc, D_X_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.dX.shape, D_X_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDWkv(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.dWkv.desc, D_WKV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.dWkv.shape, D_WKV_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDWgate(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS !=
            CheckDtypeSupport(compressorGradContext, compressorGradContext.dWgate.desc, D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.dWgate.shape, D_WGATE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaDApe(CompressorGradContext &compressorGradContext)
{
    if (compressorGradContext.dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(compressorGradContext, compressorGradContext.dApe.desc, D_APE_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(compressorGradContext.dApe.shape, D_APE_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaCmpRatio(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportInterval(compressorGradContext.cmpRatio, MIN_CMP_RATIO, MAX_CMP_RATIO,
                                                           CMP_RATIO_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSingleParaCoff(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS !=
        CheckAttrValueSupportList(compressorGradContext.coff, COFF, COFF_NAME, compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckFeature(CompressorGradContext &compressorGradContext,
                             CompressorGradBaseParams &compressorGradBaseParams)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&compressorGradBaseParams.headDim, HEAD_DIM, "headDim",
                                                      compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(compressorGradBaseParams.hiddenSize > MAX_HIDDEN_SIZE ||
                    compressorGradBaseParams.hiddenSize < MIN_HIDDEN_SIZE ||
                    compressorGradBaseParams.hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(compressorGradContext.opName, "hiddenSize",
                                                      std::to_string(compressorGradBaseParams.hiddenSize),
                                                      "should be within [" + std::to_string(MIN_HIDDEN_SIZE) + ", " +
                                                          std::to_string(MAX_HIDDEN_SIZE) + "] and be 512-aligned"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
                                         const uint32_t &dimNum, const std::string &subName, const uint32_t &expectNum,
                                         const char *opName)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, name, "dim " + std::to_string(dimNum) + "=" + std::to_string(actualNum),
                    "should be equal to " + subName + ": " + std::to_string(expectNum)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeConsistency(CompressorGradContext &compressorGradContext,
                                      CompressorGradBaseParams &compressorGradBaseParams)
{
    uint8_t coff =
        compressorGradContext.coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*compressorGradContext.coff);
    auto coffD = compressorGradBaseParams.headDim * coff;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency(
                                 "cuSeqlens", compressorGradContext.cuSeqlens.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                 "batchSize+1", compressorGradBaseParams.batchSize + 1, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("seqUsed", compressorGradContext.seqUsed.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "batchSize", compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("startPos", compressorGradContext.startPos.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "batchSize", compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wkv", compressorGradContext.wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize",
                                     compressorGradBaseParams.hiddenSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wgate", compressorGradContext.wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_1,
                                     "hiddenSize", compressorGradBaseParams.hiddenSize, compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", compressorGradContext.wkv.shape,
                                                      COMPRESSOR_GRAD_DIM_INDEX_0, "coff*headDim",
                                                      static_cast<uint32_t>(coffD), compressorGradContext.opName) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", compressorGradContext.wgate.shape,
                                                      COMPRESSOR_GRAD_DIM_INDEX_0, "coff*headDim",
                                                      static_cast<uint32_t>(coffD), compressorGradContext.opName)) {
        return ge::GRAPH_FAILED;
    }

    if (compressorGradContext.x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        compressorGradContext.kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        if (ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("dCmpKv", compressorGradContext.dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_1,
                                         "headDim", compressorGradBaseParams.headDim, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
                                                          coff * compressorGradBaseParams.cmpRatio,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                                          compressorGradBaseParams.headDim,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", compressorGradContext.kv.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
                                                          coff * compressorGradBaseParams.cmpRatio,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (compressorGradContext.x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4 &&
        compressorGradContext.kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4) {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency(
                                     "dCmpKv", compressorGradContext.dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_0,
                                     "batchSize", compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("dCmpKv", compressorGradContext.dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_2,
                                         "headDim", compressorGradBaseParams.headDim, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                                          compressorGradBaseParams.batchSize,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
                                                          coff * compressorGradBaseParams.cmpRatio,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("softmaxScore", compressorGradContext.softmaxScore.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_3, "headDim",
                                                          compressorGradBaseParams.headDim,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
                                         compressorGradBaseParams.batchSize, compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kv", compressorGradContext.kv.shape,
                                                          COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
                                                          coff * compressorGradBaseParams.cmpRatio,
                                                          compressorGradContext.opName) ||
            ge::GRAPH_SUCCESS !=
                LogErrorShapeConsistency("kv", compressorGradContext.kv.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim",
                                         compressorGradBaseParams.headDim, compressorGradContext.opName)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc, const std::string &name,
                                       CompressorGradContext &compressorGradContext)
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != compressorGradContext.dtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    compressorGradContext.opName, name, DataTypeToSerialString(actualDtype),
                    "should be same with x: " + DataTypeToSerialString(compressorGradContext.dtype)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeConsistency(CompressorGradContext &compressorGradContext)
{
    if (CheckDtypeConsistencyX(compressorGradContext.wkv.desc, WKV_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.wgate.desc, WGATE_NAME, compressorGradContext) !=
            ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dCmpKv.desc, D_CMP_KV_NAME, compressorGradContext) !=
            ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dX.desc, D_X_NAME, compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dWkv.desc, D_WKV_NAME, compressorGradContext) !=
            ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(compressorGradContext.dWgate.desc, D_WGATE_NAME, compressorGradContext) !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDimNumConsistency(CompressorGradContext &compressorGradContext)
{
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != compressorGradContext.dX.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    compressorGradContext.opName, "d_x",
                    std::to_string(compressorGradContext.dX.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    compressorGradContext.opName, "d_cmp_kv",
                    std::to_string(compressorGradContext.dCmpKv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    compressorGradContext.opName, "softmax_score",
                    std::to_string(compressorGradContext.softmaxScore.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != compressorGradContext.kv.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    compressorGradContext.opName, "kv",
                    std::to_string(compressorGradContext.kv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckBlockDimConstrain(CompressorGradContext &compressorGradContext,
                                       CompressorGradBaseParams &compressorGradBaseParams, uint32_t &aicNum)
{
    uint32_t minBlockNum = compressorGradBaseParams.headDim / 128; // 128 is the largest dBaseSize
    OP_CHECK_IF(aicNum < minBlockNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(compressorGradContext.opName, "aicNum", std::to_string(aicNum),
                                                      "should not be less than " + std::to_string(minBlockNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMultiParaConsistency(CompressorGradContext &compressorGradContext,
                                          CompressorGradBaseParams &compressorGradBaseParams)
{
    if (CheckShapeConsistency(compressorGradContext, compressorGradBaseParams) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistency(compressorGradContext) != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSinglePara(CompressorGradContext &compressorGradContext)
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDCmpKv(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxScore(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDX(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWkv(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWgate(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaDApe(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio(compressorGradContext) ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff(compressorGradContext)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckRequiredInOutExistence(CompressorGradContext &context)
{
    OP_CHECK_IF(context.x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "x", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "x", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wkv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "wkv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wkv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "wkv", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wgate.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "wgate", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.wgate.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "wgate", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dCmpKv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_cmp_kv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dCmpKv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_cmp_kv", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.softmaxScore.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "softmax_score", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.softmaxScore.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "softmax_score", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.kv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "kv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.kv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "kv", "desc is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dX.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_x", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dX.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_x", "desc is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dWkv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_wkv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dWkv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_wkv", "desc is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dWgate.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_wgate", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dWgate.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_wgate", "desc is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context.dApe.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_ape", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.dApe.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "d_ape", "desc is nullptr"),
                return ge::GRAPH_FAILED);

    if (context.layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context.cuSeqlens.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "cu_seqlens",
                                                             "in TH layout, should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cuSeqlens.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "cu_seqlens",
                                                             "in TH layout, should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            context.cuSeqlens.desc != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "cu_seqlens", "in BSH layout, must be nullptr"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context.cuSeqlens.shape != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "cu_seqlens", "in BSH layout, must be nullptr"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckRequiredAttrExistence(CompressorGradContext &context)
{
    OP_CHECK_IF(context.cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.opName, "cmp_ratio", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckRequiredParaExistence(CompressorGradContext &compressorGradcontext)
{
    if (CheckRequiredInOutExistence(compressorGradcontext) != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence(compressorGradcontext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckEmptyTensor(CompressorGradContext &compressorGradcontext)
{
    // CompressorGrad 不支持空 tensor：与正向不同（正向 x 支持 B/S/T=0 走 EMPTY_X 分支），
    // 反向无空 tensor 分支——所有输入/输出 shapeSize 必须 > 0，空则直接拦截
    if (compressorGradcontext.x.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.dCmpKv.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.softmaxScore.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.kv.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.dX.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.dWkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.dWgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        compressorGradcontext.dApe.shape->GetStorageShape().GetShapeSize() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            compressorGradcontext.opName, "x", "0",
            "CompressorGrad does not support empty tensor: all inputs/outputs shapeSize must be > 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CompressorGradTilingFunc(gert::TilingContext *context)
{
    CompressorGradContext compressorGradContext{};
    CompressorGradBaseParams compressorGradBaseParams{};
    CompressorGradWorkspaceParams compressorGradWorkspaceParams{};
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    OP_CHECK_IF(context->GetPlatformInfo() == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(
        aicNum == 0 || aivNum == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "aicNum/aivNum", "num of core obtained is 0"),
        return ge::GRAPH_FAILED);

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    if (ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "context",
            "error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }

    compressorGradContext.blockDim = aicNum;
    if (CheckRequiredParaExistence(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckEmptyTensor(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckSinglePara(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetBaseInfo(compressorGradContext, compressorGradBaseParams, aicNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckFeature(compressorGradContext, compressorGradBaseParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckMultiParaConsistency(compressorGradContext, compressorGradBaseParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckBlockDimConstrain(compressorGradContext, compressorGradBaseParams, aicNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    size_t libapiSize = 0;
    size_t workspaceSize = 0;

    if (CalcWorkSpace(compressorGradContext, compressorGradBaseParams, compressorGradWorkspaceParams, workspaceSize,
                      libapiSize, aicNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GenTilingKey(compressorGradContext) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint8_t coff =
        compressorGradContext.coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*compressorGradContext.coff);
    CompressorGradTiling *tilingData = context->GetTilingData<CompressorGradTiling>();
    tilingData->batch_size = compressorGradBaseParams.batchSize;
    tilingData->token_size = compressorGradBaseParams.tokenSize;
    tilingData->seq_size = compressorGradBaseParams.seqSize;
    tilingData->cmp_ratio = compressorGradBaseParams.cmpRatio;
    tilingData->hidden_size = compressorGradBaseParams.hiddenSize;
    tilingData->head_dim = compressorGradBaseParams.headDim;
    // ── 核数（与 launch block_dim 一致；vec 每核 2 子核）──
    tilingData->cube_core_num = compressorGradContext.blockDim;
    tilingData->core_num = compressorGradContext.blockDim * 2;
    // ── shape 派生 ──
    tilingData->total_head_dim = coff * compressorGradBaseParams.headDim;
    tilingData->cmp_row_cnt = coff * compressorGradBaseParams.cmpRatio;
    tilingData->cmp_size = coff * compressorGradBaseParams.cmpRatio * compressorGradBaseParams.headDim;
    tilingData->cmp_kv_batch_stride =
        (compressorGradBaseParams.seqSize + compressorGradBaseParams.cmpRatio - 1) / compressorGradBaseParams.cmpRatio;
    if (compressorGradContext.layout == LayoutType::LAYOUT_BSH) {
        tilingData->x_rows = compressorGradBaseParams.batchSize * compressorGradBaseParams.seqSize;
        tilingData->cmp_kv_rows = compressorGradBaseParams.batchSize * tilingData->cmp_kv_batch_stride;
    } else {
        tilingData->x_rows = compressorGradBaseParams.tokenSize;
        tilingData->cmp_kv_rows = std::min(compressorGradBaseParams.tokenSize,
                                           compressorGradBaseParams.tokenSize / compressorGradBaseParams.cmpRatio +
                                               compressorGradBaseParams.batchSize);
    }
    // ── 分核派生 ──
    tilingData->group_size = compressorGradBaseParams.headDim / 128;
    tilingData->group_num = compressorGradContext.blockDim / tilingData->group_size;
    tilingData->cube_m_base_size = 128 * (2 / coff);
    // coff=1 时每 group 每轮只布置 2*dealScNum 块（保证子槽 dealScNum*cmpRatio <= 128，不超 L1/L0 物理行）
    tilingData->deal_sc_num = 128 / compressorGradBaseParams.cmpRatio;
    tilingData->group_deal_sc_num = tilingData->deal_sc_num * (2 / coff);
    tilingData->total_sc_num_per_round = tilingData->group_num * tilingData->group_deal_sc_num;
    // xArrangeGm 每 group 实际行数 = 数据行(gs 块 × cr) + coff=2 时 1 个 cr 头部（紧凑布局）
    tilingData->group_row_stride = tilingData->group_deal_sc_num * compressorGradBaseParams.cmpRatio +
                                   (coff - 1) * compressorGradBaseParams.cmpRatio;
    tilingData->db_row_cnt = tilingData->group_num * tilingData->group_row_stride;
    // ── 编译期派生（TilingKey 折叠值，与 kernel 内联算术恒等）──
    tilingData->coff_coef = 2 / coff;
    tilingData->d_deal_size = 128 / coff;
    tilingData->m_deal_size = 128 * coff;
    // ── workspace 分区（FP32 元素数；ape/dW 单缓冲，dX/x/dXCache 双缓冲 dbRatio=2）──
    tilingData->dape_ws_size = tilingData->group_num * tilingData->cmp_size * tilingData->coff_coef;
    tilingData->d_x_ws_size = 2 * tilingData->cube_core_num * 256 * compressorGradBaseParams.hiddenSize;
    tilingData->d_w_weight_ws_size =
        tilingData->group_num * tilingData->total_head_dim * compressorGradBaseParams.hiddenSize;
    tilingData->x_ws_size = 2 * tilingData->group_num * tilingData->group_row_stride *
                            compressorGradBaseParams.hiddenSize * tilingData->group_size;
    tilingData->d_x_cache_ws_size = 2 * compressorGradBaseParams.cmpRatio * compressorGradBaseParams.hiddenSize;

    context->SetTilingKey(compressorGradContext.tilingKey);
    context->SetBlockDim(compressorGradContext.blockDim);
    OP_LOGI(compressorGradContext.opName, "block dim: %u.", compressorGradContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCompressorGrad(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorGrad)
    .Tiling(CompressorGradTilingFunc)
    .TilingParse<CompressorGradCompileInfo>(TilingParseForCompressorGrad);
} // namespace optiling
