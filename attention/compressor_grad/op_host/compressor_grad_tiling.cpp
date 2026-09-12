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
namespace {

template <typename T>
std::string to_string(const T &value)
{
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
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

static std::string DataTypeToSerialString(ge::DataType type);

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

struct ShapeCheckItem {
    const char *tensorName;
    const gert::StorageShape *shape;
    uint32_t dimIdx;
    const char *expectName;
    uint32_t expectValue;
};

static ge::graphStatus CheckShapeList(const char *opName, const ShapeCheckItem *items, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (LogErrorShapeConsistency(items[i].tensorName, items[i].shape, items[i].dimIdx, items[i].expectName,
                                     items[i].expectValue, opName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ── 辅助函数：检查单个 tensor 的 shape 和 desc 均不为空 ──
static ge::graphStatus CheckTensorShapeAndDesc(const char *opName, const char *tensorName,
                                               const gert::CompileTimeTensorDesc *desc, const gert::StorageShape *shape)
{
    OP_CHECK_IF(shape == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorName, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(desc == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorName, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct TensorCheckItem {
    const char *name;
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

static ge::graphStatus CheckTensorList(const char *opName, const TensorCheckItem *items, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (CheckTensorShapeAndDesc(opName, items[i].name, items[i].desc, items[i].shape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus CompressorGradTilingImpl::ConvertRequiredParams(gert::TilingContext &context,
                                                                CompressorGradContext &compressorGradContext)
{
    compressorGradContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorGradContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OP_CHECK_IF(compressorGradContext.x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, X_NAME, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compressorGradContext.x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, X_NAME, "desc is nullptr"),
                return ge::GRAPH_FAILED);
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
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(compressorGradContext.opName, X_NAME, std::to_string(xDimNum),
                                                 "x dimension should be 2 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorGradTilingImpl::ConvertOptionalParams(gert::TilingContext &context,
                                                     CompressorGradContext &compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorGradTilingImpl::ConvertContext(gert::TilingContext &context,
                                                         CompressorGradContext &compressorGradContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "opName", "got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    compressorGradContext.opName = context.GetNodeName();
    compressorGradContext.opType = context.GetNodeType();
    compressorGradContext.platformInfo = context.GetPlatformInfo();
    OP_CHECK_IF(ConvertRequiredParams(context, compressorGradContext) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
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

ge::graphStatus CompressorGradTilingImpl::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(
        aicNum_ == 0 || aivNum_ == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "aicNum/aivNum", "num of core obtained is 0"),
        return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::SetBaseInfo()
{
    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        baseParams_.batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        baseParams_.seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
        baseParams_.hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_2);
        baseParams_.tokenSize = baseParams_.batchSize * baseParams_.seqSize;
    } else {
        baseParams_.batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) - 1;
        baseParams_.tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        baseParams_.hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    }

    coff_ = context_->coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*context_->coff);
    baseParams_.headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) / coff_;
    baseParams_.featureDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    baseParams_.cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_.nSize = 2; // 预留（当前未参与 tiling 决策）
    baseParams_.usedCoreNum = aicNum_;
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u", baseParams_.batchSize,
            baseParams_.tokenSize, baseParams_.cmpRatio, coff_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CalcWorkSpace()
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32（sizeof(float)）

    uint8_t coff = coff_;
    uint32_t cmpRatio = baseParams_.cmpRatio;
    uint32_t headDim = baseParams_.headDim;
    uint32_t hiddenSize = baseParams_.hiddenSize;
    uint32_t cubeCoreNum = aicNum_;
    uint32_t groupSize = headDim / D_BASE_SIZE; // 与 kernel groupSize = headDim // D_BASE_SIZE 一致
    uint32_t groupNum = cubeCoreNum / groupSize;
    uint32_t cmpSize = coff * cmpRatio * headDim;
    uint32_t totalHeadDim = coff * headDim;
    uint32_t coffCoef = COFF_MAX / coff;
    uint32_t dealScNum = D_BASE_SIZE / cmpRatio;
    uint32_t groupDealScNum = dealScNum * coffCoef;
    uint32_t groupRowStride = groupDealScNum * cmpRatio + (coff - 1) * cmpRatio; // 与 kernel 一致
    uint32_t dbRatio = DB_RATIO;

    // 与 kernel 的 workspace 指针链逐分区对齐（元素数，FP32）:
    //   ape / dWkv / dWgate 单缓冲；dX / x / dXCache 按 dbRatio=2 双缓冲
    uint64_t apeWorkSpaceSize = static_cast<uint64_t>(groupNum) * cmpSize * coffCoef;
    uint64_t dXWorkSpaceSize = static_cast<uint64_t>(dbRatio) * cubeCoreNum * (M_BASE_SIZE * COFF_MAX) * hiddenSize;
    uint64_t dWeightWorkSpaceSize = static_cast<uint64_t>(groupNum) * totalHeadDim * hiddenSize;
    // dWkv / dWGate 各占一份 dWeightWorkSpaceSize
    uint64_t xWorkSpaceSize = static_cast<uint64_t>(dbRatio) * groupNum * groupRowStride * hiddenSize * groupSize;
    uint64_t dXCacheWorkSpaceSize = static_cast<uint64_t>(dbRatio) * cmpRatio * hiddenSize;

    workspaceSize_ = libapiSize_;
    workspaceSize_ +=
        (apeWorkSpaceSize + dXWorkSpaceSize + dWeightWorkSpaceSize * 2 + xWorkSpaceSize + dXCacheWorkSpaceSize) *
        MM1_RES_ELEM_SIZE;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize = %zu (ape=%llu dx=%llu dw=%llu x=%llu dxcache=%llu)",
            workspaceSize_, apeWorkSpaceSize, dXWorkSpaceSize, dWeightWorkSpaceSize * 2, xWorkSpaceSize,
            dXCacheWorkSpaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::GenTilingKey() const
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;

    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    uint8_t coff = coff_;
    // 通过 ASCENDC 宏编码 tilingKey（force-include 的 codegen 生成头
    // CompressorGradTilingKey_tilingkey.h 声明位布局，与 PyPTO 一致：
    // Coff(2bit) | Layout(1bit) | DataType(2bit)，UINT 值自动映射为索引）
    context_->tilingKey = GET_TPL_TILING_KEY(coff, layout, dtype);

    OP_LOGI(context_->opName, "CompressorGrad dtype:%hhu layout:%hhu  coff:%hhu", dtype, layout, coff);
    OP_LOGI(context_->opName, "CompressorGrad tilingKey:%lu", context_->tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckDimNumInLayoutSupport(const std::string &layout,
                                                                     const gert::StorageShape *shape,
                                                                     const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, name, std::to_string(shape->GetStorageShape().GetDimNum()),
                    "when layout is " + layout + ", dimension should be " + std::to_string(dimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                            const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context_->opName, name, "datatype support list should be specify in DTYPE_SUPPORT_MAP"),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name, context_->opName),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckDimNumSupport(const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, name,
                                                         "dim number support list should be specify in DIM_NUM_MAP"),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension", context_->opName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaX() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(compressor_grad_tiling::LayoutTypeToStr(context_->layout),
                                                        context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaDCmpKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dCmpKv.desc, D_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dCmpKv.shape, D_CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaSoftmaxScore() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->softmaxScore.desc, SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->softmaxScore.shape, SOFTMAX_SCORE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaKV() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kv.desc, KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kv.shape, KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaSeqused() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaDX() const
{
    if (context_->dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dX.desc, D_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dX.shape, D_X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaDWkv() const
{
    if (context_->dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWkv.desc, D_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWkv.shape, D_WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaDWgate() const
{
    if (context_->dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWgate.desc, D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWgate.shape, D_WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaDApe() const
{
    if (context_->dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dApe.desc, D_APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dApe.shape, D_APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaCmpRatio() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportInterval(context_->cmpRatio, MIN_CMP_RATIO, MAX_CMP_RATIO,
                                                           CMP_RATIO_NAME, context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSingleParaCoff() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportList(context_->coff, COFF, COFF_NAME, context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_.headDim, HEAD_DIM, "headDim", context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseParams_.hiddenSize > MAX_HIDDEN_SIZE || baseParams_.hiddenSize < MIN_HIDDEN_SIZE ||
            baseParams_.hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "hiddenSize", std::to_string(baseParams_.hiddenSize),
                                              "should be within [" + std::to_string(MIN_HIDDEN_SIZE) + ", " +
                                                  std::to_string(MAX_HIDDEN_SIZE) + "] and be 512-aligned"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::LogErrorShapeConsistency(const std::string &name,
                                                                   const gert::StorageShape *shape,
                                                                   const uint32_t &dimNum, const std::string &subName,
                                                                   const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, name, "dim " + std::to_string(dimNum) + "=" + std::to_string(actualNum),
                    "should be equal to " + subName + ": " + std::to_string(expectNum)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckShapeConsistency() const
{
    uint8_t coff = coff_;
    auto coffD = baseParams_.headDim * coff;
    const char *opName = context_->opName;
    // 公共部分：cuSeqlens/seqUsed/startPos + wkv/wgate 的 hiddenSize 和 coff*headDim
    const ShapeCheckItem commonChecks[] = {
        {"cuSeqlens", context_->cuSeqlens.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize+1", baseParams_.batchSize + 1},
        {"seqUsed", context_->seqUsed.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"startPos", context_->startPos.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"wkv", context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize", baseParams_.hiddenSize},
        {"wgate", context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize", baseParams_.hiddenSize},
        {"wkv", context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(coffD)},
        {"wgate", context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(coffD)},
    };
    if (CheckShapeList(opName, commonChecks, sizeof(commonChecks) / sizeof(commonChecks[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        context_->dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_2 &&
        context_->softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        context_->kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3) {
        const ShapeCheckItem checks2D[] = {
            {"dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "headDim", baseParams_.headDim},
            {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
             coff * baseParams_.cmpRatio},
            {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim", baseParams_.headDim},
            {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio", coff * baseParams_.cmpRatio},
            {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim", baseParams_.headDim},
        };
        if (CheckShapeList(opName, checks2D, sizeof(checks2D) / sizeof(checks2D[0])) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        context_->dCmpKv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_3 &&
        context_->softmaxScore.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4 &&
        context_->kv.shape->GetStorageShape().GetDimNum() == COMPRESSOR_GRAD_DIM_NUM_4) {
        const ShapeCheckItem checks3D[] = {
            {"dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
            {"dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim", baseParams_.headDim},
            {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
             baseParams_.batchSize},
            {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
             coff * baseParams_.cmpRatio},
            {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim", baseParams_.headDim},
            {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
            {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio", coff * baseParams_.cmpRatio},
            {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim", baseParams_.headDim},
        };
        if (CheckShapeList(opName, checks3D, sizeof(checks3D) / sizeof(checks3D[0])) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc,
                                                                 const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(
        actualDtype != context_->dtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->opName, name, DataTypeToSerialString(actualDtype),
                                              "should be same with x: " + DataTypeToSerialString(context_->dtype)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyX(context_->wkv.desc, WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dCmpKv.desc, D_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dX.desc, D_X_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWkv.desc, D_WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWgate.desc, D_WGATE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->dX.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "d_x", std::to_string(context_->dX.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->dCmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "d_cmp_kv", std::to_string(context_->dCmpKv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->softmaxScore.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "softmax_score",
                    std::to_string(context_->softmaxScore.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->kv.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "kv", std::to_string(context_->kv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_.headDim / D_BASE_SIZE; // D_BASE_SIZE is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "aicNum", std::to_string(aicNum_),
                                                      "should not be less than " + std::to_string(minBlockNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() || ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() || ge::GRAPH_SUCCESS != CheckSingleParaDCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxScore() || ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() || ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDX() || ge::GRAPH_SUCCESS != CheckSingleParaDWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWgate() || ge::GRAPH_SUCCESS != CheckSingleParaDApe() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() || ge::GRAPH_SUCCESS != CheckSingleParaCoff()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckRequiredInOutExistence() const
{
    const TensorCheckItem requiredTensors[] = {
        {"x", context_->x.desc, context_->x.shape},
        {"wkv", context_->wkv.desc, context_->wkv.shape},
        {"wgate", context_->wgate.desc, context_->wgate.shape},
        {"d_cmp_kv", context_->dCmpKv.desc, context_->dCmpKv.shape},
        {"softmax_score", context_->softmaxScore.desc, context_->softmaxScore.shape},
        {"kv", context_->kv.desc, context_->kv.shape},
        {"d_x", context_->dX.desc, context_->dX.shape},
        {"d_wkv", context_->dWkv.desc, context_->dWkv.shape},
        {"d_wgate", context_->dWgate.desc, context_->dWgate.shape},
        {"d_ape", context_->dApe.desc, context_->dApe.shape},
    };
    if (CheckTensorList(context_->opName, requiredTensors, sizeof(requiredTensors) / sizeof(requiredTensors[0])) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // cuSeqlens 特殊校验：TH 布局必须非空，BSH 布局必须为空
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "in TH layout, should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "in TH layout, should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            context_->cuSeqlens.desc != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens", "in BSH layout, must be nullptr"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context_->cuSeqlens.shape != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens", "in BSH layout, must be nullptr"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_ratio", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::CheckEmptyTensor() const
{
    // CompressorGrad 不支持空 tensor：与正向不同（正向 x 支持 B/S/T=0 走 EMPTY_X 分支），
    // 反向无空 tensor 分支——所有输入/输出 shapeSize 必须 > 0，空则直接拦截
    if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dCmpKv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->softmaxScore.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->kv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dX.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dWkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dWgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dApe.shape->GetStorageShape().GetShapeSize() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->opName, "x", "0",
            "CompressorGrad does not support empty tensor: all inputs/outputs shapeSize must be > 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::SetTilingData()
{
    uint8_t coff = coff_;
    tilingData->batch_size = baseParams_.batchSize;
    tilingData->token_size = baseParams_.tokenSize;
    tilingData->seq_size = baseParams_.seqSize;
    tilingData->cmp_ratio = baseParams_.cmpRatio;
    tilingData->hidden_size = baseParams_.hiddenSize;
    tilingData->head_dim = baseParams_.headDim;
    // ── 核数（与 launch block_dim 一致；vec 每核 2 子核）──
    tilingData->cube_core_num = aicNum_;
    tilingData->core_num = aicNum_ * 2;
    // ── shape 派生 ──
    tilingData->total_head_dim = coff * baseParams_.headDim;
    tilingData->cmp_row_cnt = coff * baseParams_.cmpRatio;
    tilingData->cmp_size = coff * baseParams_.cmpRatio * baseParams_.headDim;
    tilingData->cmp_kv_batch_stride = (baseParams_.seqSize + baseParams_.cmpRatio - 1) / baseParams_.cmpRatio;
    if (context_->layout == LayoutType::LAYOUT_BSH) {
        tilingData->x_rows = baseParams_.batchSize * baseParams_.seqSize;
        tilingData->cmp_kv_rows = baseParams_.batchSize * tilingData->cmp_kv_batch_stride;
    } else {
        tilingData->x_rows = baseParams_.tokenSize;
        tilingData->cmp_kv_rows =
            std::min(baseParams_.tokenSize, baseParams_.tokenSize / baseParams_.cmpRatio + baseParams_.batchSize);
    }
    // ── 分核派生 ──
    tilingData->group_size = baseParams_.headDim / D_BASE_SIZE;
    tilingData->group_num = aicNum_ / tilingData->group_size;
    tilingData->cube_m_base_size = M_BASE_SIZE * (COFF_MAX / coff);
    // coff=1 时每 group 每轮只布置 2*dealScNum 块（保证子槽 dealScNum*cmpRatio <= 128，不超 L1/L0 物理行）
    tilingData->deal_sc_num = D_BASE_SIZE / baseParams_.cmpRatio;
    tilingData->group_deal_sc_num = tilingData->deal_sc_num * (COFF_MAX / coff);
    tilingData->total_sc_num_per_round = tilingData->group_num * tilingData->group_deal_sc_num;
    // xArrangeGm 每 group 实际行数 = 数据行(gs 块 × cr) + coff=2 时 1 个 cr 头部（紧凑布局）
    tilingData->group_row_stride =
        tilingData->group_deal_sc_num * baseParams_.cmpRatio + (coff - 1) * baseParams_.cmpRatio;
    tilingData->db_row_cnt = tilingData->group_num * tilingData->group_row_stride;
    // ── 编译期派生（TilingKey 折叠值，与 kernel 内联算术恒等）──
    tilingData->coff_coef = COFF_MAX / coff;
    tilingData->d_deal_size = D_BASE_SIZE / coff;
    tilingData->m_deal_size = M_BASE_SIZE * coff;
    // ── workspace 分区（FP32 元素数；ape/dW 单缓冲，dX/x/dXCache 双缓冲 dbRatio=2）──
    tilingData->dape_ws_size = tilingData->group_num * tilingData->cmp_size * tilingData->coff_coef;
    tilingData->d_x_ws_size = DB_RATIO * tilingData->cube_core_num * (M_BASE_SIZE * COFF_MAX) * baseParams_.hiddenSize;
    tilingData->d_w_weight_ws_size = tilingData->group_num * tilingData->total_head_dim * baseParams_.hiddenSize;
    tilingData->x_ws_size = DB_RATIO * tilingData->group_num * tilingData->group_row_stride * baseParams_.hiddenSize *
                            tilingData->group_size;
    tilingData->d_x_cache_ws_size = DB_RATIO * baseParams_.cmpRatio * baseParams_.hiddenSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingImpl::RunBigKernelTiling()
{
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs{
        std::bind(&CompressorGradTilingImpl::GetNpuInfo, this),
        std::bind(&CompressorGradTilingImpl::CheckRequiredParaExistence, this),
        std::bind(&CompressorGradTilingImpl::CheckEmptyTensor, this),
        std::bind(&CompressorGradTilingImpl::CheckSinglePara, this),
        std::bind(&CompressorGradTilingImpl::SetBaseInfo, this),
        std::bind(&CompressorGradTilingImpl::CheckFeature, this),
        std::bind(&CompressorGradTilingImpl::CheckMultiParaConsistency, this),
        std::bind(&CompressorGradTilingImpl::CheckBlockDimConstrain, this),
        std::bind(&CompressorGradTilingImpl::SetTilingData, this)};
    for (const auto &func : requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    std::vector<StatusFunction> optionalTilingFuncs{std::bind(&CompressorGradTilingImpl::CalcWorkSpace, this),
                                                    std::bind(&CompressorGradTilingImpl::GenTilingKey, this)};
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCompressorGrad(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorGradContext compressorGradContext{};
    if (CompressorGradTilingImpl::ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "context",
            "error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }

    CompressorGradTilingImpl compressorGradTiling(&compressorGradContext);
    compressorGradTiling.tilingData = context->GetTilingData<CompressorGradTiling>();
    OP_CHECK_IF(compressorGradTiling.tilingData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, "tilingData", "is nullptr"),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (compressorGradTiling.RunBigKernelTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
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
    .Tiling(TilingCompressorGrad)
    .TilingParse<CompressorGradCompileInfo>(TilingParseForCompressorGrad);
} // namespace optiling
