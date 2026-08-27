/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_compressor_tiling.cpp
 * \file quant_compressor_tiling.cpp
 * \brief
 */

#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "quant_compressor_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus QuantCompressorTiling::ConvertRequiredParams(gert::TilingContext &context,
                                                             QuantCompressorContext &quantCompressorContext)
{
    quantCompressorContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    quantCompressorContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OP_CHECK_IF(quantCompressorContext.x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(quantCompressorContext.opName, X_NAME, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantCompressorContext.x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(quantCompressorContext.opName, X_NAME, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    quantCompressorContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    quantCompressorContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    quantCompressorContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    quantCompressorContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    quantCompressorContext.stateCache.desc = context.GetRequiredInputDesc(STATE_CACHE_INPUT_INDEX);
    quantCompressorContext.stateCache.shape = context.GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    quantCompressorContext.ape.desc = context.GetRequiredInputDesc(APE_INPUT_INDEX);
    quantCompressorContext.ape.shape = context.GetRequiredInputShape(APE_INPUT_INDEX);

    quantCompressorContext.cmpKv.desc = context.GetOutputDesc(CMP_KV_OUTPUT_INDEX);
    quantCompressorContext.cmpKv.shape = context.GetOutputShape(CMP_KV_OUTPUT_INDEX);

    quantCompressorContext.dtype = quantCompressorContext.x.desc->GetDataType();
    auto xDimNum = quantCompressorContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        quantCompressorContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_DIM_NUM_2) {
        quantCompressorContext.layout = LayoutType::LAYOUT_TH;
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(quantCompressorContext.opName, X_NAME, std::to_string(xDimNum),
                                                 "x dimension should be 2 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void QuantCompressorTiling::ConvertOptionalParams(gert::TilingContext &context,
                                                  QuantCompressorContext &quantCompressorContext)
{
    quantCompressorContext.xDescale.desc = context.GetOptionalInputDesc(X_DESCALE_INPUT_INDEX);
    quantCompressorContext.xDescale.shape = context.GetOptionalInputShape(X_DESCALE_INPUT_INDEX);
    quantCompressorContext.wkvDescale.desc = context.GetOptionalInputDesc(WKV_DESCALE_INPUT_INDEX);
    quantCompressorContext.wkvDescale.shape = context.GetOptionalInputShape(WKV_DESCALE_INPUT_INDEX);
    quantCompressorContext.wgateDescale.desc = context.GetOptionalInputDesc(WGATE_DESCALE_INPUT_INDEX);
    quantCompressorContext.wgateDescale.shape = context.GetOptionalInputShape(WGATE_DESCALE_INPUT_INDEX);
    quantCompressorContext.stateBlockTable.desc = context.GetOptionalInputDesc(STATE_BLOCK_TABLE_INPUT_INDEX);
    quantCompressorContext.stateBlockTable.shape = context.GetOptionalInputShape(STATE_BLOCK_TABLE_INPUT_INDEX);
    quantCompressorContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    quantCompressorContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    quantCompressorContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    quantCompressorContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    quantCompressorContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    quantCompressorContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus QuantCompressorTiling::ConvertContext(gert::TilingContext &context,
                                                      QuantCompressorContext &quantCompressorContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantCompressor", "opName", "got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    quantCompressorContext.opName = context.GetNodeName();
    quantCompressorContext.opType = context.GetNodeType();
    quantCompressorContext.platformInfo = context.GetPlatformInfo();
    OP_CHECK_IF(ConvertRequiredParams(context, quantCompressorContext) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    ConvertOptionalParams(context, quantCompressorContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "attrs", "got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    quantCompressorContext.quantMode = attrs->GetAttrPointer<int>(QUANT_MODE_ATTR_INDEX);
    quantCompressorContext.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    quantCompressorContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    quantCompressorContext.cacheMode = attrs->GetAttrPointer<int>(CACHE_MODE_ATTR_INDEX);
    quantCompressorContext.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);
    quantCompressorContext.batchConsistency = context.GetDeterministicLevel();
    OP_LOGD(context.GetNodeName(), "deterministic_level=%d", context.GetDeterministicLevel());
    OP_CHECK_IF(
        context.GetWorkspaceSizes(1) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "workSpaceSize", "got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    quantCompressorContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::GetNpuInfo()
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

ge::graphStatus QuantCompressorTiling::SetBaseInfo()
{
    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_DIM_NUM_3) {
        baseParams_->batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    }

    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) / coff;
    baseParams_->stateCacheStrideDim0 = static_cast<uint64_t>(*context_->stateCacheStrideDim0);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步
    baseParams_->usedCoreNum = aicNum_;
    baseParams_->batchConsistency = static_cast<uint32_t>(context_->batchConsistency);
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u, stateCacheStrideDim0:%llu",
            baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff,
            baseParams_->stateCacheStrideDim0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::LINEAR_BUFFER)) {
        pageAttentionParams_->maxBlockNumPerBatch =
            context_->stateBlockTable.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::SetWorkSpaceInfo()
{
    workspaceParams_->dbWorkspaceRatio = 2; // 2: dbWorkspaceRatio
    workspaceParams_->mm1KvResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    workspaceParams_->mm1ScoreResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    if (coff == 2) { // 2: overlap
        workspaceParams_->vec1TailCacheSize = baseParams_->cmpRatio * baseParams_->headDim;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::SetScenarioInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_X) {
        return ge::GRAPH_SUCCESS;
    }
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND950) {
        // 设置高性能模板
        // 4: max S; 256: max tokensize
        if (context_->layout == LayoutType::LAYOUT_BSH && baseParams_->seqSize <= 4 && baseParams_->tokenSize <= 256) {
            context_->templateId = TemplateId::FULL_LOAD;
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::SetInnerSplitInfo()
{
    if (context_->templateId == TemplateId::FULL_LOAD) {
        innerSplitParams_->mBaseSize = 256;              // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 256 / (coff * 2); // nBase = dBase * coff * 2
        uint32_t dBaseNum = baseParams_->headDim / innerSplitParams_->dBaseSize;
        uint32_t mBaseNum = (baseParams_->tokenSize + innerSplitParams_->mBaseSize - 1) / innerSplitParams_->mBaseSize;
        baseParams_->coreGroupNum = baseParams_->usedCoreNum / dBaseNum;
        baseParams_->kBaseNum = 1;
        baseParams_->kBaseSize = baseParams_->hiddenSize;
        if ((dBaseNum * mBaseNum) < baseParams_->usedCoreNum && baseParams_->batchConsistency != BATCH_CONSISTENCY) {
            baseParams_->kBaseNum = baseParams_->usedCoreNum / dBaseNum;
            uint32_t kAlignSize = (baseParams_->hiddenSize + baseParams_->kBaseNum - 1) / baseParams_->kBaseNum;
            baseParams_->kBaseSize = kAlignSize / 32 * 32; // 切k的size需要32对齐(hifloat8)
        }
        for (uint32_t i = 0; i < baseParams_->usedCoreNum; i++) {
            baseParams_->splitCoreParam[i].nStart = (i % dBaseNum) * innerSplitParams_->dBaseSize;
            baseParams_->splitCoreParam[i].nEnd = baseParams_->splitCoreParam[i].nStart + innerSplitParams_->dBaseSize;
            if (baseParams_->kBaseNum > 1) {
                uint32_t kStartIdx = i / dBaseNum;
                if (kStartIdx + 1 < baseParams_->coreGroupNum) {
                    uint32_t dealKSize = baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                } else {
                    uint32_t dealKSize = kStartIdx < baseParams_->coreGroupNum ?
                                             baseParams_->hiddenSize - kStartIdx * baseParams_->kBaseSize :
                                             0;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                }
                baseParams_->splitCoreParam[i].mStart = 0;
                baseParams_->splitCoreParam[i].mEnd = baseParams_->tokenSize;
                baseParams_->mLoopNum = 1;
            } else {
                baseParams_->splitCoreParam[i].kStart = 0;
                baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + baseParams_->kBaseSize;
                uint32_t mStart = (i / dBaseNum) * innerSplitParams_->mBaseSize;
                baseParams_->splitCoreParam[i].mStart =
                    mStart < baseParams_->tokenSize ? mStart : baseParams_->tokenSize;
                uint32_t mEnd = baseParams_->splitCoreParam[i].mStart + innerSplitParams_->mBaseSize;
                baseParams_->splitCoreParam[i].mEnd = mEnd < baseParams_->tokenSize ? mEnd : baseParams_->tokenSize;
                baseParams_->mLoopNum = mBaseNum / baseParams_->coreGroupNum;
            }
        }
    } else {
        innerSplitParams_->mBaseSize = 256;        // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 128 / coff; // 128：核间切分，D轴基本块大小
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CalcWorkSpace()
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
    uint32_t maxGroupNum = aicNum_ / (baseParams_->headDim / innerSplitParams_->dBaseSize);
    workspaceSize_ = libapiSize_;
    workspaceSize_ +=
        workspaceParams_->mm1KvResSize * maxGroupNum * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ +=
        workspaceParams_->mm1ScoreResSize * maxGroupNum * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ +=
        workspaceParams_->vec1TailCacheSize * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio * 2; // 2 kv和score

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckEmptyTensor() const
{
    if ((context_->layout == LayoutType::LAYOUT_BSH &&
         context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0) ||
        (context_->layout == LayoutType::LAYOUT_BSH &&
         context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1) == 0) ||
        (context_->layout == LayoutType::LAYOUT_TH &&
         context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0)) {
        context_->templateId = TemplateId::EMPTY_X;
    } else {
        if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateCache.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ape.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, "x", "0",
                                                  "Only input tensor x dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::RunBigKernelTiling(QuantCompressorTilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs{std::bind(&QuantCompressorTiling::GetNpuInfo, this),
                                                    std::bind(&QuantCompressorTiling::CheckRequiredParaExistence, this),
                                                    std::bind(&QuantCompressorTiling::CheckEmptyTensor, this),
                                                    std::bind(&QuantCompressorTiling::CheckSinglePara, this),
                                                    std::bind(&QuantCompressorTiling::SetBaseInfo, this),
                                                    std::bind(&QuantCompressorTiling::SetPageAttentionInfo, this),
                                                    std::bind(&QuantCompressorTiling::CheckFeature, this),
                                                    std::bind(&QuantCompressorTiling::CheckMultiParaConsistency, this),
                                                    std::bind(&QuantCompressorTiling::CheckBlockDimConstrain, this),
                                                    std::bind(&QuantCompressorTiling::SetTemplateId, this),
                                                    std::bind(&QuantCompressorTiling::SetInnerSplitInfo, this),
                                                    std::bind(&QuantCompressorTiling::SetWorkSpaceInfo, this),
                                                    std::bind(&QuantCompressorTiling::SetScenarioInfo, this)};
    for (const auto &func : requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == TemplateId::EMPTY_X) {
        workspaceSize_ = libapiSize_;
        if (context_->workSpaces) {
            context_->workSpaces[0] = workspaceSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }
    std::vector<StatusFunction> optionalTilingFuncs{std::bind(&QuantCompressorTiling::CalcWorkSpace, this),
                                                    std::bind(&QuantCompressorTiling::GenTilingKey, this)};
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::GenTilingKey() const
{
    // 0:HIFP8
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);
    uint8_t cacheMode = static_cast<uint8_t>(*context_->cacheMode);
    uint8_t quantMode = static_cast<uint8_t>(*context_->quantMode);

    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(layout, dtype, coff, cacheMode, quantMode, templateId);
    OP_LOGI(context_->opName,
            "QuantCompressor dtype:%hhu layout:%hhu coff:%hhu cacheMode:%u quant_mode:%hhu template_id:%hhu", dtype,
            layout, coff, cacheMode, quantMode, templateId);
    OP_LOGI(context_->opName, "QuantCompressor tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() || ge::GRAPH_SUCCESS != CheckSingleParaCoff() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCacheMode() || ge::GRAPH_SUCCESS != CheckSingleParaQuantMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaX() || ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() || ge::GRAPH_SUCCESS != CheckSingleParaXDescale() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkvDescale() || ge::GRAPH_SUCCESS != CheckSingleParaWgateDescale() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateCache() || ge::GRAPH_SUCCESS != CheckSingleParaApe() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateBlockTable() || ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() || ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus QuantCompressorTiling::CheckFeatureValueSupport(const T *featureValue,
                                                                const std::vector<T> &expectFeatureValList,
                                                                const std::string &name) const
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus QuantCompressorTiling::CheckAttrValueSupport(const T *attrValue,
                                                             const std::vector<T> &expectAttrValList,
                                                             const std::string &name) const
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value");
        return ge::GRAPH_FAILED;
    }

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

template <typename T>
void QuantCompressorTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                                  const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, name, to_string(actualValue),
                                          subName + " only supports " + oss.str());
}

static std::string LayoutTypeToStr(LayoutType layout)
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

ge::graphStatus QuantCompressorTiling::CheckDimNumInLayoutSupport(const std::string &layout,
                                                                  const gert::StorageShape *shape,
                                                                  const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(
        shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->opName, name, std::to_string(shape->GetStorageShape().GetDimNum()),
                                     std::to_string(dimIt->second)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
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
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void QuantCompressorTiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                                 const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE(context_->opName, name, DataTypeToSerialString(actualDtype), oss.str());
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("QuantCompressor", "datatype", std::to_string(static_cast<int32_t>(type)),
                                              "not support");
        return "UNDEFINED";
    }
}

ge::graphStatus QuantCompressorTiling::CheckDimNumSupport(const gert::StorageShape *shape,
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
    OP_CHECK_IF(
        std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
            expectDimNumList.end(),
        [&]() {
            std::ostringstream oss;
            for (size_t i = 0; i < expectDimNumList.size(); ++i) {
                oss << expectDimNumList[i];
                if (i < expectDimNumList.size() - 1) {
                    oss << " or ";
                }
            }
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->opName, name,
                                                     std::to_string(shape->GetStorageShape().GetDimNum()),
                                                     name + " dimension should be " + oss.str());
        }(),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaX() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->x.shape->GetStorageShape().GetDim(context_->x.shape->GetStorageShape().GetDimNum() - 1) >
                        MAX_HIDDEN_SIZE ||
                    context_->x.shape->GetStorageShape().GetDim(context_->x.shape->GetStorageShape().GetDimNum() - 1) <
                        MIN_HIDDEN_SIZE ||
                    context_->x.shape->GetStorageShape().GetDim(context_->x.shape->GetStorageShape().GetDimNum() - 1) %
                            ALIGN_FACTOR_HIDDEN_SIZE !=
                        0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, X_NAME,
                    "dim " + std::to_string(context_->x.shape->GetStorageShape().GetDimNum() - 1) + "=" +
                        std::to_string(context_->x.shape->GetStorageShape().GetDim(
                            context_->x.shape->GetStorageShape().GetDimNum() - 1)),
                    "hiddenSize (x dim" + std::to_string(context_->x.shape->GetStorageShape().GetDimNum() - 1) +
                        ") should be within [" + std::to_string(MIN_HIDDEN_SIZE) + ", " +
                        std::to_string(MAX_HIDDEN_SIZE) + "] and be 512-aligned"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coffVal = static_cast<uint32_t>(static_cast<uint8_t>(*context_->coff));
    uint32_t headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) / coffVal;
    OP_CHECK_IF(std::find(HEAD_DIM.begin(), HEAD_DIM.end(), headDim) == HEAD_DIM.end(),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, WKV_NAME,
                    "dim " + std::to_string(COMPRESSOR_DIM_INDEX_0) + "=" +
                        std::to_string(context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0)),
                    "headDim (wkv dim0 / coff) should be " + std::to_string(HEAD_DIM[0]) + " or " +
                        std::to_string(HEAD_DIM[1])),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaStateCache() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateCache.desc, STATE_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateCache.shape, STATE_CACHE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockSize = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    OP_CHECK_IF(
        blockSize > MAX_BLOCK_SIZE || blockSize < MIN_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, STATE_CACHE_NAME, "dim 1=" + std::to_string(blockSize),
                                              "state_cache dim 1 should be within [" + std::to_string(MIN_BLOCK_SIZE) +
                                                  ", " + std::to_string(MAX_BLOCK_SIZE) + "]"),
        return ge::GRAPH_FAILED);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::RING_BUFFER)) {
        uint32_t blockNum = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        uint32_t batchSize = context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_DIM_NUM_3 ?
                                 context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) :
                                 context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) - 1;
        OP_CHECK_IF(blockNum < batchSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->opName, STATE_CACHE_NAME, "dim 0=" + std::to_string(blockNum),
                        "when cacheMode is " + std::to_string(static_cast<uint8_t>(CACHE_MODE::RING_BUFFER)) +
                            ", state_cache dim 0 should not be less than batchSize(" + std::to_string(batchSize) + ")"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context_->stateBlockTable.shape->GetStorageShape().GetDimNum() != COMPRESSOR_DIM_NUM_1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->opName, STATE_BLOCK_TABLE_NAME,
                                         std::to_string(context_->stateBlockTable.shape->GetStorageShape().GetDimNum()),
                                         std::to_string(COMPRESSOR_DIM_NUM_1)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            context_->stateBlockTable.shape->GetStorageShape().GetDimNum() != COMPRESSOR_DIM_NUM_2,
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->opName, STATE_BLOCK_TABLE_NAME,
                                         std::to_string(context_->stateBlockTable.shape->GetStorageShape().GetDimNum()),
                                         std::to_string(COMPRESSOR_DIM_NUM_2)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaApe() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ape.desc, APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ape.shape, APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaStateBlockTable() const
{
    if (context_->stateBlockTable.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateBlockTable.desc, STATE_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateBlockTable.shape, STATE_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaCuSeqlens() const
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

ge::graphStatus QuantCompressorTiling::CheckSingleParaSeqused() const
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

ge::graphStatus QuantCompressorTiling::CheckSingleParaStartPos() const
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

ge::graphStatus QuantCompressorTiling::CheckSingleParaCmpKv() const
{
    if (context_->cmpKv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cmpKv.desc, CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cmpKv.shape, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaCmpRatio() const
{
    int32_t cmpRatio = *context_->cmpRatio;
    OP_CHECK_IF(
        cmpRatio > MAX_CMPRATIO_SIZE || cmpRatio < MIN_CMPRATIO_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "cmp_ratio", std::to_string(cmpRatio),
                                              "cmp_ratio should be within [" + std::to_string(MIN_CMPRATIO_SIZE) +
                                                  ", " + std::to_string(MAX_CMPRATIO_SIZE) + "]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaCoff() const
{
    if (CheckAttrValueSupport(context_->coff, COFF, COFF_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaCacheMode() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->cacheMode, CACHE_MODE, CACHE_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaXDescale() const
{
    if (context_->xDescale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->xDescale.desc, X_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->xDescale.shape, X_DESCALE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaWkvDescale() const
{
    if (context_->wkvDescale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkvDescale.desc, WKV_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkvDescale.shape, WKV_DESCALE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaWgateDescale() const
{
    if (context_->wgateDescale.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgateDescale.desc, WGATE_DESCALE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgateDescale.shape, WGATE_DESCALE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckSingleParaQuantMode() const
{
    const std::vector<int> QUANT_MODE_VAL_LIST{
        static_cast<int>(QUANT_MODE::A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL)};
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->quantMode, QUANT_MODE_VAL_LIST, QUANT_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "x", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "x", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wkv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wkv", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wgate", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wgate", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_cache", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_cache", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "ape", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "ape", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_block_table", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_block_table", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_kv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_kv", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    if (static_cast<uint8_t>(*context_->quantMode) ==
        static_cast<uint8_t>(QUANT_MODE::A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL)) {
        OP_CHECK_IF(context_->xDescale.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "x_descale",
                                                             "x_descale should not be nullptr in quant_mode = 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->xDescale.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "x_descale",
                                                             "x_descale should not be nullptr in quant_mode = 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->wkvDescale.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wkv_descale",
                                                             "wkv_descale should not be nullptr in quant_mode = 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->wkvDescale.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wkv_descale",
                                                             "wkv_descale should not be nullptr in quant_mode = 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->wgateDescale.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wgate_descale",
                                                             "wgate_descale should not be nullptr in quant_mode = 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->wgateDescale.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wgate_descale",
                                                             "wgate_descale should not be nullptr in quant_mode = 1"),
                    return ge::GRAPH_FAILED);
    }
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens should not be nullptr in TH layout"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens should not be nullptr in TH layout"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->cuSeqlens.desc != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens must be nullptr in BSH layout"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens must be nullptr in BSH layout"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->quantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "quant_mode", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_ratio", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckFeature() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::LogErrorShapeConsistency(const std::string &name,
                                                                const gert::StorageShape *shape, const uint32_t &dimNum,
                                                                const std::string &subName,
                                                                const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, name, "dim " + std::to_string(dimNum) + "=" + std::to_string(actualNum),
                    name + " should be equal to " + subName + ": " + std::to_string(expectNum)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckShapeConsistency() const
{
    auto coffD = coff * baseParams_->headDim;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency(STATE_BLOCK_TABLE_NAME, context_->stateBlockTable.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(CU_SEQLENS_NAME, context_->cuSeqlens.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "batchSize+1",
                                                      baseParams_->batchSize + COMPRESSOR_DIM_NUM_1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(SEQUSED_NAME, context_->seqUsed.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(START_POS_NAME, context_->startPos.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(X_DESCALE_NAME, context_->xDescale.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "1", COMPRESSOR_DIM_NUM_1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WKV_DESCALE_NAME, context_->wkvDescale.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "coff*headDim",
                                                      static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WGATE_DESCALE_NAME, context_->wgateDescale.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "coff*headDim",
                                                      static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WKV_NAME, context_->wkv.shape, COMPRESSOR_DIM_INDEX_1, "x",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WGATE_NAME, context_->wgate.shape, COMPRESSOR_DIM_INDEX_1, "x",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WKV_NAME, context_->wkv.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WGATE_NAME, context_->wgate.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(STATE_CACHE_NAME, context_->stateCache.shape,
                                                      COMPRESSOR_DIM_INDEX_2, "2*coff*headDim",
                                                      COMPRESSOR_DIM_NUM_2 * static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(APE_NAME, context_->ape.shape, COMPRESSOR_DIM_INDEX_1,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(APE_NAME, context_->ape.shape, COMPRESSOR_DIM_INDEX_0,
                                                      CMP_RATIO_NAME, baseParams_->cmpRatio)) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::LINEAR_BUFFER) &&
        (ge::GRAPH_SUCCESS != LogErrorShapeConsistency(STATE_CACHE_NAME, context_->stateCache.shape,
                                                       COMPRESSOR_DIM_INDEX_0, "blockNum",
                                                       pageAttentionParams_->blockNum) ||
         ge::GRAPH_SUCCESS != LogErrorShapeConsistency(STATE_CACHE_NAME, context_->stateCache.shape,
                                                       COMPRESSOR_DIM_INDEX_1, "blockSize",
                                                       pageAttentionParams_->blockSize))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        xDimNum != context_->cmpKv.shape->GetStorageShape().GetDimNum(),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->opName, "cmp_kv, x",
            std::to_string(context_->cmpKv.shape->GetStorageShape().GetDimNum()) + ", " + std::to_string(xDimNum),
            "dim num of cmp_kv should be equal to x"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "aicNum", std::to_string(aicNum_),
                                                      "aicNum should not be less than " + std::to_string(minBlockNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantCompressorTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingQuantCompressor(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("QuantCompressor", "context", "is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    QuantCompressorContext quantCompressorContext{};
    if (QuantCompressorTiling::ConvertContext(*context, quantCompressorContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "context",
            "error occurred while converting tilingContext to QuantCompressor context");
        return ge::GRAPH_FAILED;
    }
    QuantCompressorTiling quantCompressorTiling(&quantCompressorContext);
    QuantCompressorTilingData *tilingData = context->GetTilingData<QuantCompressorTilingData>();
    OP_CHECK_IF(tilingData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(quantCompressorContext.opName, "tilingData", "is nullptr"),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (quantCompressorTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(quantCompressorContext.tilingKey);
    context->SetBlockDim(quantCompressorContext.blockDim);
    OP_LOGI(quantCompressorContext.opName, "block dim: %u.", quantCompressorContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForQuantCompressor(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantCompressor)
    .Tiling(TilingQuantCompressor)
    .TilingParse<QuantCompressorCompileInfo>(TilingPrepareForQuantCompressor);
} // namespace optiling
