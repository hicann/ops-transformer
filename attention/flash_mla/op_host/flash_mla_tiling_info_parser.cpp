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
 * \file flash_mla_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <tuple>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "flash_mla_tiling_info_parser.h"
#include "flash_mla_tiling_utils.h"

using namespace ge;
using namespace Ops::Base;
namespace optiling {
namespace flash_mla {

ge::graphStatus FlashMlaInfoParser::GetEmptyTensorFlag()
{
    auto checkEmptyTensor = [this](const gert::StorageShape* shape, const std::string& name) -> bool {
        if (shape == nullptr) {
            return false;
        }
        for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            if (shape->GetStorageShape().GetDim(i) == 0) {
                OP_LOGE(opName_, "Tensor %s has empty dimension at axis %zu, size is 0.", name.c_str(), i);
                return true;
            }
        }
        return false;
    };

    if (checkEmptyTensor(opParamInfo_.query.shape, QUERY_NAME) ||
        checkEmptyTensor(opParamInfo_.kCache.shape, K_CACHE_NAME) ||
        checkEmptyTensor(opParamInfo_.attnOut.shape, ATTN_OUT_NAME)) {
        emptyTensorFlag_ = true;
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kCache.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of k_cache"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kCache.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of k_cache"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of atten_out"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of atten_out"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQ == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_q"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_out"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetCuSeqLenQSize(int64_t& size)
{
    if (opParamInfo_.cuSeqlensQ.tensor == nullptr) {
        OP_LOGE(opName_, "when %s's layout is %s, %s must be provided.", QUERY_NAME.c_str(),
                LayoutToSerialString(layoutQ_).c_str(), CU_SEQLENS_Q_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
    if (shapeSize <= 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(opName_, "cu_seqlens_q", std::to_string(shapeSize).c_str(), "greater than 1");
        return ge::GRAPH_FAILED;
    }
    size = shapeSize - 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("FlashMla", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510 && npuArch_ != NpuArch::DAV_9201 && npuArch_ != NpuArch::DAV_9202) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaMetadataInfo();
    GetOptionalInputParaMaskInfo();
}
void FlashMlaInfoParser::GetOptionalInputParaMaskInfo()
{
    opParamInfo_.attnMask.tensor = context_->GetOptionalInputTensor(ATTN_MASK_INDEX);
    opParamInfo_.attnMask.desc = context_->GetOptionalInputDesc(ATTN_MASK_INDEX);
}

void FlashMlaInfoParser::GetOptionalInputParaSeqLengthInfo()
{
    opParamInfo_.cacheSeqlens.tensor = context_->GetOptionalInputTensor(CACHE_SEQLENS_INDEX);
    opParamInfo_.cacheSeqlens.desc = context_->GetOptionalInputDesc(CACHE_SEQLENS_INDEX);
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
}

void FlashMlaInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void FlashMlaInfoParser::GetOptionalInputParaMetadataInfo()
{
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void FlashMlaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.kCache.desc = context_->GetInputDesc(K_CACHE_INDEX);
    opParamInfo_.kCache.shape = context_->GetInputShape(K_CACHE_INDEX);
    GetOptionalInputParaInfo();
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
}

ge::graphStatus FlashMlaInfoParser::GetStrides()
{
    // 用opbase版本判断：老版本opbase的输入非view, 视为tensorv1, k_cache必然连续
    if (context_->InputIsView(K_CACHE_INDEX) == false) {
        hasViewStride_ = false;
        return ge::GRAPH_SUCCESS;
    }
    keyStrides_ = context_->GetInputStride(K_CACHE_INDEX);
    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus FlashMlaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    opParamInfo_.headDimV = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_DIM_V_INDEX);

    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);

    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);

    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);

    opParamInfo_.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);

    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);

    opParamInfo_.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);

    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);

    opParamInfo_.returnSoftMaxLse = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_LSE_INDEX);
    opParamInfo_.winLeft = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);
    opParamInfo_.winRight = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);

    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::GetMaskParams()
{
    winLeft_ = opParamInfo_.winLeft == nullptr ? -1 : *opParamInfo_.winLeft;
    winRight_ = opParamInfo_.winRight == nullptr ? -1 : *opParamInfo_.winRight;
    maskMode_ = (opParamInfo_.maskMode == nullptr) ? 0 : *opParamInfo_.maskMode;
}

ge::graphStatus FlashMlaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.kCache.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
}

ge::graphStatus FlashMlaInfoParser::GetBatchSize()
{
    if (layoutQ_ == FlashMlaLayout::TND) {
        return GetCuSeqLenQSize(bSize_);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

void FlashMlaInfoParser::GetQueryTSize()
{
    queryTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

ge::graphStatus FlashMlaInfoParser::GetQkNopeHeadDim()
{
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t headDimV = (opParamInfo_.headDimV == nullptr) ? arch35MLA::MLA_D_DIM_512 : *opParamInfo_.headDimV;
    int64_t lastDim = queryShape_->GetShapeD();
    if (lastDim <= headDimV || (lastDim - headDimV) != arch35MLA::MLA_ROPE_D_DIM_64) {
        OP_LOGE(opName_, "q last dim(%ld) must be head_dim_v(%ld) + rope(64).", lastDim, headDimV);
        return ge::GRAPH_FAILED;
    }
    qkNopeHeadDim_ = headDimV;
    ropeHeadDim_ = lastDim - headDimV;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetS1Size()
{
    if (layoutQ_ == FlashMlaLayout::TND) {
        s1Size_ = queryTSize_;
    } else {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    }
    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::GetKvIsContiguous()
{
    // 老版本opbase不支持strides, 无法判NC, 放过
    if (!hasViewStride_) {
        return;
    }

    int32_t keyDimIdx = 0;
    bool keyContig = false;
    if (keyStrides_ != nullptr) {
        const gert::Shape& keyShape = opParamInfo_.kCache.shape->GetStorageShape();
        keyContig =
            (CheckTensorContiguous(keyShape.GetDimNum(), keyShape, keyStrides_, keyDimIdx) == ge::GRAPH_SUCCESS);
        keyNonContigDim_ = keyContig ? -1 : keyDimIdx;
    }

    if (layoutKV_ == FlashMlaLayout::PA_BBND) {
        if (keyStrides_ != nullptr) {
            keyBnStride_ = keyStrides_->GetStride(0);
            keyN2Stride_ = keyStrides_->GetStride(2);
        }
    } else if (layoutKV_ == FlashMlaLayout::PA_BNBD || layoutKV_ == FlashMlaLayout::PA_NZ) {
        if (keyStrides_ != nullptr) {
            keyBnStride_ = keyStrides_->GetStride(0);
            keyN2Stride_ = keyStrides_->GetStride(1);
        }
    }
}

void FlashMlaInfoParser::GetKvStorageMode()
{
    kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    GetKvIsContiguous();
}

ge::graphStatus FlashMlaInfoParser::GetBlockSize()
{
    if (keyShape_->CheckHasShapeBlockSize(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockSize_ = keyShape_->GetShapeBlockSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetBlockNum()
{
    if (keyShape_->CheckHasShapeBlockNum(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockNum_ = keyShape_->GetShapeBlockNum();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetS2SizeForPageAttention()
{
    OP_CHECK_IF(
        opParamInfo_.blockTable.tensor == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "block_table", "provided",
                                              "When layout_kv is PA, block_table must be provided but got nullptr."),
        return ge::GRAPH_FAILED);
    if (GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetBlockNum() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetS2Size()
{
    return GetS2SizeForPageAttention();
}

ge::graphStatus FlashMlaInfoParser::GetValueHeadDim()
{
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t headDimV = (opParamInfo_.headDimV == nullptr) ? arch35MLA::MLA_D_DIM_512 : *opParamInfo_.headDimV;
    int64_t lastDim = valueShape_->GetShapeD();
    if (lastDim <= headDimV || (lastDim - headDimV) != arch35MLA::MLA_ROPE_D_DIM_64) {
        OP_LOGE(opName_, "k_cache last dim(%ld) must be head_dim_v(%ld) + rope(64).", lastDim, headDimV);
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = headDimV;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetInAndOutLayout()
{
    using LayoutNames = std::tuple<std::string, std::string, std::string>;
    using LayoutTypes = std::tuple<FlashMlaLayout, FlashMlaLayout, FlashMlaLayout>;
    static const std::map<LayoutNames, LayoutTypes> supportedLayouts = {
        {{"BSND", "PA_BBND", "BSND"}, {FlashMlaLayout::BSND, FlashMlaLayout::PA_BBND, FlashMlaLayout::BSND}},
        {{"BSND", "PA_BNBD", "BSND"}, {FlashMlaLayout::BSND, FlashMlaLayout::PA_BNBD, FlashMlaLayout::BSND}},
        {{"BSND", "PA_NZ", "BSND"}, {FlashMlaLayout::BSND, FlashMlaLayout::PA_NZ, FlashMlaLayout::BSND}},
        {{"BNSD", "PA_BBND", "BNSD"}, {FlashMlaLayout::BNSD, FlashMlaLayout::PA_BBND, FlashMlaLayout::BNSD}},
        {{"BNSD", "PA_BNBD", "BNSD"}, {FlashMlaLayout::BNSD, FlashMlaLayout::PA_BNBD, FlashMlaLayout::BNSD}},
        {{"BNSD", "PA_NZ", "BNSD"}, {FlashMlaLayout::BNSD, FlashMlaLayout::PA_NZ, FlashMlaLayout::BNSD}},
        {{"TND", "PA_BBND", "TND"}, {FlashMlaLayout::TND, FlashMlaLayout::PA_BBND, FlashMlaLayout::TND}},
        {{"TND", "PA_BBND", "NTD"}, {FlashMlaLayout::TND, FlashMlaLayout::PA_BBND, FlashMlaLayout::NTD}},
        {{"TND", "PA_BNBD", "TND"}, {FlashMlaLayout::TND, FlashMlaLayout::PA_BNBD, FlashMlaLayout::TND}},
        {{"TND", "PA_BNBD", "NTD"}, {FlashMlaLayout::TND, FlashMlaLayout::PA_BNBD, FlashMlaLayout::NTD}},
        {{"TND", "PA_NZ", "TND"}, {FlashMlaLayout::TND, FlashMlaLayout::PA_NZ, FlashMlaLayout::TND}},
        {{"TND", "PA_NZ", "NTD"}, {FlashMlaLayout::TND, FlashMlaLayout::PA_NZ, FlashMlaLayout::NTD}}};

    const auto it = supportedLayouts.find({opParamInfo_.layoutQ, opParamInfo_.layoutKV, opParamInfo_.layoutOut});
    if (it == supportedLayouts.end()) {
        const std::string layouts =
            std::string(opParamInfo_.layoutQ) + ", " + opParamInfo_.layoutKV + ", " + opParamInfo_.layoutOut;
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "layout_q, layout_kv, layout_out", layouts.c_str(),
            "Supported Q/output pairs are BSND/BSND, BNSD/BNSD, TND/TND and TND/NTD; "
            "KV supports PA_BBND, PA_BNBD and PA_NZ. Layout names must be uppercase.");
        return ge::GRAPH_FAILED;
    }
    std::tie(layoutQ_, layoutKV_, layoutOut_) = it->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetN1Size()
{
    if (queryShape_ != nullptr && queryShape_->HasShapeN()) {
        n1Size_ = static_cast<uint32_t>(queryShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N1 size from query shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetN2Size()
{
    if (keyShape_ != nullptr && keyShape_->HasShapeN()) {
        n2Size_ = static_cast<uint32_t>(keyShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N2 size from key shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::SetFaShape()
{
    queryShape_ = std::make_shared<FlashMlaTilingShape>(opParamInfo_.query.shape->GetStorageShape(), layoutQ_,
                                                        QUERY_NAME, opName_);
    keyShape_ = std::make_shared<FlashMlaTilingShape>(opParamInfo_.kCache.shape->GetStorageShape(), layoutKV_,
                                                      K_CACHE_NAME, opName_);
    valueShape_ = std::make_shared<FlashMlaTilingShape>(opParamInfo_.kCache.shape->GetStorageShape(), layoutKV_,
                                                        K_CACHE_NAME, opName_);
}

ge::graphStatus FlashMlaInfoParser::GetGSize()
{
    if (n2Size_ == 0U) {
        OP_LOGE(opName_, "Kv Heads(%ld) should not be zero.", n2Size_);
        return ge::GRAPH_FAILED;
    }
    if (n1Size_ % n2Size_ != 0U) {
        std::string shapeStr = ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                               ToString(opParamInfo_.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and k_cache", shapeStr.c_str(),
                                               "N of query must be an integer multiple of the same axis of k_cache");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::GetActualSeqInfo()
{
    cuSeqlensQSize_ = (opParamInfo_.cuSeqlensQ.tensor == nullptr) ? 0 : opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
    cacheSeqlensSize_ =
        (opParamInfo_.cacheSeqlens.tensor == nullptr) ? 0 : opParamInfo_.cacheSeqlens.tensor->GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

void FlashMlaInfoParser::GetSinksInfo()
{
    sinkFlag_ = opParamInfo_.sinks.tensor != nullptr;
    if (sinkFlag_) {
        const auto& shape = opParamInfo_.sinks.tensor->GetStorageShape();
        sinkFlag_ = shape.GetDimNum() != 1 || shape.GetDim(0) != 0;
    }
}

void FlashMlaInfoParser::GenerateFeatureInfo(FlashMlaTilingInfo& faInfo)
{
    faInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    faInfo.blockSize = blockSize_;

    faInfo.maskMode = maskMode_;
    faInfo.sinkFlag = sinkFlag_;
    faInfo.winLeft = winLeft_;
    faInfo.winRight = winRight_;

    faInfo.returnSoftmaxLse = returnSoftmaxLse_;
    faInfo.softmaxLseFlag = softmaxLseFlag_;

    faInfo.seqUsedQSize = (opParamInfo_.sequsedQ.tensor == nullptr) ? 0 : opParamInfo_.sequsedQ.tensor->GetShapeSize();
    faInfo.cuSeqlensQSize = cuSeqlensQSize_;
    faInfo.cacheSeqlensSize = cacheSeqlensSize_;

    faInfo.maxSeqQ = maxSeqQ_;
    faInfo.maxSeqKv = maxSeqKv_;
}

void FlashMlaInfoParser::GenerateLayoutInfo(FlashMlaTilingInfo& faInfo)
{
    faInfo.qLayout = layoutQ_;
    faInfo.kvLayout = layoutKV_;
    faInfo.outLayout = layoutOut_;
}

void FlashMlaInfoParser::GenerateInfo(FlashMlaTilingInfo& faInfo)
{
    faInfo.opName = opName_;
    faInfo.platformInfo = platformInfo_;
    faInfo.opParamInfo = opParamInfo_;
    GenerateAxisInfo(faInfo);
    GenerateDtypeInfo(faInfo);
    faInfo.emptyTensorFlag = emptyTensorFlag_;

    faInfo.totalBlockNum = blockNum_;
    faInfo.softmaxScale = softmaxScale_;
    faInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    faInfo.keyBnStride = keyBnStride_;
    faInfo.keyN2Stride = keyN2Stride_;

    faInfo.hasViewStride = hasViewStride_;
    faInfo.keyNonContigDim = keyNonContigDim_;

    GenerateFeatureInfo(faInfo);
    GenerateLayoutInfo(faInfo);
}

void FlashMlaInfoParser::GenerateAxisInfo(FlashMlaTilingInfo& faInfo)
{
    faInfo.bSize = bSize_;
    faInfo.n1Size = n1Size_;
    faInfo.n2Size = n2Size_;
    faInfo.s1Size = s1Size_;
    faInfo.s2Size = s2Size_;
    faInfo.gSize = gSize_;
    faInfo.qkNopeHeadDim = qkNopeHeadDim_;
    faInfo.vHeadDim = vHeadDim_;
    faInfo.ropeHeadDim = ropeHeadDim_;
    faInfo.qTSize = queryTSize_;
}

void FlashMlaInfoParser::GenerateDtypeInfo(FlashMlaTilingInfo& faInfo)
{
    faInfo.inputQType = inputQType_;
    faInfo.inputKvType = inputKvType_;
    faInfo.outputType = outputType_;
}

ge::graphStatus FlashMlaInfoParser::Parse(FlashMlaTilingInfo& faInfo)
{
    OP_LOGI(faInfo.opName, "enter FlashMlaInfoParser::Parse!");
    if (context_ == nullptr) {
        OP_LOGE(faInfo.opName, "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence() || ge::GRAPH_SUCCESS != GetEmptyTensorFlag()) {
        return ge::GRAPH_FAILED;
    }
    GetInOutDataType();

    if (ge::GRAPH_SUCCESS != GetInAndOutLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetStrides()) {
        return ge::GRAPH_FAILED;
    }
    GetKvStorageMode();
    if (emptyTensorFlag_) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(faInfo);
    OP_LOGI(faInfo.opName, "end FlashMlaInfoParser::Parse!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::ParseAxisInfo()
{
    SetFaShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }

    GetQueryTSize();

    if (ge::GRAPH_SUCCESS != GetQkNopeHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaInfoParser::ParseFeatureInfo()
{
    GetMaskParams();
    if (maskMode_ != static_cast<int64_t>(MaskMode::NO_MASK) && maskMode_ != static_cast<int64_t>(MaskMode::CAUSAL) &&
        maskMode_ != static_cast<int64_t>(MaskMode::BAND)) {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "mask_mode", std::to_string(maskMode_).c_str(),
                                  "0 (no mask), 3 (right-down causal) or 4 (band)");
        return ge::GRAPH_FAILED;
    }
    if (maskMode_ == static_cast<int64_t>(MaskMode::NO_MASK) && opParamInfo_.attnMask.tensor != nullptr) {
        OP_LOGE(opName_, "When mask_mode=0 (no mask), attn_mask must not be provided.");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualSeqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GetSinksInfo();

    returnSoftmaxLse_ = (opParamInfo_.returnSoftMaxLse == nullptr) ? 0 : *opParamInfo_.returnSoftMaxLse;
    softmaxLseFlag_ = (returnSoftmaxLse_ != 0);

    softmaxScale_ = (opParamInfo_.softmaxScale == nullptr) ? 1.0f : *opParamInfo_.softmaxScale;
    maxSeqQ_ = (opParamInfo_.maxSeqlenQ == nullptr) ? -1 : *opParamInfo_.maxSeqlenQ;
    maxSeqKv_ = (opParamInfo_.maxSeqlenKV == nullptr) ? -1 : *opParamInfo_.maxSeqlenKV;

    return ge::GRAPH_SUCCESS;
}
} // namespace flash_mla
} // namespace optiling
