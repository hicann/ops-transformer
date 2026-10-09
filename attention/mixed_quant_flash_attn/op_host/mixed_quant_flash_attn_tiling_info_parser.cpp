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
 * \file mixed_quant_flash_attn_tiling_info_parser.cpp
 * \brief MixedQuantFlashAttn tiling info parser
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "mixed_quant_flash_attn_tiling_info_parser.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;
namespace optiling {
namespace mixed_quant_flash_attn {

ge::graphStatus FaInfoParser::GetEmptyTensorFlag()
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

    if (checkEmptyTensor(opParamInfo_.query.shape, QUERY_NAME) || checkEmptyTensor(opParamInfo_.key.shape, KEY_NAME) ||
        checkEmptyTensor(opParamInfo_.value.shape, VALUE_NAME) ||
        checkEmptyTensor(opParamInfo_.kDescale.shape, K_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.vDescale.shape, V_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.attnOut.shape, ATTN_OUT_NAME)) {
        emptyTensorFlag_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of key"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of key"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of value"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of value"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kDescale.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of k_descale"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kDescale.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of k_descale"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vDescale.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of v_descale"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vDescale.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of v_descale"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of atten_out"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of atten_out"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.quantComputeMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "quant_compute_mode"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutQ == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_q"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_attn_out"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetCuSeqLenQSize(int64_t& size)
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

ge::graphStatus FaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("MixedQuantFlashAttn", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510 && npuArch_ != NpuArch::DAV_9201) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaSinksInfo();
    GetOptionalInputParaMaskInfo();
}

void FaInfoParser::GetOptionalInputParaMaskInfo()
{
    opParamInfo_.attnMask.tensor = context_->GetOptionalInputTensor(ATTN_MASK_INDEX);
    opParamInfo_.attnMask.desc = context_->GetOptionalInputDesc(ATTN_MASK_INDEX);
}

void FaInfoParser::GetOptionalInputParaSeqLengthInfo()
{
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.seqUsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.seqUsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.seqUsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.seqUsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
}

void FaInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void FaInfoParser::GetOptionalInputParaSinksInfo()
{
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void FaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INDEX);
    opParamInfo_.kDescale.desc = context_->GetInputDesc(K_DESCALE_INDEX);
    opParamInfo_.kDescale.shape = context_->GetInputShape(K_DESCALE_INDEX);
    opParamInfo_.vDescale.desc = context_->GetInputDesc(V_DESCALE_INDEX);
    opParamInfo_.vDescale.shape = context_->GetInputShape(V_DESCALE_INDEX);
    GetOptionalInputParaInfo();
}

void FaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus FaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    opParamInfo_.quantComputeMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    opParamInfo_.winLeft = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);
    opParamInfo_.winRight = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    opParamInfo_.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_ATTN_OUT_INDEX);
    opParamInfo_.returnSoftMaxLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_LSE_INDEX);

    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetMaskParams()
{
    winLeft_ = (opParamInfo_.winLeft == nullptr) ? -1 : *opParamInfo_.winLeft;
    winRight_ = (opParamInfo_.winRight == nullptr) ? -1 : *opParamInfo_.winRight;
    maskMode_ = (opParamInfo_.maskMode == nullptr) ? 0 : *opParamInfo_.maskMode;
}

ge::graphStatus FaInfoParser::GetQuantModeInfo()
{
    quantMode_ = (opParamInfo_.quantComputeMode == nullptr) ? 1 : *opParamInfo_.quantComputeMode;
    OP_CHECK_IF(quantMode_ != 1 && quantMode_ != 2,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "quant_compute_mode", std::to_string(quantMode_).c_str(),
                                                      "quant_compute_mode must be 1 or 2"),
                return ge::GRAPH_FAILED);

    faQuantMode_ =
        (quantMode_ == 2) ? FaQuantMode::A16C4_KV_HIF4_SOFTMAX_FP32 : FaQuantMode::A16C4_KV_MXFP4_SOFTMAX_FP32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    kDescaleType_ = opParamInfo_.kDescale.desc->GetDataType();
    vDescaleType_ = opParamInfo_.vDescale.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
}

ge::graphStatus FaInfoParser::GetBatchSize()
{
    if (layoutQ_ == FaLayout::TND) {
        return GetCuSeqLenQSize(bSize_);
    } else {
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

void FaInfoParser::GetQueryTSize()
{
    queryTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

void FaInfoParser::GetKeyTSize()
{
    keyTSize_ = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
}

ge::graphStatus FaInfoParser::GetQkHeadDim()
{
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS1Size()
{
    if (layoutQ_ == FaLayout::TND) {
        OP_CHECK_IF(opParamInfo_.maxSeqlenQ == nullptr,
                    OP_LOGE(opName_, "max_seqlen_q", "max_seqlen_q cannot be empty when layout_q is TND"),
                    return ge::GRAPH_FAILED);
        s1Size_ = *opParamInfo_.maxSeqlenQ;
    } else {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    }
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GetKvStorageMode()
{
    if (opParamInfo_.blockTable.tensor != nullptr) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
}

ge::graphStatus FaInfoParser::GetS2SizeForBatchContinuous()
{
    if (layoutKV_ == FaLayout::TND) {
        OP_CHECK_IF(opParamInfo_.maxSeqlenKV == nullptr,
                    OP_LOGE(opName_, "max_seqlen_kv", "max_seqlen_kv cannot be empty when layout_kv is TND"),
                    return ge::GRAPH_FAILED);
        s2Size_ = *opParamInfo_.maxSeqlenKV;
    } else {
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetBlockSize()
{
    blockSize_ = keyShape_->GetShapeBlockSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetBlockNum()
{
    if (keyShape_->CheckHasShapeBlockNum(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockNum_ = keyShape_->GetShapeBlockNum();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetMaxBlockNumPerBatch()
{
    OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                OP_LOGE(opName_, "block_table must be provided for PagedAttention in mixed quant mode."),
                return ge::GRAPH_FAILED);
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS2SizeForPageAttention()
{
    if (GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetBlockNum() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetS2Size()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus FaInfoParser::GetValueHeadDim()
{
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetShapeD());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetInAndOutLayout()
{
    auto itQ = layoutMap.find(opParamInfo_.layoutQ);
    if (itQ == layoutMap.end()) {
        std::string reason = "layout_q: " + std::string(opParamInfo_.layoutQ) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", opParamInfo_.layoutQ, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutQ_ = itQ->second;

    auto itKV = layoutMap.find(opParamInfo_.layoutKV);
    if (itKV == layoutMap.end()) {
        std::string reason = "layout_kv: " + std::string(opParamInfo_.layoutKV) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", opParamInfo_.layoutKV, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutKV_ = itKV->second;

    auto itOut = layoutMap.find(opParamInfo_.layoutOut);
    if (itOut == layoutMap.end()) {
        std::string reason = "layout_attn_out: " + std::string(opParamInfo_.layoutOut) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_attn_out", opParamInfo_.layoutOut, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutOut_ = itOut->second;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetN1Size()
{
    if (queryShape_ != nullptr && queryShape_->HasShapeN()) {
        n1Size_ = static_cast<uint32_t>(queryShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N1 size from query shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetN2Size()
{
    if (keyShape_ != nullptr && keyShape_->HasShapeN()) {
        n2Size_ = static_cast<uint32_t>(keyShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N2 size from key shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::SetFaShape()
{
    queryShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.query.shape->GetStorageShape(), layoutQ_, QUERY_NAME, opName_);
    keyShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.key.shape->GetStorageShape(), layoutKV_, KEY_NAME, opName_);
    valueShape_ =
        std::make_shared<FaTilingShape>(opParamInfo_.value.shape->GetStorageShape(), layoutKV_, VALUE_NAME, opName_);
}

ge::graphStatus FaInfoParser::GetGSize()
{
    if (n2Size_ == 0U) {
        OP_LOGE(opName_, "Kv Heads(%ld) should not be zero.", n2Size_);
        return ge::GRAPH_FAILED;
    }
    if (n1Size_ % n2Size_ != 0U) {
        std::string shapeStr = ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                               ToString(opParamInfo_.key.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and key", shapeStr.c_str(),
                                               "N of query must be an integer multiple of the same axis of key");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::GetActualSeqInfo()
{
    seqUsedQSize_ = opParamInfo_.seqUsedQ.tensor == nullptr ? 0 : opParamInfo_.seqUsedQ.tensor->GetShapeSize();
    seqUsedKvSize_ = opParamInfo_.seqUsedKv.tensor == nullptr ? 0 : opParamInfo_.seqUsedKv.tensor->GetShapeSize();
    cuSeqLensQSize_ = opParamInfo_.cuSeqlensQ.tensor == nullptr ? 0 : opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

void FaInfoParser::GenerateFeatureInfo(FaTilingInfo& faInfo)
{
    faInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    faInfo.blockSize = blockSize_;
    faInfo.blockTypeSize = sizeof(float);

    faInfo.quantComputeMode = quantMode_;
    faInfo.faQuantMode = faQuantMode_;

    faInfo.maskMode = maskMode_;
    faInfo.winLeft = winLeft_;
    faInfo.winRight = winRight_;

    faInfo.sinksFlag = sinksFlag_;

    faInfo.softmaxLseFlag = softmaxLseFlag_;
    faInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();

    faInfo.maxSeqQ = maxSeqQ_;
    faInfo.maxSeqKv = maxSeqKv_;
}

void FaInfoParser::GenerateLayoutInfo(FaTilingInfo& faInfo)
{
    faInfo.qLayout = layoutQ_;
    faInfo.kvLayout = layoutKV_;
    faInfo.outLayout = layoutOut_;
}

void FaInfoParser::GenerateInfo(FaTilingInfo& faInfo)
{
    faInfo.opName = opName_;
    faInfo.platformInfo = platformInfo_;
    faInfo.opParamInfo = opParamInfo_;
    GenerateAxisInfo(faInfo);
    GenerateDtypeInfo(faInfo);
    faInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    faInfo.emptyTensorFlag = emptyTensorFlag_;

    faInfo.totalOutputSize = opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize();
    faInfo.totalBlockNum = blockNum_;
    faInfo.softmaxScale = softmaxScale_;
    faInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    faInfo.seqUsedQSize = seqUsedQSize_;
    faInfo.seqUsedKvSize = seqUsedKvSize_;
    faInfo.cuSeqLensQSize = cuSeqLensQSize_;

    GenerateFeatureInfo(faInfo);
    GenerateLayoutInfo(faInfo);
}

void FaInfoParser::GenerateAxisInfo(FaTilingInfo& faInfo)
{
    faInfo.bSize = bSize_;
    faInfo.n1Size = n1Size_;
    faInfo.n2Size = n2Size_;
    faInfo.s1Size = s1Size_;
    faInfo.s2Size = s2Size_;
    faInfo.gSize = gSize_;
    faInfo.qkHeadDim = qkHeadDim_;
    faInfo.vHeadDim = vHeadDim_;
    faInfo.qTSize = queryTSize_;
    faInfo.kTSize = keyTSize_;
}

void FaInfoParser::GenerateDtypeInfo(FaTilingInfo& faInfo)
{
    faInfo.inputQType = inputQType_;
    faInfo.inputKvType = inputKvType_;
    faInfo.kDescaleType = kDescaleType_;
    faInfo.vDescaleType = vDescaleType_;
    faInfo.outputType = outputType_;
}

ge::graphStatus FaInfoParser::Parse(FaTilingInfo& faInfo)
{
    OP_LOGI(faInfo.opName, "enter FaInfoParser::Parse!");
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
    GetKvStorageMode();
    if (emptyTensorFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(faInfo);
    OP_LOGI(faInfo.opName, "end FaInfoParser::Parse!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::ParseAxisInfo()
{
    SetFaShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }

    GetQueryTSize();

    if (ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }

    GetKeyTSize();

    if (ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaInfoParser::ParseFeatureInfo()
{
    if (ge::GRAPH_SUCCESS != GetQuantModeInfo()) {
        return ge::GRAPH_FAILED;
    }

    GetMaskParams();

    if (ge::GRAPH_SUCCESS != GetActualSeqInfo()) {
        return ge::GRAPH_FAILED;
    }

    sinksFlag_ = (opParamInfo_.sinks.tensor != nullptr);

    returnSoftmaxLse_ = (opParamInfo_.returnSoftMaxLse == nullptr) ? false : *opParamInfo_.returnSoftMaxLse;
    softmaxLseFlag_ = returnSoftmaxLse_;

    softmaxScale_ = (opParamInfo_.softmaxScale == nullptr) ? 1.0f : *opParamInfo_.softmaxScale;
    maxSeqQ_ = (opParamInfo_.maxSeqlenQ == nullptr) ? -1 : *opParamInfo_.maxSeqlenQ;
    maxSeqKv_ = (opParamInfo_.maxSeqlenKV == nullptr) ? -1 : *opParamInfo_.maxSeqlenKV;

    quantMode_ = (opParamInfo_.quantComputeMode == nullptr) ? 0 : *opParamInfo_.quantComputeMode;

    return ge::GRAPH_SUCCESS;
}
} // namespace mixed_quant_flash_attn
} // namespace optiling
