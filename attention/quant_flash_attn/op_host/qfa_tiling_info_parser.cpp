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
 * \file qfa_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <iostream>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "qfa_tiling_info_parser.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;
namespace optiling {
namespace quant_flash_attn {

ge::graphStatus QfaInfoParser::GetEmptyTensorFlag()
{
    auto checkEmptyTensor = [this](const gert::StorageShape *shape, const std::string &name) -> bool {
        if (shape == nullptr) {
            return false;
        }
        for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            if (shape->GetStorageShape().GetDim(i) == 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, ToString(shape->GetStorageShape()).c_str(), name.c_str(),
                                                      ("Tensor " + name + " has empty dimension at axis " +
                                                       std::to_string(i) + ", size is 0, which is not supported")
                                                          .c_str());
                return true;
            }
        }
        return false;
    };
    if (checkEmptyTensor(opParamInfo_.query.shape, QUERY_NAME) || checkEmptyTensor(opParamInfo_.key.shape, KEY_NAME) ||
        checkEmptyTensor(opParamInfo_.value.shape, VALUE_NAME) ||
        checkEmptyTensor(opParamInfo_.qDescale.shape, Q_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.kDescale.shape, K_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.vDescale.shape, V_DESCALE_NAME) ||
        checkEmptyTensor(opParamInfo_.attnOut.shape, ATTN_OUT_NAME)) {
        emptyTensorFlag_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::CheckRequiredInOutExistence() const
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
    OP_CHECK_IF(opParamInfo_.qDescale.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of q_descale"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qDescale.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of q_descale"),
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

ge::graphStatus QfaInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQ == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_q"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_out"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.quantMode == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "quant_mode"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetCuSeqLenQSize(int64_t &size)
{
    if (opParamInfo_.cuSeqlensQ.tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, CU_SEQLENS_Q_NAME.c_str());
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

ge::graphStatus QfaInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("QuantFlashAttn", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void QfaInfoParser::GetOptionalInputParaQuantInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.pScale.tensor = context_->GetOptionalInputTensor(P_SCALE_INDEX);
    opParamInfo_.pScale.desc = context_->GetOptionalInputDesc(P_SCALE_INDEX);
}

void QfaInfoParser::GetOptionalInputParaMaskInfo()
{
    opParamInfo_.attnMask.tensor = context_->GetOptionalInputTensor(ATTN_MASK_INDEX);
    opParamInfo_.attnMask.desc = context_->GetOptionalInputDesc(ATTN_MASK_INDEX);
}

void QfaInfoParser::GetOptionalInputParaSeqLengthInfo()
{
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqlensKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
}

void QfaInfoParser::GetOptionalInputParaSinksInfo()
{
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void QfaInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaQuantInfo();
    GetOptionalInputParaSeqLengthInfo();
    GetOptionalInputParaMaskInfo();
    GetOptionalInputParaSinksInfo();
}

void QfaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INDEX);
    opParamInfo_.qDescale.desc = context_->GetInputDesc(Q_DESCALE_INDEX);
    opParamInfo_.qDescale.shape = context_->GetInputShape(Q_DESCALE_INDEX);
    opParamInfo_.kDescale.desc = context_->GetInputDesc(K_DESCALE_INDEX);
    opParamInfo_.kDescale.shape = context_->GetInputShape(K_DESCALE_INDEX);
    opParamInfo_.vDescale.desc = context_->GetInputDesc(V_DESCALE_INDEX);
    opParamInfo_.vDescale.shape = context_->GetInputShape(V_DESCALE_INDEX);

    // 获取 k/v/k_descale/v_descale 的 stride，用于非连续 Tensor 校验
    if (context_->InputIsView(KEY_INDEX) == true) {
        hasStride_ = true;
        keyStrides_ = context_->GetInputStride(KEY_INDEX);
        valueStrides_ = context_->GetInputStride(VALUE_INDEX);
        kDescaleStrides_ = context_->GetInputStride(K_DESCALE_INDEX);
        vDescaleStrides_ = context_->GetInputStride(V_DESCALE_INDEX);
    }

    GetOptionalInputParaInfo();
}

void QfaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus QfaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    opParamInfo_.quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    opParamInfo_.winLeft = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);
    opParamInfo_.winRight = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    opParamInfo_.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutQDescale = attrs->GetStr(ATTR_LAYOUT_Q_DESCALE_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);
    opParamInfo_.returnSoftMaxLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_LSE_INDEX);

    return ge::GRAPH_SUCCESS;
}

void QfaInfoParser::GetMaskParams()
{
    winLeft_ = (opParamInfo_.winLeft == nullptr) ? -1 : *opParamInfo_.winLeft;
    winRight_ = (opParamInfo_.winRight == nullptr) ? -1 : *opParamInfo_.winRight;
    maskMode_ = (opParamInfo_.maskMode == nullptr) ? 0 : *opParamInfo_.maskMode;
}

ge::graphStatus QfaInfoParser::GetQuantMode()
{
    if (opParamInfo_.quantMode == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "quant_mode");
        return ge::GRAPH_FAILED;
    }
    int64_t quantModeVal = *opParamInfo_.quantMode;
    using QM = QfaQuantMode;
    if (quantModeVal != static_cast<int64_t>(QM::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) &&
        quantModeVal !=
            static_cast<int64_t>(
                QM::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) &&
        quantModeVal != static_cast<int64_t>(QM::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "quant_mode", std::to_string(quantModeVal).c_str(),
                                              "quant_mode must be 1 "
                                              "(A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) or "
                                              "6 (A8C8_QK_FP8_"
                                              "E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_"
                                              "P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) or "
                                              "0 (A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32)");
        return ge::GRAPH_FAILED;
    }
    quantMode_ = static_cast<QfaQuantMode>(quantModeVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetBlockSize()
{
    if (keyShape_->CheckHasShapeBlockSize(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockSize_ = keyShape_->GetShapeBlockSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void QfaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
    qDescaleType_ = opParamInfo_.qDescale.desc->GetDataType();
    kDescaleType_ = opParamInfo_.kDescale.desc->GetDataType();
    vDescaleType_ = opParamInfo_.vDescale.desc->GetDataType();
}

ge::graphStatus QfaInfoParser::GetBatchSize()
{
    if (layoutQ_ == QfaLayout::TND || layoutQ_ == QfaLayout::NTD) {
        return GetCuSeqLenQSize(bSize_);
    } else {
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

void QfaInfoParser::GetQueryTSize()
{
    queryTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

void QfaInfoParser::GetKeyTSize()
{
    keyTSize_ = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
}

ge::graphStatus QfaInfoParser::GetQkHeadDim()
{
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetS1Size()
{
    if (layoutQ_ == QfaLayout::TND || layoutQ_ == QfaLayout::NTD) {
        s1Size_ = (opParamInfo_.maxSeqlenQ == nullptr) ? -1 : *opParamInfo_.maxSeqlenQ;
    } else {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    }
    return ge::GRAPH_SUCCESS;
}

void QfaInfoParser::GetKvStorageMode()
{
    bool isPaLayout =
        (layoutKV_ == QfaLayout::PA_BBND || layoutKV_ == QfaLayout::PA_BNBD || layoutKV_ == QfaLayout::PA_NZ);

    if (isPaLayout) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
}

void QfaInfoParser::SetQfaShape()
{
    queryShape_ =
        std::make_shared<QfaTilingShape>(opParamInfo_.query.shape->GetStorageShape(), layoutQ_, QUERY_NAME, opName_);
    keyShape_ =
        std::make_shared<QfaTilingShape>(opParamInfo_.key.shape->GetStorageShape(), layoutKV_, KEY_NAME, opName_);
    valueShape_ =
        std::make_shared<QfaTilingShape>(opParamInfo_.value.shape->GetStorageShape(), layoutKV_, VALUE_NAME, opName_);
}

ge::graphStatus QfaInfoParser::GetS2SizeForBatchContinuous()
{
    if (layoutKV_ == QfaLayout::TND) {
        s2Size_ = keyTSize_;
    } else {
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetBlockNum()
{
    if (keyShape_->CheckHasShapeBlockNum(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockNum_ = keyShape_->GetShapeBlockNum();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetS2SizeForPageAttention()
{
    OP_CHECK_IF(
        opParamInfo_.blockTable.tensor == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "block_table", "provided",
                                              "When layout_kv is PA, block_table must be provided but got nullptr"),
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

ge::graphStatus QfaInfoParser::GetS2Size()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus QfaInfoParser::GetValueHeadDim()
{
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetShapeD());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetInAndOutLayout()
{
    auto itQ = qfaLayoutMap.find(opParamInfo_.layoutQ);
    if (itQ == qfaLayoutMap.end()) {
        std::string reason = "layout_q: " + std::string(opParamInfo_.layoutQ) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", opParamInfo_.layoutQ, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutQ_ = itQ->second;

    if (opParamInfo_.layoutQDescale != nullptr) {
        auto itQDescale = qfaLayoutMap.find(opParamInfo_.layoutQDescale);
        if (itQDescale == qfaLayoutMap.end()) {
            std::string reason = "layout_q_descale: " + std::string(opParamInfo_.layoutQDescale) + " is not supported.";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q_descale", opParamInfo_.layoutQDescale,
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
        layoutQDescale_ = itQDescale->second;
    }

    auto itKV = qfaLayoutMap.find(opParamInfo_.layoutKV);
    if (itKV == qfaLayoutMap.end()) {
        std::string reason = "layout_kv: " + std::string(opParamInfo_.layoutKV) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", opParamInfo_.layoutKV, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutKV_ = itKV->second;

    auto itOut = qfaLayoutMap.find(opParamInfo_.layoutOut);
    if (itOut == qfaLayoutMap.end()) {
        std::string reason = "layout_out: " + std::string(opParamInfo_.layoutOut) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_out", opParamInfo_.layoutOut, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutOut_ = itOut->second;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetN1Size()
{
    if (queryShape_ != nullptr && queryShape_->HasShapeN()) {
        n1Size_ = static_cast<uint32_t>(queryShape_->GetShapeN());
    } else {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, QUERY_NAME.c_str(),
                                              ToString(opParamInfo_.query.shape->GetStorageShape()).c_str(),
                                              "The shape of query must contain the N axis");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetN2Size()
{
    if (keyShape_ != nullptr && keyShape_->HasShapeN()) {
        n2Size_ = static_cast<uint32_t>(keyShape_->GetShapeN());
    } else {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, KEY_NAME.c_str(),
                                              ToString(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                                              "The shape of key must contain the N axis");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::GetGSize()
{
    if (n2Size_ == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "num_key_value_heads", "0",
                                              "The value of num_key_value_heads must be greater than 0");
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

ge::graphStatus QfaInfoParser::GetActualSeqInfo()
{
    return ge::GRAPH_SUCCESS;
}

void QfaInfoParser::GenerateFeatureInfo(QfaTilingInfo &qfaInfo)
{
    qfaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    qfaInfo.blockSize = blockSize_;
    qfaInfo.blockTypeSize = sizeof(uint8_t);

    qfaInfo.maskMode = maskMode_;
    qfaInfo.winLeft = winLeft_;
    qfaInfo.winRight = winRight_;

    qfaInfo.sinksFlag = sinksFlag_;

    qfaInfo.softmaxLseFlag = softmaxLseFlag_;
    qfaInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();

    qfaInfo.maxSeqQ = maxSeqQ_;
    qfaInfo.maxSeqKv = maxSeqKv_;
}

void QfaInfoParser::GenerateLayoutInfo(QfaTilingInfo &qfaInfo)
{
    qfaInfo.qLayout = layoutQ_;
    qfaInfo.kvLayout = layoutKV_;
    qfaInfo.outLayout = layoutOut_;
    qfaInfo.layoutQDescale = layoutQDescale_;
}

void QfaInfoParser::GenerateQuantInfo(QfaTilingInfo &qfaInfo)
{
    qfaInfo.quantMode = quantMode_;
}

void QfaInfoParser::GenerateAxisInfo(QfaTilingInfo &qfaInfo)
{
    qfaInfo.bSize = bSize_;
    qfaInfo.n1Size = n1Size_;
    qfaInfo.n2Size = n2Size_;
    qfaInfo.s1Size = s1Size_;
    qfaInfo.s2Size = s2Size_;
    qfaInfo.gSize = gSize_;
    qfaInfo.qkHeadDim = qkHeadDim_;
    qfaInfo.vHeadDim = vHeadDim_;
    qfaInfo.qTSize = queryTSize_;
    qfaInfo.kTSize = keyTSize_;
}

void QfaInfoParser::GenerateDtypeInfo(QfaTilingInfo &qfaInfo)
{
    qfaInfo.inputQType = inputQType_;
    qfaInfo.inputKvType = inputKvType_;
    qfaInfo.outputType = outputType_;
    qfaInfo.qDescaleType = qDescaleType_;
    qfaInfo.kDescaleType = kDescaleType_;
    qfaInfo.vDescaleType = vDescaleType_;
}

void QfaInfoParser::GenerateInfo(QfaTilingInfo &qfaInfo)
{
    qfaInfo.opName = opName_;
    qfaInfo.platformInfo = platformInfo_;
    qfaInfo.opParamInfo = opParamInfo_;
    qfaInfo.hasStride = hasStride_;
    qfaInfo.keyStrides = keyStrides_;
    qfaInfo.valueStrides = valueStrides_;
    qfaInfo.kDescaleStrides = kDescaleStrides_;
    qfaInfo.vDescaleStrides = vDescaleStrides_;
    GenerateAxisInfo(qfaInfo);
    GenerateDtypeInfo(qfaInfo);
    GenerateQuantInfo(qfaInfo);
    qfaInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    qfaInfo.emptyTensorFlag = emptyTensorFlag_;

    qfaInfo.totalOutputSize = opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize();
    qfaInfo.totalBlockNum = blockNum_;
    qfaInfo.softmaxScale = softmaxScale_;
    qfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    GenerateFeatureInfo(qfaInfo);
    GenerateLayoutInfo(qfaInfo);
}

ge::graphStatus QfaInfoParser::ParseAxisInfo()
{
    SetQfaShape();
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

    uint32_t qDescaleDimNum = opParamInfo_.qDescale.shape->GetStorageShape().GetDimNum();
    using QM = QfaQuantMode;
    if (quantMode_ == QM::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        if (qDescaleDimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, Q_DESCALE_NAME.c_str(),
                                                     (std::to_string(qDescaleDimNum) + "D").c_str(),
                                                     "In HIF8 scenario, the shape dim of q_descale must be 1D");
            return ge::GRAPH_FAILED;
        }
    } else if (quantMode_ ==
               QM::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) {
        if (qDescaleDimNum != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName_, Q_DESCALE_NAME.c_str(), (std::to_string(qDescaleDimNum) + "D").c_str(),
                "In GQA_FP8_FULLQUANT scenario, the shape dim of q_descale must be 2D");
            return ge::GRAPH_FAILED;
        }
    } else {
        bool isDecode = (layoutQDescale_ == QfaLayout::N2TGD);
        if (isDecode && qDescaleDimNum != 5) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName_, Q_DESCALE_NAME.c_str(), (std::to_string(qDescaleDimNum) + "D").c_str(),
                "In MxFP8 decode scenario(layout_q_descale=N2TGD), the shape dim of q_descale must be 5D");
            return ge::GRAPH_FAILED;
        }
        if (!isDecode && qDescaleDimNum != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName_, Q_DESCALE_NAME.c_str(), (std::to_string(qDescaleDimNum) + "D").c_str(),
                "In MxFP8 prefill scenario(layout_q_descale=TND), the shape dim of q_descale must be 4D");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::ParseFeatureInfo()
{
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

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaInfoParser::Parse(QfaTilingInfo &qfaInfo)
{
    OP_LOGI(qfaInfo.opName, "enter QfaInfoParser::Parse!");
    if (context_ == nullptr) {
        OP_LOGE(qfaInfo.opName, "tiling context is nullptr!");
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
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qfaInfo.opName, "input tensor", "",
                                              "Empty tensor (containing a dimension of size 0) is not supported");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetQuantMode()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(qfaInfo);
    OP_LOGI(qfaInfo.opName, "end QfaInfoParser::Parse!");
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
