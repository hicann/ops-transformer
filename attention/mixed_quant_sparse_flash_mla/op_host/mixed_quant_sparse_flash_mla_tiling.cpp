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
 * \file mixed_quant_sparse_flash_mla_tiling.cpp
 * \brief
 */

#include "mixed_quant_sparse_flash_mla_check.h"
#include "checkers/mixed_quant_sparse_flash_mla_checker.h"
#include "../../sparse_flash_mla/op_host/checkers/checker_adapter.h"
#include "mixed_quant_sparse_flash_mla_tiling.h"
#include <algorithm>

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;
namespace optiling {

constexpr int64_t BATCH_CONSISTENCY_LEVEL = 3;
constexpr uint32_t DEFAULT_D_SIZE_V = 512;
constexpr uint32_t DEFAULT_TILE_SIZE = 512;

std::vector<int64_t> MQSMLAToVector(const gert::Shape &shape)
{
    size_t mqsmlaShapeSize = shape.GetDimNum();
    std::vector<int64_t> mqsmlaShapeVec(mqsmlaShapeSize, 0);

    for (size_t mqsmlaIndex = 0; mqsmlaIndex < mqsmlaShapeSize; mqsmlaIndex++) {
        mqsmlaShapeVec[mqsmlaIndex] = shape.GetDim(mqsmlaIndex);
    }
    return mqsmlaShapeVec;
}

std::string MQSMLAToStringRaw(const gert::Shape &shape)
{
    std::ostringstream mqsmlaOss;
    auto mqsmlaShapeValues = MQSMLAToVector(shape);
    if (mqsmlaShapeValues.size() > 0) {
        for (size_t mqsmlaIndex = 0; mqsmlaIndex < mqsmlaShapeValues.size() - 1; ++mqsmlaIndex) {
            mqsmlaOss << mqsmlaShapeValues[mqsmlaIndex] << ", ";
        }
        mqsmlaOss << mqsmlaShapeValues[mqsmlaShapeValues.size() - 1];
    }
    return mqsmlaOss.str();
}

std::string MQSMLALayoutToSerialString(MQSMLALayout layout)
{
    switch (layout) {
        case MQSMLALayout::BSND:
            return "BSND";
        case MQSMLALayout::TND:
            return "TND";
        case MQSMLALayout::PA_BBND:
            return "PA_BBND";
        default:
            return "UNKNOWN";
    }
}

struct MQSMLACompileInfo {
    int64_t core_num;
};

// --------------------------QSMLAInfoParser类成员函数定义-------------------------------------
ge::graphStatus MQSMLAInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(mqsmlaParams_.q.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(mqsmlaOpName_, "Shape of tensor q"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(mqsmlaParams_.quantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(mqsmlaOpName_, "quant_mode", "Quant_mode is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MixedQuantSparseFlashMla", "opName got from TilingContext");
        return ge::GRAPH_FAILED;
    }
    mqsmlaOpName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetNpuInfo()
{
    mqsmlaPlatformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(mqsmlaPlatformInfo_ == nullptr, OP_LOGE(mqsmlaOpName_, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto mqsmlaPlatform = platform_ascendc::PlatformAscendC(mqsmlaPlatformInfo_);
    uint32_t mqsmlaAivNum = mqsmlaPlatform.GetCoreNumAiv();
    uint32_t mqsmlaAicNum = mqsmlaPlatform.GetCoreNumAic();
    OP_CHECK_IF(mqsmlaAicNum == 0 || mqsmlaAivNum == 0, OP_LOGE(mqsmlaOpName_, "num of core obtained is 0."),
                return ge::GRAPH_FAILED);

    socVersion_ = mqsmlaPlatform.GetSocVersion();
    npuArch_ = mqsmlaPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(mqsmlaOpName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    batchConsistency_ = (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL);
    OP_LOGD(mqsmlaOpName_, "deterministic_level=%d", context_->GetDeterministicLevel());

    return ge::GRAPH_SUCCESS;
}

void MQSMLAInfoParser::GetOptionalInputParaInfo()
{
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_ORI_KV_INDEX, mqsmlaParams_.oriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CMP_KV_INDEX, mqsmlaParams_.cmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_ORI_SPARSE_INDICES_INDEX,
                                                    mqsmlaParams_.oriSparseIndices);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CMP_SPARSE_INDICES_INDEX,
                                                    mqsmlaParams_.cmpSparseIndices);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_ORI_BLOCK_TABLE_INDEX, mqsmlaParams_.oriBlockTable);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CMP_BLOCK_TABLE_INDEX, mqsmlaParams_.cmpBlockTable);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_SINKS_INDEX, mqsmlaParams_.sinks);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CU_SEQLENS_Q_INDEX, mqsmlaParams_.cuSeqLensQ);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CU_SEQLENS_ORI_KV_INDEX, mqsmlaParams_.cuSeqLensOriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CU_SEQLENS_CMP_KV_INDEX, mqsmlaParams_.cuSeqLensCmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_SEQUSED_Q_INDEX, mqsmlaParams_.seqUsedQ);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_SEQUSED_ORI_KV_INDEX, mqsmlaParams_.sequsedOriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_SEQUSED_CMP_KV_INDEX, mqsmlaParams_.sequsedCmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CMP_RESIDUAL_KV_INDEX, mqsmlaParams_.cmpResidualKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_ORI_TOPK_LENGTH_INDEX, mqsmlaParams_.oriTopkLength);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_CMP_TOPK_LENGTH_INDEX, mqsmlaParams_.cmpTopkLength);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, MQ_METADATA_INDEX, mqsmlaParams_.metadata);
}

void MQSMLAInfoParser::GetInputParaInfo()
{
    mqsmlaParams_.q.desc = context_->GetInputDesc(MQ_Q_INDEX);
    mqsmlaParams_.q.shape = context_->GetInputShape(MQ_Q_INDEX);
    GetOptionalInputParaInfo();
}

void MQSMLAInfoParser::GetOutputParaInfo()
{
    mqsmlaParams_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    mqsmlaParams_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    mqsmlaParams_.softmaxLse.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    mqsmlaParams_.softmaxLse.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus MQSMLAInfoParser::GetAttrParaInfo()
{
    auto mqsmlaAttrs = context_->GetAttrs();
    OP_CHECK_IF(mqsmlaAttrs == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    mqsmlaParams_.quantMode = mqsmlaAttrs->GetAttrPointer<int64_t>(MQ_ATTR_QUANT_SCALE_INDEX);
    mqsmlaParams_.tileSize = nullptr;
    mqsmlaParams_.ropeHeadDim = mqsmlaAttrs->GetAttrPointer<int64_t>(MQ_ATTR_ROPE_HEAD_DIM_INDEX);
    mqsmlaParams_.softmaxScale = mqsmlaAttrs->GetAttrPointer<float>(MQ_ATTR_SOFTMAX_SCALE_INDEX);
    mqsmlaParams_.cmpRatio = mqsmlaAttrs->GetAttrPointer<int64_t>(MQ_ATTR_CMP_RATIO_INDEX);
    mqsmlaParams_.oriMaskMode = mqsmlaAttrs->GetAttrPointer<uint32_t>(MQ_ATTR_ORI_MASK_MODE_INDEX);
    mqsmlaParams_.cmpMaskMode = mqsmlaAttrs->GetAttrPointer<uint32_t>(MQ_ATTR_CMP_MASK_MODE_INDEX);
    mqsmlaParams_.oriWinLeft = mqsmlaAttrs->GetAttrPointer<int64_t>(MQ_ATTR_ORI_WIN_LEFT_INDEX);
    mqsmlaParams_.oriWinRight = mqsmlaAttrs->GetAttrPointer<int64_t>(MQ_ATTR_ORI_WIN_RIGHT_INDEX);
    mqsmlaParams_.layoutQ = mqsmlaAttrs->GetStr(MQ_ATTR_LAYOUT_Q_INDEX);
    mqsmlaParams_.layoutKv = mqsmlaAttrs->GetStr(MQ_ATTR_LAYOUT_KV_INDEX);
    mqsmlaParams_.topkValueMode = mqsmlaAttrs->GetAttrPointer<int64_t>(MQ_ATTR_TOPK_VALUE_MODE_INDEX);
    mqsmlaParams_.returnSoftmaxLse = mqsmlaAttrs->GetAttrPointer<bool>(MQ_ATTR_RETURN_SOFTMAX_LSE_INDEX);
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetInOutDataType()
{
    mqsmlaQType_ = mqsmlaParams_.q.desc->GetDataType();
    mqsmlaOutputType_ = mqsmlaParams_.attnOut.desc->GetDataType();
    if (mqsmlaParams_.oriKv.desc != nullptr) {
        mqsmlaOriKvType_ = mqsmlaParams_.oriKv.desc->GetDataType();
    }
    if (mqsmlaParams_.cmpKv.desc != nullptr) {
        mqsmlaCmpKvType_ = mqsmlaParams_.cmpKv.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetQueryAndOutLayout()
{
    // 获取q和attnOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<MQSMLALayout, MQSMLALayout>> mqsmlaLayoutMap = {
        {"BSND", {MQSMLALayout::BSND, MQSMLALayout::BSND}},
        {"TND", {MQSMLALayout::TND, MQSMLALayout::TND}},
    };

    std::string layout(mqsmlaParams_.layoutQ);
    auto it = mqsmlaLayoutMap.find(layout);
    if (it != mqsmlaLayoutMap.end()) {
        mqsmlaQLayout_ = it->second.first;
        mqsmlaOutputLayout_ = it->second.second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(mqsmlaOpName_, "layout_q", layout.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetKvLayout()
{
    const map<string, MQSMLALayout> mqsmlaKvLayoutMap = {
        {"PA_BBND", MQSMLALayout::PA_BBND},
        {"TND", MQSMLALayout::TND},
        {"BSND", MQSMLALayout::BSND},
    };

    std::string layout(mqsmlaParams_.layoutKv);
    auto it = mqsmlaKvLayoutMap.find(layout);
    if (it != mqsmlaKvLayoutMap.end()) {
        mqsmlaKvLayout_ = it->second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(mqsmlaOpName_, "layout_kv", layout.c_str(), "BSND, PA_BBND or TND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================

bool MQSMLAInfoParser::HasAxis(const MQSMLAAxis &axis, const MQSMLALayout &layout, const gert::Shape &shape) const
{
    const auto &mqsmlaLayoutIt = QSMLA_LAYOUT_AXIS_MAP.find(layout);
    if (mqsmlaLayoutIt == QSMLA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<MQSMLAAxis> &mqsmlaAxes = mqsmlaLayoutIt->second;
    const auto &mqsmlaAxisIt = std::find(mqsmlaAxes.begin(), mqsmlaAxes.end(), axis);
    if (mqsmlaAxisIt == mqsmlaAxes.end()) {
        return false;
    }
    const auto &mqsmlaDimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    if (mqsmlaDimIt == QSMLA_LAYOUT_DIM_MAP.end() || mqsmlaDimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t MQSMLAInfoParser::GetAxisIdx(const MQSMLAAxis &axis, const MQSMLALayout &layout) const
{
    const std::vector<MQSMLAAxis> &mqsmlaAxes = QSMLA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto &mqsmlaAxisIt = std::find(mqsmlaAxes.begin(), mqsmlaAxes.end(), axis);
    return std::distance(mqsmlaAxes.begin(), mqsmlaAxisIt);
}

int64_t MQSMLAInfoParser::GetAxisNum(const gert::Shape &shape, const MQSMLAAxis &axis, const MQSMLALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void MQSMLAInfoParser::SetQSMLAShape()
{
    mqsmlaQShape_ = mqsmlaParams_.q.shape->GetStorageShape();
    if (mqsmlaParams_.oriKv.tensor != nullptr) {
        mqsmlaOriKvShape_ = mqsmlaParams_.oriKv.tensor->GetStorageShape();
    }
    if (mqsmlaParams_.cmpKv.tensor != nullptr) {
        mqsmlaCmpKvShape_ = mqsmlaParams_.cmpKv.tensor->GetStorageShape();
    }
    if (mqsmlaParams_.oriSparseIndices.tensor != nullptr) {
        mqsmlaOriSparseIndicesShape_ = mqsmlaParams_.oriSparseIndices.tensor->GetStorageShape();
    }
    if (mqsmlaParams_.cmpSparseIndices.tensor != nullptr) {
        mqsmlaCmpSparseIndicesShape_ = mqsmlaParams_.cmpSparseIndices.tensor->GetStorageShape();
    }
}

ge::graphStatus MQSMLAInfoParser::GetN1Size()
{
    mqsmlaQueryHeads_ = GetAxisNum(mqsmlaQShape_, MQSMLAAxis::N, mqsmlaQLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetN2Size()
{
    if (mqsmlaParams_.oriKv.tensor != nullptr) {
        mqsmlaKvHeads_ = GetAxisNum(mqsmlaOriKvShape_, MQSMLAAxis::N, mqsmlaKvLayout_);
    } else if (mqsmlaParams_.cmpKv.tensor != nullptr) {
        mqsmlaKvHeads_ = GetAxisNum(mqsmlaCmpKvShape_, MQSMLAAxis::N, mqsmlaKvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetGSize()
{
    if (mqsmlaKvHeads_ != 0) {
        mqsmlaGroupSize_ = mqsmlaQueryHeads_ / mqsmlaKvHeads_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetActualSeqLenSize(int64_t &size, const gert::Tensor *tensor, MQSMLALayout &layout,
                                                      const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            mqsmlaOpName_, name.c_str(),
            "When layout_q is " + MQSMLALayoutToSerialString(layout) + ", " + name + " must be provided");
        return ge::GRAPH_FAILED;
    }
    size = tensor->GetShapeSize();
    if (size <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(mqsmlaOpName_, name.c_str(), std::to_string(size).c_str(),
                                                  "The shape size of " + name + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetActualSeqLenQSize(int64_t &size)
{
    if (mqsmlaParams_.cuSeqLensQ.tensor != nullptr) {
        int64_t mqsmlaShapeSize = mqsmlaParams_.cuSeqLensQ.tensor->GetShapeSize();
        if (mqsmlaShapeSize <= 1) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                mqsmlaOpName_, "cu_seqlens_q", std::to_string(mqsmlaParams_.cuSeqLensQ.tensor->GetShapeSize()).c_str(),
                "The shape size of cu_seqlens_q should be greater than 1");
            return ge::GRAPH_FAILED;
        }
        size = mqsmlaShapeSize - 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (mqsmlaQLayout_ == MQSMLALayout::TND) {
        return GetActualSeqLenQSize(mqsmlaBatchSize_);
    } else { // BSND
        mqsmlaBatchSize_ = GetAxisNum(mqsmlaQShape_, MQSMLAAxis::B, mqsmlaQLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus MQSMLAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    mqsmlaQueryTokenSize_ =
        (mqsmlaQLayout_ == MQSMLALayout::TND) ? GetAxisNum(mqsmlaQShape_, MQSMLAAxis::T, mqsmlaQLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (mqsmlaQLayout_ == MQSMLALayout::TND) {
        mqsmlaQuerySeqSize_ = GetAxisNum(mqsmlaQShape_, MQSMLAAxis::T, mqsmlaQLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        mqsmlaQuerySeqSize_ = GetAxisNum(mqsmlaQShape_, MQSMLAAxis::S, mqsmlaQLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetMaxBlockNumPerBatch()
{
    if (mqsmlaKvLayout_ == MQSMLALayout::TND || mqsmlaKvLayout_ == MQSMLALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }
    if (mqsmlaParams_.oriBlockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            mqsmlaOpName_, "ori_block_table",
            "The layout_kv is " + MQSMLALayoutToSerialString(mqsmlaKvLayout_) + ", ori_block_table must be provided");
        return ge::GRAPH_FAILED;
    }
    uint32_t mqsmlaOriDimNum = mqsmlaParams_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (mqsmlaOriDimNum != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(mqsmlaOpName_, "ori_block_table", std::to_string(mqsmlaOriDimNum).c_str(),
                                     std::to_string(DIM_NUM_TWO).c_str());
        return ge::GRAPH_FAILED;
    }
    if (mqsmlaParams_.oriBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            mqsmlaOpName_, ORI_BLOCK_TABLE_NAME.c_str(),
            MQSMLAToStringRaw(mqsmlaParams_.oriBlockTable.tensor->GetStorageShape()).c_str(),
            ORI_BLOCK_TABLE_NAME + "'s second dimension should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    mqsmlaOriMaxBlocksPerBatch_ = mqsmlaParams_.oriBlockTable.tensor->GetStorageShape().GetDim(1);

    if (mqsmlaParams_.cmpBlockTable.tensor != nullptr) {
        uint32_t mqsmlaCmpDimNum = mqsmlaParams_.cmpBlockTable.tensor->GetStorageShape().GetDimNum();
        if (mqsmlaCmpDimNum != DIM_NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(mqsmlaOpName_, "cmp_block_table", std::to_string(mqsmlaCmpDimNum).c_str(),
                                         std::to_string(DIM_NUM_TWO).c_str());
            return ge::GRAPH_FAILED;
        }
        if (mqsmlaParams_.cmpBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                mqsmlaOpName_, CMP_BLOCK_TABLE_NAME.c_str(),
                MQSMLAToStringRaw(mqsmlaParams_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                CMP_BLOCK_TABLE_NAME + "'s second dimension should be greater than 0");
            return ge::GRAPH_FAILED;
        }
        mqsmlaCmpMaxBlocksPerBatch_ = mqsmlaParams_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetBlockSize()
{
    if (mqsmlaParams_.oriKv.tensor != nullptr) {
        mqsmlaOriBlockSize_ = GetAxisNum(mqsmlaOriKvShape_, MQSMLAAxis::Bs, mqsmlaKvLayout_);
    }
    if (mqsmlaParams_.cmpKv.tensor != nullptr) {
        mqsmlaCmpBlockSize_ = GetAxisNum(mqsmlaCmpKvShape_, MQSMLAAxis::Bs, mqsmlaKvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    mqsmlaKvSeqSize_ = mqsmlaOriMaxBlocksPerBatch_ * mqsmlaOriBlockSize_;
    mqsmlaCmpKvSeqSize_ = mqsmlaCmpMaxBlocksPerBatch_ * mqsmlaCmpBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetS2Size()
{
    if (mqsmlaKvLayout_ == MQSMLALayout::TND) {
        mqsmlaKvSeqSize_ = GetAxisNum(mqsmlaOriKvShape_, MQSMLAAxis::T, mqsmlaKvLayout_);
        mqsmlaCmpKvSeqSize_ = GetAxisNum(mqsmlaCmpKvShape_, MQSMLAAxis::T, mqsmlaKvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (mqsmlaKvLayout_ == MQSMLALayout::BSND) {
        mqsmlaKvSeqSize_ = GetAxisNum(mqsmlaOriKvShape_, MQSMLAAxis::S, mqsmlaKvLayout_);
        mqsmlaCmpKvSeqSize_ = GetAxisNum(mqsmlaCmpKvShape_, MQSMLAAxis::S, mqsmlaKvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (mqsmlaKvLayout_ == MQSMLALayout::PA_BBND) {
        return GetS2SizeForPageAttention();
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus MQSMLAInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    mqsmlaQkHeadDim_ = GetAxisNum(mqsmlaQShape_, MQSMLAAxis::D, mqsmlaQLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetSparseBlockCount()
{
    if (mqsmlaParams_.cmpSparseIndices.tensor != nullptr) {
        mqsmlaCmpSparseBlockCount_ = GetAxisNum(mqsmlaCmpSparseIndicesShape_, MQSMLAAxis::K, mqsmlaQLayout_);
    }
    if (mqsmlaParams_.oriSparseIndices.tensor != nullptr) {
        mqsmlaOriSparseBlockCount_ = GetAxisNum(mqsmlaOriSparseIndicesShape_, MQSMLAAxis::K, mqsmlaQLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = mqsmlaKvSeqSize_;
    if (npuArch_ != NpuArch::DAV_2201) {
        return ge::GRAPH_SUCCESS;
    }
    if (mqsmlaQLayout_ == MQSMLALayout::TND && mqsmlaParams_.cuSeqLensQ.tensor != nullptr) {
        mqsmlaActualQueryLenDims_ = static_cast<uint32_t>(mqsmlaParams_.cuSeqLensQ.tensor->GetShapeSize() - 1);
    } else if (mqsmlaParams_.seqUsedQ.tensor != nullptr) {
        mqsmlaActualQueryLenDims_ = static_cast<uint32_t>(mqsmlaParams_.seqUsedQ.tensor->GetShapeSize());
    }
    if (mqsmlaKvLayout_ == MQSMLALayout::PA_BBND && mqsmlaParams_.sequsedOriKv.tensor != nullptr) {
        mqsmlaActualKvLenDims_ = static_cast<uint32_t>(mqsmlaParams_.sequsedOriKv.tensor->GetShapeSize());
    } else if (mqsmlaParams_.cuSeqLensOriKv.tensor != nullptr) {
        mqsmlaActualKvLenDims_ = static_cast<uint32_t>(mqsmlaParams_.cuSeqLensOriKv.tensor->GetShapeSize() - 1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetDSizeQ()
{
    mqsmlaQueryDim_ = GetAxisNum(mqsmlaQShape_, MQSMLAAxis::D, mqsmlaQLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetDSizeKV()
{
    mqsmlaKvDim_ = GetAxisNum(mqsmlaOriKvShape_, MQSMLAAxis::D, mqsmlaKvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetKvstride()
{
    auto mqsmlaOriKvStrides = context_->GetDynamicInputStride(MQ_ORI_KV_INDEX, 0);
    auto mqsmlaCmpKvStrides = context_->GetDynamicInputStride(MQ_CMP_KV_INDEX, 0);
    if (mqsmlaOriKvStrides != nullptr && mqsmlaOriKvStrides->GetDimNum() > 0) {
        for (size_t mqsmlaIndex = 0; mqsmlaIndex < mqsmlaOriKvStrides->GetDimNum(); mqsmlaIndex++) {
            mqsmlaOriKvStrides_.push_back(mqsmlaOriKvStrides->GetStride(mqsmlaIndex));
        }
        if (mqsmlaKvLayout_ == MQSMLALayout::PA_BBND) {
            mqsmlaOriKvStride_ = mqsmlaOriKvStrides->GetStride(0);
        }
    } else if (mqsmlaKvLayout_ == MQSMLALayout::PA_BBND) {
        mqsmlaOriKvStride_ = mqsmlaOriBlockSize_ * mqsmlaKvHeads_ * mqsmlaKvDim_;
        if (npuArch_ == NpuArch::DAV_2201) {
            mqsmlaOriKvStrides_.push_back(mqsmlaOriKvStride_);
        }
    }
    if (mqsmlaCmpKvStrides != nullptr && mqsmlaCmpKvStrides->GetDimNum() > 0) {
        for (size_t mqsmlaIndex = 0; mqsmlaIndex < mqsmlaCmpKvStrides->GetDimNum(); mqsmlaIndex++) {
            mqsmlaCmpKvStrides_.push_back(mqsmlaCmpKvStrides->GetStride(mqsmlaIndex));
        }
        if (mqsmlaKvLayout_ == MQSMLALayout::PA_BBND) {
            mqsmlaCmpKvStride_ = mqsmlaCmpKvStrides->GetStride(0);
        }
    } else if (mqsmlaKvLayout_ == MQSMLALayout::PA_BBND) {
        if (npuArch_ == NpuArch::DAV_2201) {
            const int64_t mqsmlaCmpHeadDim = GetAxisNum(mqsmlaCmpKvShape_, MQSMLAAxis::D, mqsmlaKvLayout_);
            mqsmlaCmpKvStride_ = static_cast<int64_t>(mqsmlaCmpBlockSize_) * mqsmlaKvHeads_ * mqsmlaCmpHeadDim;
            mqsmlaCmpKvStrides_.push_back(mqsmlaCmpKvStride_);
        } else {
            mqsmlaCmpKvStride_ = mqsmlaCmpBlockSize_ * mqsmlaKvHeads_ * mqsmlaKvDim_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MQSMLAInfoParser::GenerateInfo(MQSMLATilingInfo &mqsmlaInfo)
{
    mqsmlaInfo.opName = mqsmlaOpName_;
    mqsmlaInfo.platformInfo = mqsmlaPlatformInfo_;
    mqsmlaInfo.opParamInfo = mqsmlaParams_;
    mqsmlaInfo.socVersion = socVersion_;
    mqsmlaInfo.npuArch = npuArch_;

    mqsmlaInfo.bSize = mqsmlaBatchSize_;
    mqsmlaInfo.n1Size = mqsmlaQueryHeads_;
    mqsmlaInfo.n2Size = mqsmlaKvHeads_;
    mqsmlaInfo.s1Size = mqsmlaQuerySeqSize_;
    mqsmlaInfo.s2Size = mqsmlaKvSeqSize_;
    mqsmlaInfo.cmpS2Size = mqsmlaCmpKvSeqSize_;
    mqsmlaInfo.gSize = mqsmlaGroupSize_;
    mqsmlaInfo.qkHeadDim = mqsmlaQkHeadDim_;
    mqsmlaInfo.qTSize = mqsmlaQueryTokenSize_;
    mqsmlaInfo.actualLenDimsQ = mqsmlaActualQueryLenDims_;
    mqsmlaInfo.actualLenDimsKV = mqsmlaActualKvLenDims_;
    mqsmlaInfo.oriSparseBlockCount = mqsmlaOriSparseBlockCount_;
    mqsmlaInfo.cmpSparseBlockCount = mqsmlaCmpSparseBlockCount_;

    mqsmlaInfo.qType = mqsmlaQType_;
    mqsmlaInfo.oriKvType = mqsmlaOriKvType_;
    mqsmlaInfo.cmpKvType = mqsmlaCmpKvType_;
    mqsmlaInfo.outputType = mqsmlaOutputType_;
    mqsmlaInfo.dSize = mqsmlaQueryDim_;
    mqsmlaInfo.dSizeV = DEFAULT_D_SIZE_V;
    mqsmlaInfo.dSizeVInput = mqsmlaKvDim_;

    mqsmlaInfo.totalBlockNum =
        (mqsmlaParams_.oriKv.tensor != nullptr) ? mqsmlaParams_.oriKv.tensor->GetStorageShape().GetDim(0) : 0;
    mqsmlaInfo.sparseBlockSize = 1;
    mqsmlaInfo.oriBlockSize = mqsmlaOriBlockSize_;
    mqsmlaInfo.cmpBlockSize = mqsmlaCmpBlockSize_;
    mqsmlaInfo.blockTypeSize = sizeof(float);
    mqsmlaInfo.oriMaxBlockNumPerBatch = mqsmlaOriMaxBlocksPerBatch_;
    mqsmlaInfo.cmpMaxBlockNumPerBatch = mqsmlaCmpMaxBlocksPerBatch_;

    mqsmlaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    mqsmlaInfo.batchConsistency = batchConsistency_;

    mqsmlaInfo.quantMode = *mqsmlaParams_.quantMode;
    mqsmlaInfo.tileSize = DEFAULT_TILE_SIZE;
    mqsmlaInfo.ropeHeadDim = *mqsmlaParams_.ropeHeadDim;
    mqsmlaInfo.softmaxScale = *mqsmlaParams_.softmaxScale;
    mqsmlaInfo.oriKvStride = mqsmlaOriKvStride_;
    mqsmlaInfo.cmpKvStride = mqsmlaCmpKvStride_;
    mqsmlaInfo.oriKvStrides = mqsmlaOriKvStrides_;
    mqsmlaInfo.cmpKvStrides = mqsmlaCmpKvStrides_;
    mqsmlaInfo.oriKvStorageShape = mqsmlaOriKvShape_;
    mqsmlaInfo.cmpKvStorageShape = mqsmlaCmpKvShape_;
    mqsmlaInfo.cmpRatio = *mqsmlaParams_.cmpRatio;
    mqsmlaInfo.oriMaskMode = *mqsmlaParams_.oriMaskMode;
    mqsmlaInfo.cmpMaskMode = *mqsmlaParams_.cmpMaskMode;
    mqsmlaInfo.oriWinLeft = *mqsmlaParams_.oriWinLeft;
    mqsmlaInfo.oriWinRight = *mqsmlaParams_.oriWinRight;
    mqsmlaInfo.topkValueMode = *mqsmlaParams_.topkValueMode;
    mqsmlaInfo.qLayout = mqsmlaQLayout_;
    mqsmlaInfo.kvLayout = mqsmlaKvLayout_;
    mqsmlaInfo.outLayout = mqsmlaOutputLayout_;
    mqsmlaInfo.returnSoftmaxLse = (mqsmlaParams_.returnSoftmaxLse != nullptr) ? *mqsmlaParams_.returnSoftmaxLse : false;
}

ge::graphStatus MQSMLAInfoParser::Parse(MQSMLATilingInfo &mqsmlaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("SparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() || ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout()) {
        return ge::GRAPH_FAILED;
    }

    SetQSMLAShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size() || ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetQTSize() || ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount() || ge::GRAPH_SUCCESS != GetDSizeQ() ||
        ge::GRAPH_SUCCESS != GetDSizeKV() || ge::GRAPH_SUCCESS != GetKvstride()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(mqsmlaInfo);
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForMixedQuantSparseFlashMla(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------MixedQuantSparseFlashMlaTiling类成员函数定义-----------------------
ge::graphStatus MixedQuantSparseFlashMlaTiling::DoOpTiling(MQSMLATilingInfo *tilingInfo)
{
    if (tilingInfo->npuArch == NpuArch::DAV_2201) {
        return DoTurboQuantTiling(tilingInfo);
    }
    return DoCsaTiling(tilingInfo);
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingMixedQuantSparseFlashMla(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("MixedQuantSparseFlashMla", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    MQSMLATilingInfo mqsmlaInfo;
    MQSMLAInfoParser mqsmlaInfoParser(context);
    if (mqsmlaInfoParser.Parse(mqsmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    MixedQuantSparseFlashMlaChecker mqsmlaTilingChecker(mqsmlaInfo);
    if (mqsmlaTilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MixedQuantSparseFlashMlaTiling tiling(context);
    return tiling.DoOpTiling(&mqsmlaInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(MixedQuantSparseFlashMla)
    .Tiling(TilingMixedQuantSparseFlashMla)
    .TilingParse<MQSMLACompileInfo>(TilingPrepareForMixedQuantSparseFlashMla);

} // namespace optiling
