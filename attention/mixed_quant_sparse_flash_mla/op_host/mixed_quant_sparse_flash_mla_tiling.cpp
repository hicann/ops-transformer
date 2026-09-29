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

std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
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

struct QSMLACompileInfo {
    int64_t core_num;
};

// --------------------------QSMLAInfoParser类成员函数定义-------------------------------------
ge::graphStatus MQSMLAInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.q.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor q"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.quantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "quant_mode", "Quant_mode is nullptr"),
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
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    batchConsistency_ = (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL);
    OP_LOGD(opName_, "deterministic_level=%d", context_->GetDeterministicLevel());

    return ge::GRAPH_SUCCESS;
}

void MQSMLAInfoParser::GetOptionalInputParaInfo()
{
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_KV_INDEX, opParamInfo_.oriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_KV_INDEX, opParamInfo_.cmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_SPARSE_INDICES_INDEX, opParamInfo_.oriSparseIndices);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_SPARSE_INDICES_INDEX, opParamInfo_.cmpSparseIndices);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_BLOCK_TABLE_INDEX, opParamInfo_.oriBlockTable);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_BLOCK_TABLE_INDEX, opParamInfo_.cmpBlockTable);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SINKS_INDEX, opParamInfo_.sinks);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CU_SEQLENS_Q_INDEX, opParamInfo_.cuSeqLensQ);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CU_SEQLENS_ORI_KV_INDEX, opParamInfo_.cuSeqLensOriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CU_SEQLENS_CMP_KV_INDEX, opParamInfo_.cuSeqLensCmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SEQUSED_Q_INDEX, opParamInfo_.seqUsedQ);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SEQUSED_ORI_KV_INDEX, opParamInfo_.sequsedOriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SEQUSED_CMP_KV_INDEX, opParamInfo_.sequsedCmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_RESIDUAL_KV_INDEX, opParamInfo_.cmpResidualKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_TOPK_LENGTH_INDEX, opParamInfo_.oriTopkLength);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_TOPK_LENGTH_INDEX, opParamInfo_.cmpTopkLength);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, METADATA_INDEX, opParamInfo_.metadata);
}

void MQSMLAInfoParser::GetInputParaInfo()
{
    opParamInfo_.q.desc = context_->GetInputDesc(Q_INDEX);
    opParamInfo_.q.shape = context_->GetInputShape(Q_INDEX);
    GetOptionalInputParaInfo();
}

void MQSMLAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.softmaxLse.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.softmaxLse.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus MQSMLAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_SCALE_INDEX);
    opParamInfo_.tileSize = nullptr;
    opParamInfo_.ropeHeadDim = attrs->GetAttrPointer<int64_t>(ATTR_ROPE_HEAD_DIM_INDEX);
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.oriMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_MASK_MODE_INDEX);
    opParamInfo_.cmpMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_MASK_MODE_INDEX);
    opParamInfo_.oriWinLeft = attrs->GetAttrPointer<int64_t>(ATTR_ORI_WIN_LEFT_INDEX);
    opParamInfo_.oriWinRight = attrs->GetAttrPointer<int64_t>(ATTR_ORI_WIN_RIGHT_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKv = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    opParamInfo_.topkValueMode = attrs->GetAttrPointer<int64_t>(ATTR_TOPK_VALUE_MODE_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE_INDEX);
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
    qType_ = opParamInfo_.q.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
    if (opParamInfo_.oriKv.desc != nullptr) {
        oriKvType_ = opParamInfo_.oriKv.desc->GetDataType();
    }
    if (opParamInfo_.cmpKv.desc != nullptr) {
        cmpKvType_ = opParamInfo_.cmpKv.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetQueryAndOutLayout()
{
    // 获取q和attnOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<MQSMLALayout, MQSMLALayout>> layoutMap = {
        {"BSND", {MQSMLALayout::BSND, MQSMLALayout::BSND}},
        {"TND", {MQSMLALayout::TND, MQSMLALayout::TND}},
    };

    std::string layout(opParamInfo_.layoutQ);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_q", layout.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetKvLayout()
{
    const map<string, MQSMLALayout> layoutKVMap = {
        {"PA_BBND", MQSMLALayout::PA_BBND},
        {"TND", MQSMLALayout::TND},
        {"BSND", MQSMLALayout::BSND},
    };

    std::string layout(opParamInfo_.layoutKv);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_kv", layout.c_str(), "BSND, PA_BBND or TND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================

bool MQSMLAInfoParser::HasAxis(const MQSMLAAxis &axis, const MQSMLALayout &layout, const gert::Shape &shape) const
{
    const auto &layoutIt = QSMLA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == QSMLA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<MQSMLAAxis> &axes = layoutIt->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto &dimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == QSMLA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t MQSMLAInfoParser::GetAxisIdx(const MQSMLAAxis &axis, const MQSMLALayout &layout) const
{
    const std::vector<MQSMLAAxis> &axes = QSMLA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

int64_t MQSMLAInfoParser::GetAxisNum(const gert::Shape &shape, const MQSMLAAxis &axis, const MQSMLALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void MQSMLAInfoParser::SetQSMLAShape()
{
    qShape_ = opParamInfo_.q.shape->GetStorageShape();
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvShape_ = opParamInfo_.oriKv.tensor->GetStorageShape();
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvShape_ = opParamInfo_.cmpKv.tensor->GetStorageShape();
    }
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        oriSparseIndicesShape_ = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
    }
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        cmpSparseIndicesShape_ = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
    }
}

ge::graphStatus MQSMLAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(qShape_, MQSMLAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetN2Size()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(oriKvShape_, MQSMLAAxis::N, kvLayout_);
    } else if (opParamInfo_.cmpKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(cmpKvShape_, MQSMLAAxis::N, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetActualSeqLenSize(int64_t &size, const gert::Tensor *tensor, MQSMLALayout &layout,
                                                      const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, name.c_str(),
            "When layout_q is " + MQSMLALayoutToSerialString(layout) + ", " + name + " must be provided");
        return ge::GRAPH_FAILED;
    }
    size = tensor->GetShapeSize();
    if (size <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, name.c_str(), std::to_string(size).c_str(),
                                                  "The shape size of " + name + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetActualSeqLenQSize(int64_t &size)
{
    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        int64_t shapeSize = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize();
        if (shapeSize <= 1) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cu_seqlens_q", std::to_string(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize()).c_str(),
                "The shape size of cu_seqlens_q should be greater than 1");
            return ge::GRAPH_FAILED;
        }
        size = shapeSize - 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == MQSMLALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(qShape_, MQSMLAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus MQSMLAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == MQSMLALayout::TND) ? GetAxisNum(qShape_, MQSMLAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == MQSMLALayout::TND) {
        s1Size_ = GetAxisNum(qShape_, MQSMLAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(qShape_, MQSMLAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetMaxBlockNumPerBatch()
{
    if (kvLayout_ == MQSMLALayout::TND || kvLayout_ == MQSMLALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.oriBlockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "ori_block_table",
            "The layout_kv is " + MQSMLALayoutToSerialString(kvLayout_) + ", ori_block_table must be provided");
        return ge::GRAPH_FAILED;
    }
    uint32_t oriDimNum = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (oriDimNum != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "ori_block_table", std::to_string(oriDimNum).c_str(),
                                     std::to_string(DIM_NUM_TWO).c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, ORI_BLOCK_TABLE_NAME.c_str(),
                                              ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                                              ORI_BLOCK_TABLE_NAME + "'s second dimension should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    oriMaxBlockNumPerBatch_ = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1);

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        uint32_t cmpDimNum = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDimNum();
        if (cmpDimNum != DIM_NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "cmp_block_table", std::to_string(cmpDimNum).c_str(),
                                         std::to_string(DIM_NUM_TWO).c_str());
            return ge::GRAPH_FAILED;
        }
        if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, CMP_BLOCK_TABLE_NAME.c_str(),
                ToStringRaw(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                CMP_BLOCK_TABLE_NAME + "'s second dimension should be greater than 0");
            return ge::GRAPH_FAILED;
        }
        cmpMaxBlockNumPerBatch_ = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetBlockSize()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriBlockSize_ = GetAxisNum(oriKvShape_, MQSMLAAxis::Bs, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpBlockSize_ = GetAxisNum(cmpKvShape_, MQSMLAAxis::Bs, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = oriMaxBlockNumPerBatch_ * oriBlockSize_;
    cmpS2Size_ = cmpMaxBlockNumPerBatch_ * cmpBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetS2Size()
{
    if (kvLayout_ == MQSMLALayout::TND) {
        s2Size_ = GetAxisNum(oriKvShape_, MQSMLAAxis::T, kvLayout_);
        cmpS2Size_ = GetAxisNum(cmpKvShape_, MQSMLAAxis::T, kvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == MQSMLALayout::BSND) {
        s2Size_ = GetAxisNum(oriKvShape_, MQSMLAAxis::S, kvLayout_);
        cmpS2Size_ = GetAxisNum(cmpKvShape_, MQSMLAAxis::S, kvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == MQSMLALayout::PA_BBND) {
        return GetS2SizeForPageAttention();
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus MQSMLAInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(qShape_, MQSMLAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetSparseBlockCount()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        cmpSparseBlockCount_ = GetAxisNum(cmpSparseIndicesShape_, MQSMLAAxis::K, qLayout_);
    }
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        oriSparseBlockCount_ = GetAxisNum(oriSparseIndicesShape_, MQSMLAAxis::K, qLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = s2Size_;
    if (npuArch_ != NpuArch::DAV_2201) {
        return ge::GRAPH_SUCCESS;
    }
    if (qLayout_ == MQSMLALayout::TND && opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        actualLenDimsQ_ = static_cast<uint32_t>(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() - 1);
    } else if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        actualLenDimsQ_ = static_cast<uint32_t>(opParamInfo_.seqUsedQ.tensor->GetShapeSize());
    }
    if (kvLayout_ == MQSMLALayout::PA_BBND && opParamInfo_.sequsedOriKv.tensor != nullptr) {
        actualLenDimsKV_ = static_cast<uint32_t>(opParamInfo_.sequsedOriKv.tensor->GetShapeSize());
    } else if (opParamInfo_.cuSeqLensOriKv.tensor != nullptr) {
        actualLenDimsKV_ = static_cast<uint32_t>(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize() - 1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetDSizeQ()
{
    dSizeQ_ = GetAxisNum(qShape_, MQSMLAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetDSizeKV()
{
    dSizeKV_ = GetAxisNum(oriKvShape_, MQSMLAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLAInfoParser::GetKvstride()
{
    auto oriKvStrides = context_->GetDynamicInputStride(ORI_KV_INDEX, 0);
    auto cmpKvStrides = context_->GetDynamicInputStride(CMP_KV_INDEX, 0);
    if (oriKvStrides != nullptr && oriKvStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < oriKvStrides->GetDimNum(); i++) {
            oriKvStridesVec_.push_back(oriKvStrides->GetStride(i));
        }
        if (kvLayout_ == MQSMLALayout::PA_BBND) {
            oriKvStride_ = oriKvStrides->GetStride(0);
        }
    } else if (kvLayout_ == MQSMLALayout::PA_BBND) {
        oriKvStride_ = oriBlockSize_ * n2Size_ * dSizeKV_;
        if (npuArch_ == NpuArch::DAV_2201) {
            oriKvStridesVec_.push_back(oriKvStride_);
        }
    }
    if (cmpKvStrides != nullptr && cmpKvStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < cmpKvStrides->GetDimNum(); i++) {
            cmpKvStridesVec_.push_back(cmpKvStrides->GetStride(i));
        }
        if (kvLayout_ == MQSMLALayout::PA_BBND) {
            cmpKvStride_ = cmpKvStrides->GetStride(0);
        }
    } else if (kvLayout_ == MQSMLALayout::PA_BBND) {
        if (npuArch_ == NpuArch::DAV_2201) {
            const int64_t cmpHeadDim = GetAxisNum(cmpKvShape_, MQSMLAAxis::D, kvLayout_);
            cmpKvStride_ = static_cast<int64_t>(cmpBlockSize_) * n2Size_ * cmpHeadDim;
            cmpKvStridesVec_.push_back(cmpKvStride_);
        } else {
            cmpKvStride_ = cmpBlockSize_ * n2Size_ * dSizeKV_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MQSMLAInfoParser::GenerateInfo(MQSMLATilingInfo &qsmlaInfo)
{
    qsmlaInfo.opName = opName_;
    qsmlaInfo.platformInfo = platformInfo_;
    qsmlaInfo.opParamInfo = opParamInfo_;
    qsmlaInfo.socVersion = socVersion_;
    qsmlaInfo.npuArch = npuArch_;

    qsmlaInfo.bSize = bSize_;
    qsmlaInfo.n1Size = n1Size_;
    qsmlaInfo.n2Size = n2Size_;
    qsmlaInfo.s1Size = s1Size_;
    qsmlaInfo.s2Size = s2Size_;
    qsmlaInfo.cmpS2Size = cmpS2Size_;
    qsmlaInfo.gSize = gSize_;
    qsmlaInfo.qkHeadDim = qkHeadDim_;
    qsmlaInfo.qTSize = qTSize_;
    qsmlaInfo.actualLenDimsQ = actualLenDimsQ_;
    qsmlaInfo.actualLenDimsKV = actualLenDimsKV_;
    qsmlaInfo.oriSparseBlockCount = oriSparseBlockCount_;
    qsmlaInfo.cmpSparseBlockCount = cmpSparseBlockCount_;

    qsmlaInfo.qType = qType_;
    qsmlaInfo.oriKvType = oriKvType_;
    qsmlaInfo.cmpKvType = cmpKvType_;
    qsmlaInfo.outputType = outputType_;
    qsmlaInfo.dSize = dSizeQ_;
    qsmlaInfo.dSizeV = DEFAULT_D_SIZE_V;
    qsmlaInfo.dSizeVInput = dSizeKV_;

    qsmlaInfo.totalBlockNum =
        (opParamInfo_.oriKv.tensor != nullptr) ? opParamInfo_.oriKv.tensor->GetStorageShape().GetDim(0) : 0;
    qsmlaInfo.sparseBlockSize = 1; // 写死为1
    qsmlaInfo.oriBlockSize = oriBlockSize_;
    qsmlaInfo.cmpBlockSize = cmpBlockSize_;
    qsmlaInfo.blockTypeSize = sizeof(float);
    qsmlaInfo.oriMaxBlockNumPerBatch = oriMaxBlockNumPerBatch_;
    qsmlaInfo.cmpMaxBlockNumPerBatch = cmpMaxBlockNumPerBatch_;

    qsmlaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    qsmlaInfo.batchConsistency = batchConsistency_;

    qsmlaInfo.quantMode = *opParamInfo_.quantMode;
    qsmlaInfo.tileSize = DEFAULT_TILE_SIZE;
    qsmlaInfo.ropeHeadDim = *opParamInfo_.ropeHeadDim;
    qsmlaInfo.softmaxScale = *opParamInfo_.softmaxScale;
    qsmlaInfo.oriKvStride = oriKvStride_;
    qsmlaInfo.cmpKvStride = cmpKvStride_;
    qsmlaInfo.oriKvStrides = oriKvStridesVec_;
    qsmlaInfo.cmpKvStrides = cmpKvStridesVec_;
    qsmlaInfo.oriKvStorageShape = oriKvShape_;
    qsmlaInfo.cmpKvStorageShape = cmpKvShape_;
    qsmlaInfo.cmpRatio = *opParamInfo_.cmpRatio;
    qsmlaInfo.oriMaskMode = *opParamInfo_.oriMaskMode;
    qsmlaInfo.cmpMaskMode = *opParamInfo_.cmpMaskMode;
    qsmlaInfo.oriWinLeft = *opParamInfo_.oriWinLeft;
    qsmlaInfo.oriWinRight = *opParamInfo_.oriWinRight;
    qsmlaInfo.topkValueMode = *opParamInfo_.topkValueMode;
    qsmlaInfo.qLayout = qLayout_;
    qsmlaInfo.kvLayout = kvLayout_;
    qsmlaInfo.outLayout = outLayout_;
    qsmlaInfo.returnSoftmaxLse = (opParamInfo_.returnSoftmaxLse != nullptr) ? *opParamInfo_.returnSoftmaxLse : false;
}

ge::graphStatus MQSMLAInfoParser::Parse(MQSMLATilingInfo &qsmlaInfo)
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

    GenerateInfo(qsmlaInfo);
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
    MQSMLATilingInfo qsmlaInfo;
    MQSMLAInfoParser qsmlaInfoParser(context);
    if (qsmlaInfoParser.Parse(qsmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    MixedQuantSparseFlashMlaChecker qsmlaTilingChecker(qsmlaInfo);
    if (qsmlaTilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MixedQuantSparseFlashMlaTiling tiling(context);
    return tiling.DoOpTiling(&qsmlaInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(MixedQuantSparseFlashMla)
    .Tiling(TilingMixedQuantSparseFlashMla)
    .TilingParse<QSMLACompileInfo>(TilingPrepareForMixedQuantSparseFlashMla);

} // namespace optiling
