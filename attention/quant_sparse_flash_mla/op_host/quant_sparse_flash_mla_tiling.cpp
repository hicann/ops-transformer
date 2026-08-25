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
 * \file quant_sparse_flash_mla_tiling.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"
#include "checkers/quant_sparse_flash_mla_checker.h"
#include "../../sparse_flash_mla/op_host/checkers/checker_adapter.h"
#include "../op_kernel/quant_sparse_flash_mla_template_tiling_key.h"
#include "quant_sparse_flash_mla_tiling.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::pair;
using std::string;
namespace optiling {

std::string QSMLALayoutToSerialString(QSMLALayout layout)
{
    switch (layout) {
        case QSMLALayout::BSND:
            return "BSND";
        case QSMLALayout::TND:
            return "TND";
        case QSMLALayout::PA_BBND:
            return "PA_BBND";
        default:
            return "UNKNOWN";
    }
}

struct QSMLACompileInfo {
    int64_t core_num;
};

// --------------------------QSMLAInfoParser类成员函数定义-------------------------------------
ge::graphStatus QSMLAInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.q.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor q"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("QuantSparseFlashMla", "opName got from TilingContext");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    OP_LOGD(opName_, "deterministic_level=%d", context_->GetDeterministicLevel());

    return ge::GRAPH_SUCCESS;
}

void QSMLAInfoParser::GetOptionalInputParaInfo()
{
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_KV_INDEX, opParamInfo_.oriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_KV_INDEX, opParamInfo_.cmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, Q_DESCALE, opParamInfo_.qDescale);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_KV_DESCALE, opParamInfo_.oriKvDescale);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_KV_DESCALE, opParamInfo_.cmpKvDescale);
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

void QSMLAInfoParser::GetInputParaInfo()
{
    opParamInfo_.q.desc = context_->GetInputDesc(Q_INDEX);
    opParamInfo_.q.shape = context_->GetInputShape(Q_INDEX);
    GetOptionalInputParaInfo();
}

void QSMLAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.softmaxLse.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.softmaxLse.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus QSMLAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
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

ge::graphStatus QSMLAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetInOutDataType()
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

ge::graphStatus QSMLAInfoParser::GetQueryAndOutLayout()
{
    // 获取q和attnOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<QSMLALayout, QSMLALayout>> layoutMap = {
        {"BSND", {QSMLALayout::BSND, QSMLALayout::BSND}},
        {"TND", {QSMLALayout::TND, QSMLALayout::TND}},
    };

    std::string layout(opParamInfo_.layoutQ);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE(opName_, "layout of Q is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetKvLayout()
{
    const map<string, QSMLALayout> layoutKVMap = {
        {"PA_BBND", QSMLALayout::PA_BBND},
        {"TND", QSMLALayout::TND},
        {"BSND", QSMLALayout::BSND},
    };

    std::string layout(opParamInfo_.layoutKv);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================

bool QSMLAInfoParser::HasAxis(const QSMLAAxis &axis, const QSMLALayout &layout, const gert::Shape &shape) const
{
    const auto &layoutIt = QSMLA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == QSMLA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<QSMLAAxis> &axes = layoutIt->second;
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

size_t QSMLAInfoParser::GetAxisIdx(const QSMLAAxis &axis, const QSMLALayout &layout) const
{
    const std::vector<QSMLAAxis> &axes = QSMLA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t QSMLAInfoParser::GetAxisNum(const gert::Shape &shape, const QSMLAAxis &axis, const QSMLALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void QSMLAInfoParser::SetQSMLAShape()
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

ge::graphStatus QSMLAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(qShape_, QSMLAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetN2Size()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(oriKvShape_, QSMLAAxis::N, kvLayout_);
    } else if (opParamInfo_.cmpKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(cmpKvShape_, QSMLAAxis::N, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, QSMLALayout &layout,
                                                     const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE(opName_, "when layout of q is %s, %s must be provided.", QSMLALayoutToSerialString(layout).c_str(),
                name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.", name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        int64_t shapeSize = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize();
        if (shapeSize <= 1) {
            OP_LOGE(opName_, "the shape size of cuSeqLensQ is %ld, it should be greater than 1.", shapeSize);
            return ge::GRAPH_FAILED;
        }
        size = static_cast<uint32_t>(shapeSize - 1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == QSMLALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(qShape_, QSMLAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QSMLAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == QSMLALayout::TND) ? GetAxisNum(qShape_, QSMLAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == QSMLALayout::TND) {
        s1Size_ = GetAxisNum(qShape_, QSMLAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(qShape_, QSMLAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetMaxBlockNumPerBatch()
{
    if (kvLayout_ == QSMLALayout::TND || kvLayout_ == QSMLALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.oriBlockTable.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, blockTable must be provided.",
                QSMLALayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t oriDimNum = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (oriDimNum != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "ori_block_table", std::to_string(oriDimNum).c_str(),
                                     std::to_string(DIM_NUM_TWO).c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, ORI_BLOCK_TABLE_NAME.c_str(),
            Ops::Base::ToString(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
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
                Ops::Base::ToString(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                CMP_BLOCK_TABLE_NAME + "'s second dimension should be greater than 0");
            return ge::GRAPH_FAILED;
        }
        cmpMaxBlockNumPerBatch_ = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetBlockSize()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriBlockSize_ = GetAxisNum(oriKvShape_, QSMLAAxis::Bs, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpBlockSize_ = GetAxisNum(cmpKvShape_, QSMLAAxis::Bs, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = oriMaxBlockNumPerBatch_ * oriBlockSize_;
    cmpS2Size_ = cmpMaxBlockNumPerBatch_ * cmpBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetS2Size()
{
    if (kvLayout_ == QSMLALayout::TND) {
        s2Size_ = GetAxisNum(oriKvShape_, QSMLAAxis::T, kvLayout_);
        cmpS2Size_ = GetAxisNum(cmpKvShape_, QSMLAAxis::T, kvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == QSMLALayout::BSND) {
        s2Size_ = GetAxisNum(oriKvShape_, QSMLAAxis::S, kvLayout_);
        cmpS2Size_ = GetAxisNum(cmpKvShape_, QSMLAAxis::S, kvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == QSMLALayout::PA_BBND) {
        return GetS2SizeForPageAttention();
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus QSMLAInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(qShape_, QSMLAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetSparseBlockCount()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        cmpSparseBlockCount_ = GetAxisNum(cmpSparseIndicesShape_, QSMLAAxis::K, qLayout_);
    }
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        oriSparseBlockCount_ = GetAxisNum(oriSparseIndicesShape_, QSMLAAxis::K, qLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetDSizeQ()
{
    dSizeQ_ = GetAxisNum(qShape_, QSMLAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetDSizeKV()
{
    dSizeKV_ = GetAxisNum(oriKvShape_, QSMLAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLAInfoParser::GetKvstride()
{
    auto oriKvStrides = context_->GetDynamicInputStride(ORI_KV_INDEX, 0);
    auto cmpKvStrides = context_->GetDynamicInputStride(CMP_KV_INDEX, 0);
    if (oriKvStrides != nullptr && oriKvStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < oriKvStrides->GetDimNum(); i++) {
            oriKvStridesVec_.push_back(oriKvStrides->GetStride(i));
        }
        if (kvLayout_ == QSMLALayout::PA_BBND) {
            oriKvStride_ = oriKvStrides->GetStride(0);
        }
    } else if (kvLayout_ == QSMLALayout::PA_BBND) {
        oriKvStride_ = oriBlockSize_ * n2Size_ * dSizeKV_;
    }
    if (cmpKvStrides != nullptr && cmpKvStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < cmpKvStrides->GetDimNum(); i++) {
            cmpKvStridesVec_.push_back(cmpKvStrides->GetStride(i));
        }
        if (kvLayout_ == QSMLALayout::PA_BBND) {
            cmpKvStride_ = cmpKvStrides->GetStride(0);
        }
    } else if (kvLayout_ == QSMLALayout::PA_BBND) {
        cmpKvStride_ = cmpBlockSize_ * n2Size_ * dSizeKV_;
    }
    return ge::GRAPH_SUCCESS;
}

void QSMLAInfoParser::GenerateInfo(QSMLATilingInfo &qsmlaInfo)
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
    qsmlaInfo.oriSparseBlockCount = oriSparseBlockCount_;
    qsmlaInfo.cmpSparseBlockCount = cmpSparseBlockCount_;

    qsmlaInfo.qType = qType_;
    qsmlaInfo.oriKvType = oriKvType_;
    qsmlaInfo.cmpKvType = cmpKvType_;
    qsmlaInfo.outputType = outputType_;
    qsmlaInfo.dSize = dSizeQ_;
    qsmlaInfo.dSizeV = 512;
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

    qsmlaInfo.quantMode = *opParamInfo_.quantMode;
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
    qsmlaInfo.topkValueMode = *opParamInfo_.topkValueMode;
    qsmlaInfo.oriWinLeft = *opParamInfo_.oriWinLeft;
    qsmlaInfo.oriWinRight = *opParamInfo_.oriWinRight;

    qsmlaInfo.qLayout = qLayout_;
    qsmlaInfo.kvLayout = kvLayout_;
    qsmlaInfo.outLayout = outLayout_;
    qsmlaInfo.returnSoftmaxLse = (opParamInfo_.returnSoftmaxLse != nullptr) ? *opParamInfo_.returnSoftmaxLse : false;
}

ge::graphStatus QSMLAInfoParser::Parse(QSMLATilingInfo &qsmlaInfo)
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
static ge::graphStatus TilingPrepareForQuantSparseFlashMla(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantSparseFlashMlaTiling类成员函数定义-----------------------
ge::graphStatus QuantSparseFlashMlaTiling::DoOpTiling(QSMLATilingInfo *tilingInfo)
{
    if (tilingInfo->opParamInfo.cmpKv.tensor == nullptr) {
        OP_CHECK_IF(tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr,
                    OP_LOGE("QuantSparseFlashMla", "cmpSparseIndices must be empty when cmpKv is not provided."),
                    return ge::GRAPH_FAILED);
        if (tilingInfo->opParamInfo.oriSparseIndices.tensor != nullptr) {
            perfMode_ = QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE;
        } else {
            perfMode_ = QSMLATemplateMode::SWA_TEMPLATE_MODE;
        }
    } else if (tilingInfo->opParamInfo.cmpSparseIndices.tensor != nullptr) {
        if (tilingInfo->opParamInfo.oriSparseIndices.tensor != nullptr) {
            perfMode_ = QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE;
        } else {
            perfMode_ = QSMLATemplateMode::CSA_TEMPLATE_MODE;
        }
    } else {
        perfMode_ = QSMLATemplateMode::HCA_TEMPLATE_MODE;
    }
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(tilingInfo->opName, "QSMLA block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    // -------------set workspacesize-----------------
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t M_BASE_SIZE = 64;   // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 128; // S2轴基本块大小
    constexpr uint32_t D_SIZE = 512;
    constexpr uint32_t VEC_RES_ELEM_SIZE = 2;        // 2: fp16/bf16
    constexpr uint32_t TOPK_MAX_SIZE = 2048;         // TopK选取个数
    constexpr uint32_t UB_SIZE = 248 * 1024;         // UB大小共256KB,预留8k
    constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128; // VF向量化处理的元素对齐粒度
    uint32_t alignedOriSparseBlockCount = (tilingInfo->oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint32_t alignedCmpSparseBlockCount = (tilingInfo->cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint32_t totalBS1 =
        (tilingInfo->qLayout == QSMLALayout::TND) ? tilingInfo->s1Size : (tilingInfo->bSize * tilingInfo->s1Size);

    uint32_t oriBlocksizeFlag = static_cast<uint32_t>((tilingInfo->oriBlockSize & (tilingInfo->oriBlockSize - 1)) == 0);
    uint32_t cmpBlocksizeFlag = static_cast<uint32_t>((tilingInfo->cmpBlockSize & (tilingInfo->cmpBlockSize - 1)) == 0);
    uint32_t blocksizeFlag = (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) ?
                                 oriBlocksizeFlag :
                                 (oriBlocksizeFlag && cmpBlocksizeFlag);

    uint64_t cmpUbSize = static_cast<uint64_t>(tilingInfo->cmpMaxBlockNumPerBatch) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedCmpSparseBlockCount) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedCmpSparseBlockCount) * sizeof(int64_t);
    uint64_t oriUbSize = static_cast<uint64_t>(tilingInfo->oriMaxBlockNumPerBatch) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedOriSparseBlockCount) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedOriSparseBlockCount) * sizeof(int64_t);
    uint64_t vectorizeUbSize = std::max(cmpUbSize, oriUbSize);

    uint32_t vectorizeFlag = static_cast<uint32_t>((perfMode_ == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                                                    perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                                                    perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                                                   (vectorizeUbSize <= UB_SIZE) && blocksizeFlag &&
                                                   (tilingInfo->kvLayout == QSMLALayout::PA_BBND));
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (tilingInfo->gSize > 64) {
        workspaceSize += (S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM * (aicNum >> 1));
    } else {
        workspaceSize += (S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM * aicNum);
    }
    if (vectorizeFlag) {
        uint64_t oriPhyAddrSize = 0;
        if (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
            perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            oriPhyAddrSize = static_cast<uint64_t>(totalBS1) * alignedOriSparseBlockCount * sizeof(int64_t);
        }
        uint64_t cmpPhyAddrSize = static_cast<uint64_t>(totalBS1) * alignedCmpSparseBlockCount * sizeof(int64_t);
        workspaceSize += oriPhyAddrSize + cmpPhyAddrSize;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaces);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_cmpKvSeqSize(tilingInfo->cmpS2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_oriSparseBlockCount(tilingInfo->oriSparseBlockCount);
    tilingData_.baseParams.set_cmpSparseBlockCount(tilingInfo->cmpSparseBlockCount);
    tilingData_.baseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.baseParams.set_paOriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.baseParams.set_paCmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.baseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.baseParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);

    tilingData_.baseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.baseParams.set_oriKvStride(tilingInfo->oriKvStride);
    tilingData_.baseParams.set_cmpKvStride(tilingInfo->cmpKvStride);
    tilingData_.baseParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.baseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.baseParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.baseParams.set_dSize(tilingInfo->dSize);
    tilingData_.baseParams.set_dSizeVInput(tilingInfo->dSizeVInput);
    tilingData_.baseParams.set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t qType = static_cast<uint32_t>(tilingInfo->qType);
    uint32_t oriKvType = static_cast<uint32_t>(tilingInfo->oriKvType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);
    uint32_t tilingKey = GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(perfMode_),
                                            static_cast<uint32_t>(tilingInfo->gSize > 64), DTYPE_HIF8, vectorizeFlag);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingQuantSparseFlashMla(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("QuantSparseFlashMla", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    QSMLATilingInfo qsmlaInfo;
    QSMLAInfoParser qsmlaInfoParser(context);
    if (qsmlaInfoParser.Parse(qsmlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantSparseFlashMlaChecker qsmlaTilingChecker(qsmlaInfo);
    if (qsmlaTilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantSparseFlashMlaTiling tiling(context);
    return tiling.DoOpTiling(&qsmlaInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantSparseFlashMla)
    .Tiling(TilingQuantSparseFlashMla)
    .TilingParse<QSMLACompileInfo>(TilingPrepareForQuantSparseFlashMla);

} // namespace optiling
