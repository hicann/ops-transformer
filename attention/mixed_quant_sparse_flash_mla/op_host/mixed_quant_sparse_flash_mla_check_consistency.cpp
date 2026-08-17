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
 * \file mixed_quant_sparse_flash_mla_check_consistency.cpp
 * \brief
 */

#include "mixed_quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;
namespace optiling {

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

ge::graphStatus MQSMLATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
                                                         const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(opName_, name.c_str(), MQSMLADataTypeToSerialString(actualDtype).c_str(),
                                  MQSMLADataTypeToSerialString(expectDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                       const MQSMLALayout &layout, const std::string &name) const
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, name.c_str(),
            "When layout_q is " + MQSMLALayoutToSerialString(layout) + ", " + name + " must be provided");
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, name.c_str(), std::to_string(shapeSize).c_str(),
                                                  "the shape size of " + name + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
                                                    const QSMLATilingShapeCompareParam &param,
                                                    const MQSMLALayout &layout) const
{
    if (layout == MQSMLALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == MQSMLALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == MQSMLALayout::PA_BBND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", MQSMLALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CompareShape(QSMLATilingShapeCompareParam &param, const gert::Shape &shape,
                                                const MQSMLALayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, name.c_str(), std::to_string(shape.GetDimNum()).c_str(),
                                                 std::to_string(shapeExpected.GetDimNum()).c_str());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPE(opName_, name.c_str(), ToStringRaw(shape).c_str(),
                                      ToStringRaw(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void MQSMLATilingCheck::SetQSMLAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.q.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.cmpSparseIndices.tensor->GetShape().GetStorageShape();
    keyShapeCmp_ = opParamInfo_.oriKv.tensor->GetShape().GetStorageShape();
    valueShapeCmp_ = opParamInfo_.cmpKv.tensor->GetShape().GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attnOut.shape->GetStorageShape();
}

ge::graphStatus MQSMLATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, ORI_BLOCK_TABLE_NAME.c_str(),
                        ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                        "when the layout_kv is " + MQSMLALayoutToSerialString(kvLayout_) + ", " + ORI_BLOCK_TABLE_NAME +
                            " should be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, CMP_BLOCK_TABLE_NAME.c_str(),
                        ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                        "when the layout_kv is " + MQSMLALayoutToSerialString(kvLayout_) + ", " + CMP_BLOCK_TABLE_NAME +
                            " should be null"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t oriBlockTableBatch = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(
        oriBlockTableBatch != bSize_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "oriBlockTableBatch", ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
            "oriBlockTableBatch's first dimension(" + std::to_string(oriBlockTableBatch) +
                ") should be equal to batch size" + std::to_string(bSize_)),
        return ge::GRAPH_FAILED);

    uint32_t cmpBlockTableBatch = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(
        cmpBlockTableBatch != bSize_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "cmpBlockTableBatch", ToStringRaw(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
            "cmpBlockTableBatch's first dimension(" + std::to_string(cmpBlockTableBatch) +
                ") should be equal to batch size(" + std::to_string(bSize_) + ")"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckTopkShape() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MQSMLATilingCheck::CheckAttenOutShape()
{
    QSMLATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = 512; // 512:输出的head_dim
    shapeParams.T = qTSize_;
    if (CheckDTypeConsistency(opParamInfo_.attnOut.desc->GetDataType(), qType_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attnOut.desc->GetDataType(), qType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSoftmaxLse() const
{
    bool returnSoftmaxLse = (opParamInfo_.returnSoftmaxLse != nullptr) ? *opParamInfo_.returnSoftmaxLse : false;
    if (returnSoftmaxLse) {
        if (opParamInfo_.softmaxLse.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "softmaxLse",
                std::to_string(opParamInfo_.softmaxLse.shape->GetStorageShape().GetShapeSize()).c_str(),
                "The shape size of softmax_lse should be greater than 0");
            return ge::GRAPH_FAILED;
        }
        if (opParamInfo_.softmaxLse.desc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "softmaxLse",
                MQSMLADataTypeToSerialString(opParamInfo_.softmaxLse.desc->GetDataType()).c_str(),
                "The dtype of softmax_lse must be FLOAT");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckKVShapeForBatchContinuous()
{
    QSMLATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t MQSMLATilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus MQSMLATilingCheck::CheckKVShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    QSMLATilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = bSize_;
    shapeParams.T = kvTSize_;
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckKVShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckKVShapeForPageAttention();
    }

    OP_LOGE(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus MQSMLATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(cmpKvType_, oriKvType_, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.cuSeqLensQ.desc == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
                                                 "Cu_seqlens_q is not empty, but the dtype of cu_seqlens_q is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.cuSeqLensQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "cu_seqlens_q", MQSMLADataTypeToSerialString(opParamInfo_.cuSeqLensQ.desc->GetDataType()).c_str(),
            "The dtype of cu_seqlens_q must be DT_INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cuSeqLensQ") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_ + 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "cu_seqlens_q", std::to_string(shapeSize).c_str(),
            "The shape size of cu_seqlens_q should be equal to batch size " + std::to_string(bSize_));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.sequsedOriKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.sequsedOriKv.desc == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sequsedOriKv's dtype",
                                                 "sequsedOriKv is not empty, but sequsedOriKv's dtype is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sequsedOriKv.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "sequsedOriKv",
            MQSMLADataTypeToSerialString(opParamInfo_.sequsedOriKv.desc->GetDataType()).c_str(),
            "sequsedOriKv's dtype should be DT_INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.sequsedOriKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.sequsedOriKv.tensor, kvLayout_, "sequsedOriKv") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "sequsedOriKv", std::to_string(shapeSize).c_str(),
            "sequsedOriKv shape size should be equal to batch size[" + std::to_string(bSize_) + "]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckMultiParaConsistency()
{
    SetQSMLAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() || ge::GRAPH_SUCCESS != CheckTopK() || ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() || ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckLayoutQKvConsistency() const
{
    if (kvLayout_ != MQSMLALayout::PA_BBND) {
        OP_CHECK_IF(qLayout_ != kvLayout_,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "layout_kv and layout_q",
                        MQSMLALayoutToSerialString(kvLayout_) + " and " + MQSMLALayoutToSerialString(qLayout_),
                        "When layout_kv is not PA_BSND, layout_kv and layout_q must be same"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSparseIndicesShapeMatchQ() const
{
    if (perfMode_ != QSMLATemplateMode::CSA_TEMPLATE_MODE && perfMode_ != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        perfMode_ != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }

    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        const auto &oriSparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == MQSMLALayout::BSND) {
            OP_CHECK_IF(
                oriSparseShape.GetDim(DIM_0) != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_sparse_indices", ToStringRaw(oriSparseShape).c_str(),
                    "When layout_q is BSND, ori_sparse_indices's B(" + std::to_string(oriSparseShape.GetDim(DIM_0)) +
                        ") should be equal to q's B(" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                oriSparseShape.GetDim(DIM_1) != s1Size_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_sparse_indices", ToStringRaw(oriSparseShape).c_str(),
                    "When layout_q is BSND, ori_sparse_indices's S1(" + std::to_string(oriSparseShape.GetDim(DIM_1)) +
                        ") should be equal to q's S1(" + std::to_string(s1Size_) + ")"),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                oriSparseShape.GetDim(DIM_0) != qTSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_sparse_indices", ToStringRaw(oriSparseShape).c_str(),
                    "When layout_q is TND, ori_sparse_indices's T1(" + std::to_string(oriSparseShape.GetDim(DIM_0)) +
                        ") should be equal to q's T1(" + std::to_string(qTSize_) + ")"),
                return ge::GRAPH_FAILED);
        }
    }

    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        const auto &cmpSparseShape = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == MQSMLALayout::BSND) {
            OP_CHECK_IF(
                cmpSparseShape.GetDim(DIM_0) != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_sparse_indices", ToStringRaw(cmpSparseShape).c_str(),
                    "When layout_q is BSND, cmp_sparse_indices's B(" + std::to_string(cmpSparseShape.GetDim(DIM_0)) +
                        ") should be equal to q's B(" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                cmpSparseShape.GetDim(DIM_1) != s1Size_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_sparse_indices", ToStringRaw(cmpSparseShape).c_str(),
                    "When layout_q is BSND, cmp_sparse_indices's S1(" + std::to_string(cmpSparseShape.GetDim(DIM_1)) +
                        ") should be equal to q's S1(" + std::to_string(s1Size_) + ")"),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                cmpSparseShape.GetDim(DIM_0) != qTSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_sparse_indices", ToStringRaw(cmpSparseShape).c_str(),
                    "When layout_q is TND, cmp_sparse_indices's T1(" + std::to_string(cmpSparseShape.GetDim(DIM_0)) +
                        ") should be equal to q's T1(" + std::to_string(qTSize_) + ")"),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckTopkLengthConsistency() const
{
    if (opParamInfo_.oriTopkLength.tensor != nullptr) {
        const auto &oriTopkShape = opParamInfo_.oriTopkLength.tensor->GetStorageShape();
        if (qLayout_ == MQSMLALayout::BSND) {
            OP_CHECK_IF(oriTopkShape.GetDim(DIM_0) != bSize_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "ori_topk_length", ToStringRaw(oriTopkShape).c_str(),
                            "When layout_q is BSND, ori_topk_length's B(" + std::to_string(oriTopkShape.GetDim(DIM_0)) +
                                ") should be equal to q's B(" + std::to_string(bSize_) + ")"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                oriTopkShape.GetDim(DIM_1) != s1Size_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_topk_length", ToStringRaw(oriTopkShape).c_str(),
                    "When layout_q is BSND, ori_topk_length's S1(" + std::to_string(oriTopkShape.GetDim(DIM_1)) +
                        ") should be equal to q's S1(" + std::to_string(s1Size_) + ")"),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(oriTopkShape.GetDim(DIM_0) != qTSize_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "ori_topk_length", ToStringRaw(oriTopkShape).c_str(),
                            "When layout_q is TND, ori_topk_length's T1(" + std::to_string(oriTopkShape.GetDim(DIM_0)) +
                                ") should be equal to q's T1(" + std::to_string(qTSize_) + ")"),
                        return ge::GRAPH_FAILED);
        }
    }

    if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
        const auto &cmpTopkShape = opParamInfo_.cmpTopkLength.tensor->GetStorageShape();
        if (qLayout_ == MQSMLALayout::BSND) {
            OP_CHECK_IF(cmpTopkShape.GetDim(DIM_0) != bSize_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "cmp_topk_length", ToStringRaw(cmpTopkShape).c_str(),
                            "When layout_q is BSND, cmp_topk_length's B(" + std::to_string(cmpTopkShape.GetDim(DIM_0)) +
                                ") should be equal to q's B(" + std::to_string(bSize_) + ")"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                cmpTopkShape.GetDim(DIM_1) != s1Size_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_topk_length", ToStringRaw(cmpTopkShape).c_str(),
                    "When layout_q is BSND, cmp_topk_length's S1(" + std::to_string(cmpTopkShape.GetDim(DIM_1)) +
                        ") should be equal to q's S1(" + std::to_string(s1Size_) + ")"),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(cmpTopkShape.GetDim(DIM_0) != qTSize_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "cmp_topk_length", ToStringRaw(cmpTopkShape).c_str(),
                            "When layout_q is TND, cmp_topk_length's T1(" + std::to_string(cmpTopkShape.GetDim(DIM_0)) +
                                ") should be equal to q's T1(" + std::to_string(qTSize_) + ")"),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckN2Consistency() const
{
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        uint32_t cmpKvN2 = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), MQSMLAAxis::N, kvLayout_);
        OP_CHECK_IF(cmpKvN2 != n2Size_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "cmp_kv and ori_kv",
                        Ops::Base::ToString(opParamInfo_.cmpKv.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of ori_kv(" + std::to_string(cmpKvN2) +
                            ") should be equal to the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        const auto &oriSparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        int64_t oriSparseN2 =
            (qLayout_ == MQSMLALayout::BSND) ? oriSparseShape.GetDim(DIM_2) : oriSparseShape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(oriSparseN2) != n2Size_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "ori_sparse_indices and ori_kv",
                        Ops::Base::ToString(oriSparseShape) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of ori_sparse_indices(" + std::to_string(oriSparseN2) +
                            ") should be equal to the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        const auto &cmpSparseShape = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        int64_t cmpSparseN2 =
            (qLayout_ == MQSMLALayout::BSND) ? cmpSparseShape.GetDim(DIM_2) : cmpSparseShape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(cmpSparseN2) != n2Size_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "cmp_sparse_indices and ori_kv",
                        Ops::Base::ToString(cmpSparseShape) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of cmp_sparse_indices(" + std::to_string(cmpSparseN2) +
                            ") should be equal to the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.oriTopkLength.tensor != nullptr) {
        const auto &oriTopkShape = opParamInfo_.oriTopkLength.tensor->GetStorageShape();
        int64_t oriTopkN2 = (qLayout_ == MQSMLALayout::BSND) ? oriTopkShape.GetDim(DIM_2) : oriTopkShape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(oriTopkN2) != n2Size_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "ori_topk_length and ori_kv",
                        Ops::Base::ToString(oriTopkShape) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of ori_topk_length(" + std::to_string(oriTopkN2) +
                            ") should be equal to the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
        const auto &cmpTopkShape = opParamInfo_.cmpTopkLength.tensor->GetStorageShape();
        int64_t cmpTopkN2 = (qLayout_ == MQSMLALayout::BSND) ? cmpTopkShape.GetDim(DIM_2) : cmpTopkShape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(cmpTopkN2) != n2Size_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "cmp_topk_length and ori_kv",
                        Ops::Base::ToString(cmpTopkShape) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of cmp_topk_length(" + std::to_string(cmpTopkN2) +
                            ") should be equal to the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckBConsistency() const
{
    if (opParamInfo_.oriBlockTable.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(DIM_0) != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "ori_block_table", ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                "Ori_block_table's B(" +
                    std::to_string(opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(DIM_0)) +
                    ") should be equal to q's B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(DIM_0) != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "cmp_block_table", ToStringRaw(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                "Cmp_block_table's B(" +
                    std::to_string(opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(DIM_0)) +
                    ") should be equal to q's B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_ + 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cu_seqlens_q", std::to_string(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize()).c_str(),
                        "The shape size of cu_seqlens_q is not equal to B + 1:" + std::to_string(bSize_ + 1)),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cuSeqLensOriKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize() != bSize_ + 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cu_seqlens_ori_kv",
                        std::to_string(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize()).c_str(),
                        "The shape size of cu_seqlens_ori_kv is not equal to B + 1:" + std::to_string(bSize_ + 1)),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cuSeqLensCmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize() != bSize_ + 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cu_seqlens_cmp_kv",
                        std::to_string(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize()).c_str(),
                        "The shape size of cu_seqlens_cmp_kv is not equal to B + 1:" + std::to_string(bSize_ + 1)),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor->GetShapeSize() != bSize_,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "seqused_q", std::to_string(opParamInfo_.seqUsedQ.tensor->GetShapeSize()).c_str(),
                        "The shape size of seqused_q should be equal to B(" + std::to_string(bSize_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.sequsedOriKv.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "seqused_ori_kv", std::to_string(opParamInfo_.sequsedOriKv.tensor->GetShapeSize()).c_str(),
                "The shape size of seqused_ori_kv should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.sequsedCmpKv.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() != bSize_,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "seqused_cmp_kv", std::to_string(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize()).c_str(),
                "The shape size of seqused_cmp_kv should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpResidualKv.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.cmpResidualKv.tensor->GetShapeSize() != bSize_,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cmp_residual_kv", std::to_string(opParamInfo_.cmpResidualKv.tensor->GetShapeSize()).c_str(),
                "The shape size of cmp_residual_kv should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckConsistency() const
{
    if (ge::GRAPH_SUCCESS != CheckLayoutQKvConsistency() || ge::GRAPH_SUCCESS != CheckSparseIndicesShapeMatchQ() ||
        ge::GRAPH_SUCCESS != CheckTopkLengthConsistency() || ge::GRAPH_SUCCESS != CheckN2Consistency() ||
        ge::GRAPH_SUCCESS != CheckBConsistency()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
