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
 * \file quant_sparse_flash_mla_check_consistency.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"

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

ge::graphStatus QSMLATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
                                                        const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(opName_, name.c_str(), QSMLADataTypeToSerialString(actualDtype).c_str(),
                                  QSMLADataTypeToSerialString(expectDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                      const QSMLALayout &layout, const std::string &name) const
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of query is %s, %s must be provided.", QSMLALayoutToSerialString(layout).c_str(),
                name.c_str());
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

ge::graphStatus QSMLATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
                                                   const QSMLATilingShapeCompareParam &param,
                                                   const QSMLALayout &layout) const
{
    if (layout == QSMLALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == QSMLALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == QSMLALayout::PA_BBND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", QSMLALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CompareShape(QSMLATilingShapeCompareParam &param, const gert::Shape &shape,
                                               const QSMLALayout &layout, const std::string &name) const
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
            OP_LOGE(opName_, "%s layout is %s, shape is %s, expected shape is %s.", name.c_str(),
                    QSMLALayoutToSerialString(layout).c_str(), QSMLAGetShapeStr(shape).c_str(),
                    QSMLAGetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void QSMLATilingCheck::SetQSMLAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.q.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.cmpSparseIndices.tensor->GetShape().GetStorageShape();
    keyShapeCmp_ = opParamInfo_.oriKv.tensor->GetShape().GetStorageShape();
    valueShapeCmp_ = opParamInfo_.cmpKv.tensor->GetShape().GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attnOut.shape->GetStorageShape();
}

ge::graphStatus QSMLATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, ORI_BLOCK_TABLE_NAME.c_str(),
                        Ops::Base::ToString(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                        "when the layout_kv is " + QSMLALayoutToSerialString(kvLayout_) + ", " + ORI_BLOCK_TABLE_NAME +
                            " should be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, CMP_BLOCK_TABLE_NAME.c_str(),
                        Ops::Base::ToString(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                        "when the layout_kv is " + QSMLALayoutToSerialString(kvLayout_) + ", " + CMP_BLOCK_TABLE_NAME +
                            " should be null"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t oriBlockTableBatch = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(oriBlockTableBatch != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "oriBlockTableBatch",
                    Ops::Base::ToString(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
                    "oriBlockTableBatch's first dimension(" + std::to_string(oriBlockTableBatch) +
                        ") should be equal to batch size" + std::to_string(bSize_)),
                return ge::GRAPH_FAILED);

    uint32_t cmpBlockTableBatch = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(cmpBlockTableBatch != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmpBlockTableBatch",
                    Ops::Base::ToString(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                    "cmpBlockTableBatch's first dimension(" + std::to_string(cmpBlockTableBatch) +
                        ") should be equal to batch size(" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckTopkShape() { return ge::GRAPH_SUCCESS; }

ge::graphStatus QSMLATilingCheck::CheckAttenOutShape()
{
    QSMLATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = 512; // 512:输出的head_dim
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attnOut.desc->GetDataType(), qType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckKVShapeForBatchContinuous()
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

uint32_t QSMLATilingCheck::GetTypeSize(ge::DataType dtype) const
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

ge::graphStatus QSMLATilingCheck::CheckKVShapeForPageAttention()
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

ge::graphStatus QSMLATilingCheck::CheckKVShape()
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

ge::graphStatus QSMLATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(cmpKvType_, oriKvType_, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.cuSeqLensQ.desc == nullptr) {
        OP_LOGE(opName_, "cuSeqLensQ is not empty,"
                         "but cuSeqLensQ's dtype is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.cuSeqLensQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "cuSeqLensQ's dtype is %s, it should be DT_INT32.",
                QSMLADataTypeToSerialString(opParamInfo_.cuSeqLensQ.desc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cuSeqLensQ") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_ + 1) {
        OP_LOGE(opName_, "cuSeqLensQ shape size is %u, it should be equal to batch size[%u]", shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckActualSeqLensDType()
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
            opName_, "sequsedOriKv", QSMLADataTypeToSerialString(opParamInfo_.sequsedOriKv.desc->GetDataType()).c_str(),
            "sequsedOriKv's dtype should be DT_INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckActualSeqLensShape()
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

ge::graphStatus QSMLATilingCheck::CheckMultiParaConsistency()
{
    SetQSMLAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() || ge::GRAPH_SUCCESS != CheckTopK() || ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() || ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckLayoutQKvConsistency() const
{
    if (kvLayout_ != QSMLALayout::PA_BBND) {
        OP_CHECK_IF(qLayout_ != kvLayout_,
                    OP_LOGE(opName_, "When kv layout is not PA_BBND, q layout(%s) must be same as kv layout(%s).",
                            QSMLALayoutToSerialString(qLayout_).c_str(), QSMLALayoutToSerialString(kvLayout_).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSparseIndicesShapeMatchQ() const
{
    if (perfMode_ != QSMLATemplateMode::CSA_TEMPLATE_MODE && perfMode_ != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        perfMode_ != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }

    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        const auto &oriSparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == QSMLALayout::BSND) {
            OP_CHECK_IF(
                oriSparseShape.GetDim(DIM_0) != bSize_,
                OP_LOGE(opName_, "When q layout is BSND, oriSparseIndices's B(%ld) should be equal to q's B(%u).",
                        oriSparseShape.GetDim(DIM_0), bSize_),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                oriSparseShape.GetDim(DIM_1) != s1Size_,
                OP_LOGE(opName_, "When q layout is BSND, oriSparseIndices's S1(%ld) should be equal to q's S1(%u).",
                        oriSparseShape.GetDim(DIM_1), s1Size_),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                oriSparseShape.GetDim(DIM_0) != qTSize_,
                OP_LOGE(opName_, "When q layout is TND, oriSparseIndices's T1(%ld) should be equal to q's T1(%u).",
                        oriSparseShape.GetDim(DIM_0), qTSize_),
                return ge::GRAPH_FAILED);
        }
    }

    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        const auto &cmpSparseShape = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == QSMLALayout::BSND) {
            OP_CHECK_IF(
                cmpSparseShape.GetDim(DIM_0) != bSize_,
                OP_LOGE(opName_, "When q layout is BSND, cmpSparseIndices's B(%ld) should be equal to q's B(%u).",
                        cmpSparseShape.GetDim(DIM_0), bSize_),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                cmpSparseShape.GetDim(DIM_1) != s1Size_,
                OP_LOGE(opName_, "When q layout is BSND, cmpSparseIndices's S1(%ld) should be equal to q's S1(%u).",
                        cmpSparseShape.GetDim(DIM_1), s1Size_),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                cmpSparseShape.GetDim(DIM_0) != qTSize_,
                OP_LOGE(opName_, "When q layout is TND, cmpSparseIndices's T1(%ld) should be equal to q's T1(%u).",
                        cmpSparseShape.GetDim(DIM_0), qTSize_),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckTopkLengthConsistency() const
{
    if (opParamInfo_.oriTopkLength.tensor != nullptr) {
        const auto &oriTopkShape = opParamInfo_.oriTopkLength.tensor->GetStorageShape();
        if (qLayout_ == QSMLALayout::BSND) {
            OP_CHECK_IF(oriTopkShape.GetDim(DIM_0) != bSize_,
                        OP_LOGE(opName_, "When q layout is BSND, oriTopkLength's B(%ld) should be equal to q's B(%u).",
                                oriTopkShape.GetDim(DIM_0), bSize_),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(oriTopkShape.GetDim(DIM_1) != s1Size_,
                        OP_LOGE(opName_, "oriTopkLength's S1(%ld) should be equal to q's S1(%u).",
                                oriTopkShape.GetDim(DIM_1), s1Size_),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(oriTopkShape.GetDim(DIM_0) != qTSize_,
                        OP_LOGE(opName_, "When q layout is TND, oriTopkLength's T1(%ld) should be equal to q's T1(%u).",
                                oriTopkShape.GetDim(DIM_0), qTSize_),
                        return ge::GRAPH_FAILED);
        }
    }

    if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
        const auto &cmpTopkShape = opParamInfo_.cmpTopkLength.tensor->GetStorageShape();
        if (qLayout_ == QSMLALayout::BSND) {
            OP_CHECK_IF(cmpTopkShape.GetDim(DIM_0) != bSize_,
                        OP_LOGE(opName_, "When q layout is BSND, cmpTopkLength's B(%ld) should be equal to q's B(%u).",
                                cmpTopkShape.GetDim(DIM_0), bSize_),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(cmpTopkShape.GetDim(DIM_1) != s1Size_,
                        OP_LOGE(opName_, "cmpTopkLength's S1(%ld) should be equal to q's S1(%u).",
                                cmpTopkShape.GetDim(DIM_1), s1Size_),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(cmpTopkShape.GetDim(DIM_0) != qTSize_,
                        OP_LOGE(opName_, "When q layout is TND, cmpTopkLength's T1(%ld) should be equal to q's T1(%u).",
                                cmpTopkShape.GetDim(DIM_0), qTSize_),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckN2Consistency() const
{
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        uint32_t cmpKvN2 = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), QSMLAAxis::N, kvLayout_);
        OP_CHECK_IF(cmpKvN2 != n2Size_,
                    OP_LOGE(opName_, "cmpKv's N2(%u) should be equal to oriKv's N2(%u).", cmpKvN2, n2Size_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        const auto &oriSparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        int64_t oriSparseN2 =
            (qLayout_ == QSMLALayout::BSND) ? oriSparseShape.GetDim(DIM_2) : oriSparseShape.GetDim(DIM_1);
        OP_CHECK_IF(
            static_cast<uint32_t>(oriSparseN2) != n2Size_,
            OP_LOGE(opName_, "oriSparseIndices's N2(%ld) should be equal to oriKv's N2(%u).", oriSparseN2, n2Size_),
            return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        const auto &cmpSparseShape = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        int64_t cmpSparseN2 =
            (qLayout_ == QSMLALayout::BSND) ? cmpSparseShape.GetDim(DIM_2) : cmpSparseShape.GetDim(DIM_1);
        OP_CHECK_IF(
            static_cast<uint32_t>(cmpSparseN2) != n2Size_,
            OP_LOGE(opName_, "cmpSparseIndices's N2(%ld) should be equal to oriKv's N2(%u).", cmpSparseN2, n2Size_),
            return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.oriTopkLength.tensor != nullptr) {
        const auto &oriTopkShape = opParamInfo_.oriTopkLength.tensor->GetStorageShape();
        int64_t oriTopkN2 = (qLayout_ == QSMLALayout::BSND) ? oriTopkShape.GetDim(DIM_2) : oriTopkShape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(oriTopkN2) != n2Size_,
                    OP_LOGE(opName_, "oriTopkLength's N2(%ld) should be equal to oriKv's N2(%u).", oriTopkN2, n2Size_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
        const auto &cmpTopkShape = opParamInfo_.cmpTopkLength.tensor->GetStorageShape();
        int64_t cmpTopkN2 = (qLayout_ == QSMLALayout::BSND) ? cmpTopkShape.GetDim(DIM_2) : cmpTopkShape.GetDim(DIM_1);
        OP_CHECK_IF(static_cast<uint32_t>(cmpTopkN2) != n2Size_,
                    OP_LOGE(opName_, "cmpTopkLength's N2(%ld) should be equal to oriKv's N2(%u).", cmpTopkN2, n2Size_),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckBConsistency() const
{
    if (opParamInfo_.oriBlockTable.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(DIM_0) != bSize_,
                    OP_LOGE(opName_, "oriBlockTable's B(%ld) should be equal to q's B(%u).",
                            opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(DIM_0), bSize_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(DIM_0) != bSize_,
                    OP_LOGE(opName_, "cmpBlockTable's B(%ld) should be equal to q's B(%u).",
                            opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(DIM_0), bSize_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_ + 1,
                    OP_LOGE(opName_, "cuSeqLensQ's shapeSize(%ld) should be equal to B+1(%u).",
                            opParamInfo_.cuSeqLensQ.tensor->GetShapeSize(), bSize_ + 1),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cuSeqLensOriKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize() != bSize_ + 1,
                    OP_LOGE(opName_, "cuSeqLensOriKv's shapeSize(%ld) should be equal to B+1(%u).",
                            opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize(), bSize_ + 1),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cuSeqLensCmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize() != bSize_ + 1,
                    OP_LOGE(opName_, "cuSeqLensCmpKv's shapeSize(%ld) should be equal to B+1(%u).",
                            opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize(), bSize_ + 1),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor->GetShapeSize() != bSize_,
                    OP_LOGE(opName_, "seqUsedQ's shapeSize(%ld) should be equal to B(%u).",
                            opParamInfo_.seqUsedQ.tensor->GetShapeSize(), bSize_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.sequsedOriKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_,
                    OP_LOGE(opName_, "sequsedOriKv's shapeSize(%ld) should be equal to B(%u).",
                            opParamInfo_.sequsedOriKv.tensor->GetShapeSize(), bSize_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.sequsedCmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() != bSize_,
                    OP_LOGE(opName_, "sequsedCmpKv's shapeSize(%ld) should be equal to B(%u).",
                            opParamInfo_.sequsedCmpKv.tensor->GetShapeSize(), bSize_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpResidualKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpResidualKv.tensor->GetShapeSize() != bSize_,
                    OP_LOGE(opName_, "cmpResidualKv's shapeSize(%ld) should be equal to B(%u).",
                            opParamInfo_.cmpResidualKv.tensor->GetShapeSize(), bSize_),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckConsistency() const
{
    if (ge::GRAPH_SUCCESS != CheckLayoutQKvConsistency() || ge::GRAPH_SUCCESS != CheckSparseIndicesShapeMatchQ() ||
        ge::GRAPH_SUCCESS != CheckTopkLengthConsistency() || ge::GRAPH_SUCCESS != CheckN2Consistency() ||
        ge::GRAPH_SUCCESS != CheckBConsistency()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
