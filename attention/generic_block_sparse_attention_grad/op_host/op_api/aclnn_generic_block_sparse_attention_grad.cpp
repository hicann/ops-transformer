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
 * \file aclnn_generic_block_sparse_attention_grad.cpp
 * \brief L2 aclnn API for GenericBlockSparseAttentionGrad.
 */

#include "aclnn_generic_block_sparse_attention_grad.h"
#include "generic_block_sparse_attention_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include <acl/acl.h>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t GSAG_MAX_HEAD_NUM = 128;
static constexpr int64_t GSAG_HEAD_DIM = 128;
static constexpr int64_t GSAG_SUPPORTED_MASK_TYPE = 1;
static constexpr int64_t GSAG_SUPPORTED_SOFTMAX_PRECISION = 0;
static constexpr size_t GSAG_SPARSE_IDX_RANK = 4;
static constexpr size_t GSAG_SPARSE_CNT_RANK = 3;
static constexpr int64_t GSAG_METADATA_HEADER_SIZE = 80;

struct LayoutDims {
    int64_t batch;
    int64_t seq;
    int64_t head;
    int64_t headDim;
};

aclnnStatus CheckRequiredTensor(const aclTensor *t, const char *name)
{
    if (t == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GenericBlockSparseAttentionGrad: %s is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (t->GetViewShape().GetDimNum() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: %s is empty (rank=0 / shape=[]), not supported.", name);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorPositiveDims(const aclTensor *t, const char *name)
{
    const auto &shape = t->GetViewShape();
    const size_t dimNum = shape.GetDimNum();
    for (size_t i = 0; i < dimNum; ++i) {
        const int64_t dim = shape.GetDim(static_cast<int64_t>(i));
        if (dim <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: %s dim[%zu]=%ld must be > 0, shape[%s].",
                    name, i, dim, op::ToString(shape).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLayoutTensorRank(const aclTensor *t, const char *name, const char *layout, size_t expectRank)
{
    const auto &shape = t->GetViewShape();
    const size_t dimNum = shape.GetDimNum();
    if (dimNum != expectRank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: %s rank=%zu must be %zu for layout=%s, shape[%s].", name, dimNum,
                expectRank, layout, op::ToString(shape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

bool IsTndLayout(const char *layout)
{
    return strcmp(layout, "TND") == 0;
}

LayoutDims GetLayoutDims(const char *layout)
{
    if (IsTndLayout(layout)) {
        return {-1, 0, 1, 2}; // [T, N, D]
    }
    if (strcmp(layout, "BNSD") == 0) {
        return {0, 2, 1, 3};
    }
    return {0, 1, 2, 3}; // BSND
}

int64_t GetHeadNum(const aclTensor *t, const char *layout)
{
    const auto &shape = t->GetViewShape();
    const LayoutDims dims = GetLayoutDims(layout);
    return shape.GetDimNum() > static_cast<size_t>(dims.head) ? shape.GetDim(dims.head) : -1;
}

bool IsSupportedLayout(const char *layout)
{
    return layout != nullptr &&
           (strcmp(layout, "TND") == 0 || strcmp(layout, "BNSD") == 0 || strcmp(layout, "BSND") == 0);
}

aclnnStatus CheckQkvShapesByLayout(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                   const aclTensor *dout, const aclTensor *out, const aclTensor *dQuery,
                                   const aclTensor *dKey, const aclTensor *dValue, const char *layout)
{
    const size_t expectQRank = (strcmp(layout, "TND") == 0) ? 3U : 4U;
    const size_t expectKvRank = expectQRank;
    const aclTensor *qLike[] = {query, dout, out, dQuery};
    const char *qLikeNames[] = {"query", "dout", "out", "dQuery"};
    for (size_t i = 0; i < sizeof(qLike) / sizeof(qLike[0]); ++i) {
        aclnnStatus st = CheckLayoutTensorRank(qLike[i], qLikeNames[i], layout, expectQRank);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
        st = CheckTensorPositiveDims(qLike[i], qLikeNames[i]);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    const aclTensor *kvLike[] = {key, value, dKey, dValue};
    const char *kvLikeNames[] = {"key", "value", "dKey", "dValue"};
    for (size_t i = 0; i < sizeof(kvLike) / sizeof(kvLike[0]); ++i) {
        aclnnStatus st = CheckLayoutTensorRank(kvLike[i], kvLikeNames[i], layout, expectKvRank);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
        st = CheckTensorPositiveDims(kvLike[i], kvLikeNames[i]);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShapeEqual(const aclTensor *lhs, const aclTensor *rhs, const char *lhsName, const char *rhsName)
{
    const auto &lhsShape = lhs->GetViewShape();
    const auto &rhsShape = rhs->GetViewShape();
    bool same = lhsShape.GetDimNum() == rhsShape.GetDimNum();
    for (size_t i = 0; same && i < lhsShape.GetDimNum(); ++i) {
        same = lhsShape.GetDim(static_cast<int64_t>(i)) == rhsShape.GetDim(static_cast<int64_t>(i));
    }
    if (!same) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: %s shape[%s] must equal %s shape[%s].",
                lhsName, op::ToString(lhsShape).GetString(), rhsName, op::ToString(rhsShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckCrossTensorShapes(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                   const aclTensor *dout, const aclTensor *out, const aclTensor *dQuery,
                                   const aclTensor *dKey, const aclTensor *dValue, const char *layout)
{
    const struct {
        const aclTensor *lhs;
        const aclTensor *rhs;
        const char *lhsName;
        const char *rhsName;
    } pairs[] = {{dout, query, "dout", "query"}, {out, query, "out", "query"}, {dQuery, query, "dQuery", "query"},
                 {value, key, "value", "key"},   {dKey, key, "dKey", "key"},   {dValue, key, "dValue", "key"}};
    for (const auto &pair : pairs) {
        aclnnStatus st = CheckShapeEqual(pair.lhs, pair.rhs, pair.lhsName, pair.rhsName);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }

    const LayoutDims dims = GetLayoutDims(layout);
    const auto &qShape = query->GetViewShape();
    const auto &kShape = key->GetViewShape();
    if (dims.batch >= 0 && qShape.GetDim(dims.batch) != kShape.GetDim(dims.batch)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: query batch(%ld) must equal key batch(%ld), layout=%s.",
                qShape.GetDim(dims.batch), kShape.GetDim(dims.batch), layout);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t qHeadDim = qShape.GetDim(dims.headDim);
    if (qHeadDim != kShape.GetDim(dims.headDim) || qHeadDim != GSAG_HEAD_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: query/key headDim must both be %ld, got %ld and %ld.", GSAG_HEAD_DIM,
                qHeadDim, kShape.GetDim(dims.headDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLseShape(const aclTensor *lse, const aclTensor *query, const char *layout, int64_t qHeadNum)
{
    const LayoutDims dims = GetLayoutDims(layout);
    const auto &qShape = query->GetViewShape();
    const int64_t qSeqLen = qShape.GetDim(dims.seq);
    const int64_t batch = dims.batch >= 0 ? qShape.GetDim(dims.batch) : 1;
    const auto &lseShape = lse->GetViewShape();
    if (lseShape.GetShapeSize() != batch * qHeadNum * qSeqLen ||
        lseShape.GetDim(0) != (dims.batch >= 0 ? batch : qSeqLen)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: lse shape[%s] must be [T1, N1] for TND or [B, N1, S1] otherwise, "
                "query shape[%s] layout=%s.",
                op::ToString(lseShape).GetString(), op::ToString(qShape).GetString(), layout);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSparseAndSeqShapes(const aclTensor *query, const aclTensor *sparseBlockIdx,
                                    const aclTensor *sparseBlockCount, const aclTensor *metadata,
                                    const aclTensor *cuSeqLengthsQ, const aclTensor *cuSeqLengthsKv,
                                    const aclTensor *sequsedQ, const aclTensor *sequsedKv, const char *layout)
{
    const auto &idxShape = sparseBlockIdx->GetViewShape();
    const auto &cntShape = sparseBlockCount->GetViewShape();
    if (idxShape.GetDimNum() != GSAG_SPARSE_IDX_RANK || cntShape.GetDimNum() != GSAG_SPARSE_CNT_RANK ||
        cntShape.GetDim(0) != idxShape.GetDim(0) || cntShape.GetDim(1) != idxShape.GetDim(1) ||
        cntShape.GetDim(2) != idxShape.GetDim(2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: sparseBlockIdx must be [B, N2, J, maxS1] and sparseBlockCount the "
                "matching [B, N2, J], got idx shape[%s] cnt shape[%s].",
                op::ToString(idxShape).GetString(), op::ToString(cntShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t metaElems = metadata->GetViewShape().GetShapeSize();
    if (metaElems < GSAG_METADATA_HEADER_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: metadata must hold at least %ld int32 elements, got %ld.",
                GSAG_METADATA_HEADER_SIZE, metaElems);
        return ACLNN_ERR_PARAM_INVALID;
    }

    const bool isTnd = IsTndLayout(layout);
    if (isTnd && (cuSeqLengthsQ == nullptr || cuSeqLengthsKv == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: cuSeqLengthsQ and cuSeqLengthsKv are mandatory for layout=TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t batch = isTnd ? idxShape.GetDim(0) : query->GetViewShape().GetDim(0);
    const struct {
        const aclTensor *tensor;
        const char *name;
        DataType dtype;
        int64_t elems;
    } seqTensors[] = {{cuSeqLengthsQ, "cuSeqLengthsQ", DataType::DT_INT64, batch + 1},
                      {cuSeqLengthsKv, "cuSeqLengthsKv", DataType::DT_INT64, batch + 1},
                      {sequsedQ, "sequsedQ", DataType::DT_INT32, batch},
                      {sequsedKv, "sequsedKv", DataType::DT_INT32, batch}};
    for (const auto &seq : seqTensors) {
        if (seq.tensor == nullptr) {
            continue;
        }
        if (seq.tensor->GetDataType() != seq.dtype || seq.tensor->GetViewShape().GetShapeSize() != seq.elems) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: %s must be %s with %ld elements, got %s shape[%s].", seq.name,
                    op::ToString(seq.dtype).GetString(), seq.elems, op::ToString(seq.tensor->GetDataType()).GetString(),
                    op::ToString(seq.tensor->GetViewShape()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Validate(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout,
                     const aclTensor *out, const aclTensor *lse, const aclTensor *sparseBlockIdx,
                     const aclTensor *sparseBlockCount, const aclTensor *metadataOptional,
                     const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsQOptional,
                     const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional,
                     const aclTensor *sequsedKvOptional, const aclIntArray *blockShape, int64_t isPackedGQA,
                     char *layoutQ, char *layoutKv, int64_t maskType, int64_t softmaxPrecision, int64_t winLeft,
                     int64_t winRight, const aclTensor *dQuery, const aclTensor *dKey, const aclTensor *dValue)
{
    const aclTensor *required[] = {
        query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadataOptional, dQuery, dKey, dValue};
    const char *requiredNames[] = {"query",    "key",    "value",          "dout",
                                   "out",      "lse",    "sparseBlockIdx", "sparseBlockCount",
                                   "metadata", "dQuery", "dKey",           "dValue"};
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        aclnnStatus st = CheckRequiredTensor(required[i], requiredNames[i]);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    if (layoutQ == nullptr || layoutKv == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GenericBlockSparseAttentionGrad: layoutQ and layoutKv must not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strcmp(layoutQ, layoutKv) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: layoutQ(%s) must equal layoutKv(%s).",
                layoutQ, layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsSupportedLayout(layoutQ)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: layout must be TND/BSND/BNSD, got layoutQ=%s.", layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    {
        aclnnStatus st = CheckQkvShapesByLayout(query, key, value, dout, out, dQuery, dKey, dValue, layoutQ);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    {
        const aclTensor *otherTensors[] = {lse, sparseBlockIdx, sparseBlockCount, metadataOptional};
        const char *otherNames[] = {"lse", "sparseBlockIdx", "sparseBlockCount", "metadata"};
        for (size_t i = 0; i < sizeof(otherTensors) / sizeof(otherTensors[0]); ++i) {
            aclnnStatus st = CheckTensorPositiveDims(otherTensors[i], otherNames[i]);
            if (st != ACLNN_SUCCESS) {
                return st;
            }
        }
    }
    {
        aclnnStatus st = CheckCrossTensorShapes(query, key, value, dout, out, dQuery, dKey, dValue, layoutQ);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    if (isPackedGQA != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: only support isPackedGQA == 1, got %ld.",
                isPackedGQA);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maskType != GSAG_SUPPORTED_MASK_TYPE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: only support maskType == %ld, got %ld.",
                GSAG_SUPPORTED_MASK_TYPE, maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (softmaxPrecision != GSAG_SUPPORTED_SOFTMAX_PRECISION) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: only support softmaxPrecision == %ld, got %ld.",
                GSAG_SUPPORTED_SOFTMAX_PRECISION, softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (winLeft != -1 || winRight != -1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: only support winLeft == -1 and winRight == -1, got %ld, %ld.",
                winLeft, winRight);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (attenMaskOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: attenMask must be nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockShape != nullptr) {
        if (blockShape->Size() < 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: blockShape must contain [x, y], got size %zu.",
                    blockShape->Size());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*blockShape)[0] != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: only support blockShape[0] == 1, got %ld.", (*blockShape)[0]);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*blockShape)[1] < 128 || ((*blockShape)[1] % 64 != 0)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "GenericBlockSparseAttentionGrad: blockShape[1] must be >= 128 and 64-aligned, got %ld.",
                    (*blockShape)[1]);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t qHeadNum = GetHeadNum(query, layoutQ);
    const int64_t kvHeadNum = GetHeadNum(key, layoutKv);
    if (qHeadNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: failed to get qHeadNum from query shape[%s] with layout=%s.",
                op::ToString(query->GetViewShape()).GetString(), layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qHeadNum > GSAG_MAX_HEAD_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: qHeadNum=%ld must be in [1, %ld], layout=%s, query shape[%s].",
                qHeadNum, GSAG_MAX_HEAD_NUM, layoutQ, op::ToString(query->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kvHeadNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: failed to get kvHeadNum from key shape[%s] with layout=%s.",
                op::ToString(key->GetViewShape()).GetString(), layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kvHeadNum > GSAG_MAX_HEAD_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: kvHeadNum=%ld must be in [1, %ld], layout=%s, key shape[%s].",
                kvHeadNum, GSAG_MAX_HEAD_NUM, layoutKv, op::ToString(key->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qHeadNum % kvHeadNum != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: qHeadNum(%ld) must be divisible by kvHeadNum(%ld).", qHeadNum,
                kvHeadNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    {
        aclnnStatus st = CheckLseShape(lse, query, layoutQ, qHeadNum);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
        st = CheckSparseAndSeqShapes(query, sparseBlockIdx, sparseBlockCount, metadataOptional, cuSeqLengthsQOptional,
                                     cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, layoutQ);
        if (st != ACLNN_SUCCESS) {
            return st;
        }
    }
    DataType qDtype = query->GetDataType();
    if (qDtype != ACL_FLOAT16 && qDtype != ACL_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: query dtype must be FP16 or BF16, got %s.",
                op::ToString(qDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (key->GetDataType() != qDtype || value->GetDataType() != qDtype || dout->GetDataType() != qDtype ||
        out->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: key/value/dout/out dtype must match query (%s), "
                "got key=%s value=%s dout=%s out=%s.",
                op::ToString(qDtype).GetString(), op::ToString(key->GetDataType()).GetString(),
                op::ToString(value->GetDataType()).GetString(), op::ToString(dout->GetDataType()).GetString(),
                op::ToString(out->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dQuery->GetDataType() != qDtype || dKey->GetDataType() != qDtype || dValue->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: dQuery/dKey/dValue dtype must match query (%s), "
                "got dQuery=%s dKey=%s dValue=%s.",
                op::ToString(qDtype).GetString(), op::ToString(dQuery->GetDataType()).GetString(),
                op::ToString(dKey->GetDataType()).GetString(), op::ToString(dValue->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (lse->GetDataType() != ACL_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: lse dtype must be FP32, got %s.",
                op::ToString(lse->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseBlockIdx->GetDataType() != ACL_INT32 || sparseBlockCount->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: sparseBlockIdx and sparseBlockCount dtype must be INT32, "
                "got idx=%s cnt=%s.",
                op::ToString(sparseBlockIdx->GetDataType()).GetString(),
                op::ToString(sparseBlockCount->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadataOptional->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad: metadata dtype must be INT32, got %s.",
                op::ToString(metadataOptional->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t deterministicLevel = 0;
    if (aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicLevel) == ACL_SUCCESS && deterministicLevel != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GenericBlockSparseAttentionGrad: deterministic computation is not supported, got level %ld.",
                deterministicLevel);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout, const aclTensor *out,
    const aclTensor *lse, const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional, const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclIntArray *blockShape, int64_t isPackedGQA, char *layoutQ, char *layoutKv, double scaleValue,
    int64_t maskType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, aclTensor *dQuery, aclTensor *dKey,
    aclTensor *dValue, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(
        aclnnGenericBlockSparseAttentionGrad,
        DFX_IN(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadataOptional, attenMaskOptional,
               cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, blockShape,
               isPackedGQA, layoutQ, layoutKv, scaleValue, maskType, softmaxPrecision, winLeft, winRight),
        DFX_OUT(dQuery, dKey, dValue));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = Validate(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadataOptional,
                        attenMaskOptional, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional,
                        sequsedKvOptional, blockShape, isPackedGQA, layoutQ, layoutKv, maskType, softmaxPrecision,
                        winLeft, winRight, dQuery, dKey, dValue);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto queryC = l0op::Contiguous(query, uniqueExecutor.get());
    auto keyC = l0op::Contiguous(key, uniqueExecutor.get());
    auto valueC = l0op::Contiguous(value, uniqueExecutor.get());
    auto doutC = l0op::Contiguous(dout, uniqueExecutor.get());
    auto outC = l0op::Contiguous(out, uniqueExecutor.get());
    auto lseC = l0op::Contiguous(lse, uniqueExecutor.get());
    auto idxC = l0op::Contiguous(sparseBlockIdx, uniqueExecutor.get());
    auto cntC = l0op::Contiguous(sparseBlockCount, uniqueExecutor.get());
    auto metaC = l0op::Contiguous(metadataOptional, uniqueExecutor.get());
    CHECK_RET(queryC && keyC && valueC && doutC && outC && lseC && idxC && cntC && metaC, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *attenC = nullptr;
    const aclTensor *cuQC = nullptr;
    if (cuSeqLengthsQOptional != nullptr) {
        cuQC = l0op::Contiguous(cuSeqLengthsQOptional, uniqueExecutor.get());
        CHECK_RET(cuQC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuKvC = nullptr;
    if (cuSeqLengthsKvOptional != nullptr) {
        cuKvC = l0op::Contiguous(cuSeqLengthsKvOptional, uniqueExecutor.get());
        CHECK_RET(cuKvC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedQC = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQC = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        CHECK_RET(sequsedQC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedKvC = nullptr;
    if (sequsedKvOptional != nullptr) {
        sequsedKvC = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
        CHECK_RET(sequsedKvC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outs = l0op::GenericBlockSparseAttentionGrad(queryC, keyC, valueC, doutC, outC, lseC, idxC, cntC, metaC,
                                                      attenC, cuQC, cuKvC, sequsedQC, sequsedKvC, blockShape,
                                                      isPackedGQA, layoutQ, layoutKv, scaleValue, maskType,
                                                      softmaxPrecision, winLeft, winRight, uniqueExecutor.get());
    CHECK_RET(outs[0] != nullptr && outs[1] != nullptr && outs[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dqView = l0op::ViewCopy(outs[0], dQuery, uniqueExecutor.get());
    auto dkView = l0op::ViewCopy(outs[1], dKey, uniqueExecutor.get());
    auto dvView = l0op::ViewCopy(outs[2], dValue, uniqueExecutor.get());
    CHECK_RET(dqView != nullptr && dkView != nullptr && dvView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGrad(void *workspace,
                                                                                        uint64_t workspaceSize,
                                                                                        aclOpExecutor *executor,
                                                                                        const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGenericBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
