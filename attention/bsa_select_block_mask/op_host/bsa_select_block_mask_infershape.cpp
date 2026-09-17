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
 * \file bsa_select_block_mask_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {
enum InputIdx {
    queryEnum = 0,
    keyEnum,
    blockShapeEnum,
    postBlockShapeEnum,
    actualSeqLensQEnum,
    actualSeqLensKvEnum,
    actualBlockLenQEnum,
    actualBlockLenKvEnum
};

enum OutputIdx {
    maskOutEnum = 0
};

static constexpr uint32_t TND_DIM_NUM = 3;
static constexpr uint32_t BNSD_DIM_NUM = 4;

// 校验 actual_seq_lengths(_kv) 的 shape 与常量数据合法性；
static bool GetValidatedMaxSeqLen(const gert::Shape *lensShape, const gert::Tensor *lensTensor, int64_t totalTokens,
                                  const char *inputName, int64_t &maxSeqLen)
{
    int64_t batchCnt = lensShape->GetShapeSize();
    const int64_t *lensData = (lensTensor != nullptr) ? lensTensor->GetData<int64_t>() : nullptr;
    if (lensData == nullptr) {
        OP_LOGE("BSASelectBlockMask", "TND layout requires const %s, but its data is not readable at compile time.",
                inputName);
        return false;
    }
    int64_t sumLen = 0;
    for (int64_t i = 0; i < batchCnt; i++) {
        sumLen += lensData[i];
        if (lensData[i] > maxSeqLen) {
            maxSeqLen = lensData[i];
        }
    }
    if (sumLen != totalTokens) {
        OP_LOGE("BSASelectBlockMask", "Invalid %s: sum[%ld] is not equal to total tokens[%ld].", inputName, sumLen,
                totalTokens);
        return false;
    }
    return true;
}

ge::graphStatus InferShapeBSASelectBlockMask(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BSASelectBlockMask", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeBSASelectBlockMask");

    const gert::Shape *queryShape = context->GetInputShape(queryEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(keyEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *maskShape = context->GetOutputShape(maskOutEnum);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    auto attrs = context->GetAttrs();
    size_t idx = 0;
    auto qLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto kvLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    if (qLayoutPtr == nullptr || kvLayoutPtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrPointer of q_input_layout or kv_input_layout failed.");
        return ge::GRAPH_FAILED;
    }

    int64_t sqLen = 0;
    int64_t skvLen = 0;
    int64_t batchSize = 0;
    int64_t numHeads = 0;

    // 校验 layout：仅支持 BNSD/TND，且 Q 与 KV 的 layout 必须一致
    // （参照 BlockSparseAttention InferShape 的校验方式，比较实际值而非长度）
    std::string qLayout(qLayoutPtr);
    std::string kvLayout(kvLayoutPtr);
    if (qLayout != "BNSD" && qLayout != "TND") {
        OP_LOGE(context->GetNodeName(), "Unsupported q_input_layout: %s. Only BNSD/TND are supported.", qLayoutPtr);
        return ge::GRAPH_FAILED;
    }
    if (qLayout != kvLayout) {
        OP_LOGE(context->GetNodeName(),
                "The parameters q_input_layout and kv_input_layout must be consistent, but currently q_input_layout "
                "is %s and kv_input_layout is %s.",
                qLayoutPtr, kvLayoutPtr);
        return ge::GRAPH_FAILED;
    }

    if (qLayout == "BNSD") {
        // BNSD：[B, N, S, D]；sqLen/skvLen 取 padded 维度，与 tiling AnalyzeLayout 一致
        if (queryShape->GetDimNum() != BNSD_DIM_NUM || keyShape->GetDimNum() != BNSD_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD, queryDims(%zu) and keyDims(%zu) must be 4!",
                    queryShape->GetDimNum(), keyShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        batchSize = queryShape->GetDim(0);
        numHeads = queryShape->GetDim(1);
        sqLen = queryShape->GetDim(2);
        skvLen = keyShape->GetDim(2);
    } else {
        // TND：[T, N, D]；actual_seq_lengths(_kv) 均必传，batch 取元素个数
        if (queryShape->GetDimNum() != TND_DIM_NUM || keyShape->GetDimNum() != TND_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "Layout TND, queryDims(%zu) and keyDims(%zu) must be 3!",
                    queryShape->GetDimNum(), keyShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        numHeads = queryShape->GetDim(1);

        // Q 侧：actual_seq_lengths 必传
        const gert::Shape *seqLensQShape = context->GetInputShape(actualSeqLensQEnum);
        if (seqLensQShape == nullptr || seqLensQShape->GetDimNum() == 0) {
            OP_LOGE(context->GetNodeName(), "TND layout requires actual_seq_lengths, but it is not provided.");
            return ge::GRAPH_FAILED;
        }
        const gert::Tensor *seqLensQTensor = context->GetInputTensor(actualSeqLensQEnum);
        int64_t totalQTokens = queryShape->GetDim(0);
        int64_t maxQSeqLen = 0;
        if (!GetValidatedMaxSeqLen(seqLensQShape, seqLensQTensor, totalQTokens, "actual_seq_lengths", maxQSeqLen)) {
            return ge::GRAPH_FAILED;
        }
        batchSize = seqLensQShape->GetShapeSize();
        sqLen = (maxQSeqLen > 0) ? maxQSeqLen : totalQTokens;

        // KV 侧：与 Q 校验思路一致，actual_seq_lengths_kv 同样必传
        const gert::Shape *seqLensKvShape = context->GetInputShape(actualSeqLensKvEnum);
        if (seqLensKvShape == nullptr || seqLensKvShape->GetDimNum() == 0) {
            OP_LOGE(context->GetNodeName(), "TND layout requires actual_seq_lengths_kv, but it is not provided.");
            return ge::GRAPH_FAILED;
        }
        const gert::Tensor *seqLensKvTensor = context->GetInputTensor(actualSeqLensKvEnum);
        int64_t totalKvTokens = keyShape->GetDim(0);
        int64_t maxKvSeqLen = 0;
        if (!GetValidatedMaxSeqLen(seqLensKvShape, seqLensKvTensor, totalKvTokens, "actual_seq_lengths_kv",
                                   maxKvSeqLen)) {
            return ge::GRAPH_FAILED;
        }
        if (seqLensKvShape->GetShapeSize() != batchSize) {
            OP_LOGE(context->GetNodeName(),
                    "Batch mismatch: actual_seq_lengths batch[%ld] vs actual_seq_lengths_kv batch[%ld].", batchSize,
                    seqLensKvShape->GetShapeSize());
            return ge::GRAPH_FAILED;
        }
        skvLen = (maxKvSeqLen > 0) ? maxKvSeqLen : totalKvTokens;
    }

    // 块尺寸默认 [128,128]（与 tiling 默认值一致）；block_shape 为 [2] 常量输入时从张量取值
    int64_t blockShapeX = 128;
    int64_t blockShapeY = 128;
    const gert::Shape *blockShapeShape = context->GetInputShape(blockShapeEnum);
    if (blockShapeShape != nullptr && blockShapeShape->GetDimNum() != 0) {
        const gert::Tensor *blockShapeTensor = context->GetInputTensor(blockShapeEnum);
        if (blockShapeTensor != nullptr) {
            const int64_t *blockShapeData = blockShapeTensor->GetData<int64_t>();
            if (blockShapeData != nullptr && blockShapeData[0] > 0 && blockShapeData[1] > 0) {
                blockShapeX = blockShapeData[0];
                blockShapeY = blockShapeData[1];
            }
        }
    }

    // 默认输出为 fine 块级 [B,N,Xblocks,Yblocks]；启用 post_block_shape 时输出为粗粒度
    // [B,N,postXBlocks,postYBlocks]（二次 pooling 语义：TopK 在粗粒度分数上选择并直接输出）
    int64_t maskDimX = (sqLen + blockShapeX - 1) / blockShapeX;
    int64_t maskDimY = (skvLen + blockShapeY - 1) / blockShapeY;
    const gert::Shape *postBlockShapeShape = context->GetInputShape(postBlockShapeEnum);
    if (postBlockShapeShape != nullptr && postBlockShapeShape->GetDimNum() != 0) {
        const gert::Tensor *postBlockShapeTensor = context->GetInputTensor(postBlockShapeEnum);
        if (postBlockShapeTensor != nullptr) {
            const int64_t *postData = postBlockShapeTensor->GetData<int64_t>();
            if (postData != nullptr && postData[0] > 0 && postData[1] > 0) {
                maskDimX = (maskDimX + postData[0] - 1) / postData[0];
                maskDimY = (maskDimY + postData[1] - 1) / postData[1];
            }
        }
    }

    maskShape->SetDimNum(4);
    maskShape->SetDim(0, batchSize);
    maskShape->SetDim(1, numHeads);
    maskShape->SetDim(2, maskDimX);
    maskShape->SetDim(3, maskDimY);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeBSASelectBlockMask");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeBSASelectBlockMask(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BSASelectBlockMask", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeBSASelectBlockMask");
    context->SetOutputDataType(maskOutEnum, DT_INT8);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeBSASelectBlockMask");
    return GRAPH_SUCCESS;
}

IMPL_OP(BSASelectBlockMask).InferShape(InferShapeBSASelectBlockMask).InferDataType(InferDataTypeBSASelectBlockMask);
} // namespace ops
