/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file dense_lightning_indexer_kl_loss_grad_tiling_general_regbase.cpp
 * \brief
 */

#include "dense_lightning_indexer_kl_loss_grad_tiling_general_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include <tiling/tiling_api.h>

using namespace ge;
using namespace AscendC;

namespace optiling {
static const int64_t PING_PONG_VALUE = 2L;
static const int64_t THREE_BUFFER_VALUE = 3L;
static const int64_t GM_ALIGN = 512;
static const int32_t FRACTAL_NUM = 16L;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t B32 = 4; // 4B
static constexpr uint32_t B16 = 2;
static constexpr uint32_t BASE_LEN_256 = 256;
static constexpr uint32_t BLOCK_SIZE = 128;

static constexpr size_t SIZE_1 = 1;
static constexpr size_t SIZE_128 = 128;
static const uint64_t DIM_NUM_0 = 0;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;

static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_33K = 33 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_128K = 128 * 1024;

static constexpr uint32_t NQUERY_SIZE_8 = 8;
static constexpr uint32_t NQUERY_SIZE_16 = 16;
static constexpr uint32_t NQUERY_SIZE_32 = 32;
static constexpr uint32_t NQUERY_SIZE_64 = 64;
static constexpr uint32_t NQUERY_SIZE_128 = 128;

static constexpr uint32_t NQUERYINDEX_SIZE_1 = 1;
static constexpr uint32_t NQUERYINDEX_SIZE_8 = 8;
static constexpr uint32_t NQUERYINDEX_SIZE_16 = 16;
static constexpr uint32_t NQUERYINDEX_SIZE_32 = 32;
static constexpr uint32_t NQUERYINDEX_SIZE_64 = 64;
static constexpr uint32_t NQUERYINDEX_SIZE_128 = 128;

static constexpr uint32_t NKEYINDEX_SIZE_1 = 1;
static constexpr uint32_t D_SIZE_512 = 512;
static constexpr uint32_t DINDEX_SIZE_128 = 128;
static constexpr uint32_t TOPK_SIZE_2048 = 2048;
static constexpr uint32_t CMP_RATIO_1 = 1;
static constexpr uint32_t CMP_RATIO_128 = 128;
static constexpr uint32_t DETER_MODE_0 = 0;
static constexpr uint32_t DETER_MODE_2 = 2;
static constexpr int64_t WINDOWS_INF = -1;
static constexpr int64_t SPARSE_MODE_SIZE_3 = 3;
static constexpr int64_t SPARSE_MODE_SIZE_0 = 0;
static constexpr int64_t SPARSE_COUNT_SIZE_2048 = 2048;

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::ValidateRequiredInputs()
{
    auto queryDesc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    auto keyDesc = context_->GetInputDesc(KEY_INPUT_INDEX);
    auto weightDesc = context_->GetInputDesc(WEIGHT_INPUT_INDEX);
    auto attnSoftmaxL1NormDesc = context_->GetInputDesc(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX);
    auto lseDesc = context_->GetInputDesc(SOFTMAX_LSE_INPUT_INDEX);

    OP_CHECK_IF(queryDesc == nullptr, OP_LOGE(opName, "query must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDesc == nullptr, OP_LOGE(opName, "key must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightDesc == nullptr, OP_LOGE(opName, "weight must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(attnSoftmaxL1NormDesc == nullptr, OP_LOGE(opName, "attnSoftmaxL1Norm must be provided."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(lseDesc == nullptr, OP_LOGE(opName, "lse must be provided."), return ge::GRAPH_FAILED);

    auto metadataTensor = context_->GetOptionalInputTensor(METADATA_INPUT_INDEX);
    auto metadataDesc = context_->GetOptionalInputDesc(METADATA_INPUT_INDEX);
    OP_CHECK_IF(metadataTensor == nullptr, OP_LOGE(opName, "metadata must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataDesc == nullptr, OP_LOGE(opName, "metadata desc must be provided."), return ge::GRAPH_FAILED);

    if (layoutQuery == "TND") {
        auto cuSeqQInput = context_->GetOptionalInputTensor(CU_SEQLENS_QUERY_INPUT_INDEX);
        OP_CHECK_IF(cuSeqQInput == nullptr, OP_LOGE(opName, "cuSeqlensQ must be provided when layoutQ is TND."),
                    return ge::GRAPH_FAILED);
    }
    if (layoutKey == "TND") {
        auto cuSeqKInput = context_->GetOptionalInputTensor(CU_SEQLENS_KEY_INPUT_INDEX);
        OP_CHECK_IF(cuSeqKInput == nullptr, OP_LOGE(opName, "cuSeqlensK must be provided when layoutK is TND."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::CheckOutShape(gert::Shape &inputshape,
                                                                                   const char *inputName,
                                                                                   gert::Shape &outputshape)
{
    if (layoutQuery[DIM_NUM_0] == 'T' && layoutQuery[DIM_NUM_1] == 'N' && layoutQuery[DIM_NUM_2] == 'D') {
        if (inputshape.GetDim(DIM_NUM_0) != outputshape.GetDim(DIM_NUM_0) ||
            inputshape.GetDim(DIM_NUM_1) != outputshape.GetDim(DIM_NUM_1) ||
            inputshape.GetDim(DIM_NUM_2) != outputshape.GetDim(DIM_NUM_2)) {
            OP_LOGE(
                context_,
                "DenseLightningIndexerKLLossGrad Input %s [%ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, %ld]",
                inputName, inputshape.GetDim(DIM_NUM_0), inputshape.GetDim(DIM_NUM_1), inputshape.GetDim(DIM_NUM_2),
                inputName, outputshape.GetDim(DIM_NUM_0), outputshape.GetDim(DIM_NUM_1), outputshape.GetDim(DIM_NUM_2));
            return ge::GRAPH_FAILED;
        }
    } else {
        if (inputshape.GetDim(DIM_NUM_0) != outputshape.GetDim(DIM_NUM_0) ||
            inputshape.GetDim(DIM_NUM_1) != outputshape.GetDim(DIM_NUM_1) ||
            inputshape.GetDim(DIM_NUM_2) != outputshape.GetDim(DIM_NUM_2) ||
            inputshape.GetDim(DIM_NUM_3) != outputshape.GetDim(DIM_NUM_3)) {
            OP_LOGE(
                context_,
                "DenseLightningIndexerKLLossGrad Input %s [%ld, %ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, "
                "%ld, %ld]",
                inputName, inputshape.GetDim(DIM_NUM_0), inputshape.GetDim(DIM_NUM_1), inputshape.GetDim(DIM_NUM_2),
                inputshape.GetDim(DIM_NUM_3), inputName, outputshape.GetDim(DIM_NUM_0), outputshape.GetDim(DIM_NUM_1),
                outputshape.GetDim(DIM_NUM_2), outputshape.GetDim(DIM_NUM_3));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 比较输出与输入的维度是否对应
ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::CheckOutPut()
{
    auto queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();
    auto softmaxLseShape = context_->GetInputShape(SOFTMAX_LSE_INPUT_INDEX)->GetStorageShape();

    auto dQueryIndexShape = context_->GetOutputShape(D_QUERY_OUTPUT_INDEX)->GetStorageShape();
    auto dKeyIndexShape = context_->GetOutputShape(D_KEY_OUTPUT_INDEX)->GetStorageShape();
    auto dWeightsShape = context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX)->GetStorageShape();
    auto softmaxOutShape = context_->GetOutputShape(SOFTMAX_OUT_OUTPUT_INDEX)->GetStorageShape();

    size_t layoutLen = strlen(layoutQuery);
    OP_CHECK_IF(queryIndexShape.GetDimNum() != layoutLen || keyIndexShape.GetDimNum() != layoutLen ||
                    attnSoftmaxL1Shape.GetDimNum() != layoutLen,
                OP_LOGE(opName,
                        "Invalid data, inputdata q shapelen [%ld] k shapelen [%ld] attnSoftmaxL1Norm shapelen [%ld] "
                        "is not equal to layoutQuery [%s] len[%ld].",
                        queryIndexShape.GetDimNum(), keyIndexShape.GetDimNum(), attnSoftmaxL1Shape.GetDimNum(),
                        layoutQuery, layoutLen),
                return ge::GRAPH_FAILED);

    auto qIdxDimNum = queryIndexShape.GetDimNum();
    if (qIdxDimNum != dQueryIndexShape.GetDimNum() ||
        queryIndexShape.GetDim(DIM_NUM_0) != dQueryIndexShape.GetDim(DIM_NUM_0) ||
        queryIndexShape.GetDim(DIM_NUM_1) != dQueryIndexShape.GetDim(DIM_NUM_1) ||
        (qIdxDimNum > DIM_NUM_2 && queryIndexShape.GetDim(DIM_NUM_2) != dQueryIndexShape.GetDim(DIM_NUM_2)) ||
        (qIdxDimNum > DIM_NUM_3 && queryIndexShape.GetDim(DIM_NUM_3) != dQueryIndexShape.GetDim(DIM_NUM_3))) {
        OP_LOGE(context_, "DenseLightningIndexerKLLossGrad Input queryIndex %s is not equal to Output dQueryIndex %s",
                Ops::Base::ToString(queryIndexShape).c_str(), Ops::Base::ToString(dQueryIndexShape).c_str());
        return ge::GRAPH_FAILED;
    }

    auto kIdxDimNum = keyIndexShape.GetDimNum();
    if (kIdxDimNum != dKeyIndexShape.GetDimNum() ||
        keyIndexShape.GetDim(DIM_NUM_0) != dKeyIndexShape.GetDim(DIM_NUM_0) ||
        keyIndexShape.GetDim(DIM_NUM_1) != dKeyIndexShape.GetDim(DIM_NUM_1) ||
        (kIdxDimNum > DIM_NUM_2 && keyIndexShape.GetDim(DIM_NUM_2) != dKeyIndexShape.GetDim(DIM_NUM_2)) ||
        (kIdxDimNum > DIM_NUM_3 && keyIndexShape.GetDim(DIM_NUM_3) != dKeyIndexShape.GetDim(DIM_NUM_3))) {
        OP_LOGE(context_, "DenseLightningIndexerKLLossGrad Input keyIndex %s is not equal to Output dKeyIndex %s",
                Ops::Base::ToString(keyIndexShape).c_str(), Ops::Base::ToString(dKeyIndexShape).c_str());
        return ge::GRAPH_FAILED;
    }

    auto wDimNum = weightsShape.GetDimNum();
    if (wDimNum != dWeightsShape.GetDimNum() || weightsShape.GetDim(DIM_NUM_0) != dWeightsShape.GetDim(DIM_NUM_0) ||
        weightsShape.GetDim(DIM_NUM_1) != dWeightsShape.GetDim(DIM_NUM_1) ||
        (wDimNum > DIM_NUM_2 && weightsShape.GetDim(DIM_NUM_2) != dWeightsShape.GetDim(DIM_NUM_2))) {
        OP_LOGE(context_, "DenseLightningIndexerKLLossGrad Input weights %s is not equal to Output dWeights %s",
                Ops::Base::ToString(weightsShape).c_str(), Ops::Base::ToString(dWeightsShape).c_str());
        return ge::GRAPH_FAILED;
    }

    auto attnSoftmaxDimNum = attnSoftmaxL1Shape.GetDimNum();
    if (attnSoftmaxDimNum != softmaxOutShape.GetDimNum() ||
        attnSoftmaxL1Shape.GetDim(DIM_NUM_0) != softmaxOutShape.GetDim(DIM_NUM_0) ||
        attnSoftmaxL1Shape.GetDim(DIM_NUM_1) != softmaxOutShape.GetDim(DIM_NUM_1) ||
        (attnSoftmaxDimNum > DIM_NUM_2 && attnSoftmaxL1Shape.GetDim(DIM_NUM_2) != softmaxOutShape.GetDim(DIM_NUM_2)) ||
        (attnSoftmaxDimNum > DIM_NUM_3 && attnSoftmaxL1Shape.GetDim(DIM_NUM_3) != softmaxOutShape.GetDim(DIM_NUM_3))) {
        OP_LOGE(context_,
                "DenseLightningIndexerKLLossGrad Input attnSoftmaxL1Norm %s is not equal to Output softmaxOut %s",
                Ops::Base::ToString(attnSoftmaxL1Shape).c_str(), Ops::Base::ToString(softmaxOutShape).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    // 输入属性参数验证
    auto layoutQueryPtr = attrs->GetAttrPointer<char>(idx++);
    auto layoutKeyPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int32_t>(idx++);
    auto cmpRatioPtr = attrs->GetAttrPointer<int32_t>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutQueryPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutKeyPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cmpRatioPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    // 输入shape验证
    auto queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX);
    auto keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX);
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX);
    auto attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX);
    auto softmaxLseShape = context_->GetInputShape(SOFTMAX_LSE_INPUT_INDEX);
    // 可选输入shape验证
    auto metadataShape = context_->GetOptionalInputShape(METADATA_INPUT_INDEX);
    // 输出shape验证
    auto dQueryIndexShape = context_->GetOutputShape(D_QUERY_OUTPUT_INDEX);
    auto dKeyIndexShape = context_->GetOutputShape(D_KEY_OUTPUT_INDEX);
    auto dWeightsShape = context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX);
    auto softmaxOutShape = context_->GetOutputShape(SOFTMAX_OUT_OUTPUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attnSoftmaxL1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxLseShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, metadataShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dQueryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dKeyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

bool DenseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto layoutQueryPtr = attrs->GetAttrPointer<char>(idx++);
    auto layoutKeyPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int32_t>(idx++);
    auto cmpRatioPtr = attrs->GetAttrPointer<int32_t>(idx++);
    auto cmpResidualK = context_->GetOptionalInputShape(CMP_RESIDUAL_KEY_INPUT_INDEX);
    layoutQuery = layoutQueryPtr;
    layoutKey = layoutKeyPtr;
    sparseMode = *sparseModePtr;
    cmpRatio = *cmpRatioPtr;
    deterministic = (context_->GetDeterministic() == 1);
    OP_CHECK_IF(strcmp(layoutQuery, "TND") != 0 && strcmp(layoutQuery, "BSND") != 0,
                OP_LOGE(opName, "LayoutQ only support TND or BSND, now LayoutQ is %s.", layoutQuery), return false);
    OP_CHECK_IF(strcmp(layoutKey, "TND") != 0 && strcmp(layoutKey, "BSND") != 0,
                OP_LOGE(opName, "LayoutK only support TND or BSND, now LayoutK is %s.", layoutKey), return false);
    OP_CHECK_IF(
        strcmp(layoutQuery, layoutKey) != 0,
        OP_LOGE(opName, "Layout of Query and Key need to be consistent, but now layoutQuery is %s and layoutKey is %s.",
                layoutQuery, layoutKey),
        return false);
    OP_CHECK_IF(
        (sparseMode != SPARSE_MODE_SIZE_3 && sparseMode != SPARSE_MODE_SIZE_0),
        OP_LOGE(opName, " The value of mask_mode is [%d], but currently only supports mode [0,3].", sparseMode),
        return false);
    OP_CHECK_IF((cmpRatio < CMP_RATIO_1 || cmpRatio > CMP_RATIO_128),
                OP_LOGE(opName, " The value of cmpRatio is [%d], but currently only supports ranging from 1 to 128.",
                        cmpRatio),
                return false);
    OP_CHECK_IF((sparseMode == SPARSE_MODE_SIZE_3 && cmpRatio != 1 && cmpResidualK == nullptr),
                OP_LOGE(opName, " cmp_residual_k is required when mask_mode is 3 with cmp_ratio not equal to 1!"),
                return false);

    OP_LOGD(context_, "attrs: layout_query[%s] layout_key[%s] mask_mode[%d] cmp_ratio[%d].", layoutQuery, layoutKey,
            sparseMode, cmpRatio);

    if (CheckOutPut() == ge::GRAPH_FAILED) {
        return false;
    }
    return true;
}

bool DenseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeDimLayout(const gert::Shape &queryIndexShape,
                                                                           const gert::Shape &keyIndexShape,
                                                                           const gert::Shape &weightsShape,
                                                                           const gert::Shape &attnSoftmaxL1Shape,
                                                                           size_t layoutLen)
{
    if (layoutLen == 3UL) {
        if (layoutQuery[0] == 'T' && layoutQuery[1] == 'N' && layoutQuery[2] == 'D') {
            t1Size = queryIndexShape.GetDim(0); // TND只有三个数值 T:0 N:1 D:2
            t2Size = keyIndexShape.GetDim(0);
            realT1Size = t1Size;
            const gert::Shape &actSeqQLenShape =
                context_->GetOptionalInputTensor(CU_SEQLENS_QUERY_INPUT_INDEX)->GetStorageShape();
            bSize = actSeqQLenShape.GetDim(0) - 1;
            s1Size = t1Size;
            s2Size = t2Size;
            accumS2 = t2Size;
            n1Size = queryIndexShape.GetDim(1);
            n2Size = keyIndexShape.GetDim(1);
            OP_CHECK_IF(n1Size < 1 || n1Size > 128,
                        OP_LOGE(opName, "Inputshape N Size of Query should be range in 1~128, but got %d.", n1Size),
                        return false);
            OP_CHECK_IF(n2Size != 1, OP_LOGE(opName, "Inputshape N Size of Key should be 1, but got %d.", n2Size),
                        return false);
            gSizeQueryIndex = queryIndexShape.GetDim(1) / n2Size;
            dSizeQueryIndex = queryIndexShape.GetDim(2);
            OP_CHECK_IF(dSizeQueryIndex != 128,
                        OP_LOGE(opName, "Inputshape D Size should be 128, but got %d.", dSizeQueryIndex),
                        return false);
            maxSeqlenK = attnSoftmaxL1Shape.GetDim(2);
            tilingData->baseParams.set_maxSeqlenK(static_cast<uint32_t>(maxSeqlenK));
            topKRange = TopKRangeRegbase::RANGE_0_2K;
            tilingData->baseParams.set_layoutType(static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND));
            tilingKeyLayout = LayoutTypeRegbase::LAYOUT_TND;
        }
    } else if (layoutLen == 4UL) {
        if (layoutQuery[0] == 'B' && layoutQuery[1] == 'S' && layoutQuery[2] == 'N' && layoutQuery[3] == 'D') {
            bSize = queryIndexShape.GetDim(0);
            s1Size = queryIndexShape.GetDim(1);
            s2Size = keyIndexShape.GetDim(1);
            n1Size = queryIndexShape.GetDim(2);
            n2Size = keyIndexShape.GetDim(2);
            OP_CHECK_IF(bSize < SIZE_1,
                        OP_LOGE(opName, "Inputshape B Size should be greater than 0, but got %d.", bSize),
                        return false);
            OP_CHECK_IF(s1Size < SIZE_1, OP_LOGE(opName, "Query s1Size should be greater than 0, but got %d.", s1Size),
                        return false);
            OP_CHECK_IF(s2Size < SIZE_1, OP_LOGE(opName, "Query s2Size should be greater than 0, but got %d.", s2Size),
                        return false);
            OP_CHECK_IF(n1Size < 1 || n1Size > 128,
                        OP_LOGE(opName, "Inputshape N Size of Query should be range in 1~128, but got %d.", n1Size),
                        return false);
            OP_CHECK_IF(n2Size != 1, OP_LOGE(opName, "Inputshape N Size of Key should be 1, but got %d.", n2Size),
                        return false);
            gSizeQueryIndex = queryIndexShape.GetDim(2) / n2Size;
            dSizeQueryIndex = queryIndexShape.GetDim(3);
            OP_CHECK_IF(dSizeQueryIndex != 128,
                        OP_LOGE(opName, "Inputshape D Size should be 128, but got %d.", dSizeQueryIndex),
                        return false);
            maxSeqlenK = attnSoftmaxL1Shape.GetDim(3);
            tilingData->baseParams.set_maxSeqlenK(static_cast<uint32_t>(maxSeqlenK));
            topKRange = TopKRangeRegbase::RANGE_0_2K;
            tilingData->baseParams.set_layoutType(static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_BSND));
            tilingKeyLayout = LayoutTypeRegbase::LAYOUT_BSND;
        }
    } else {
        return false;
    }

    return true;
}

// 对每一个输入参数的变量类型进行对比校验
bool DenseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeDtype()
{
    // 对5个必须输入的参数进行参数类型判断
    // 输入空指针校验
    auto queryDesc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    auto keyDesc = context_->GetInputDesc(KEY_INPUT_INDEX);
    auto weightsDesc = context_->GetInputDesc(WEIGHT_INPUT_INDEX);
    auto attnSoftmaxL1Desc = context_->GetInputDesc(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX);
    auto softmaxLseDesc = context_->GetInputDesc(SOFTMAX_LSE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attnSoftmaxL1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxLseDesc);

    // 以下2个为16类型保持一致
    auto queryDtype = queryDesc->GetDataType();
    auto keyDtype = keyDesc->GetDataType();

    // 以下3个为32类型
    auto weightsDtype = weightsDesc->GetDataType();
    auto softmaxLseDtype = softmaxLseDesc->GetDataType();
    auto attnSoftmaxL1Dtype = attnSoftmaxL1Desc->GetDataType();

    bool same16 = false; // 判断输入为fp16或者部分16的是否一致
    bool same32 = false; // 判断输入为int32或者fp32的类型是否正确
    if (queryDtype == ge::DT_FLOAT16 && keyDtype == ge::DT_FLOAT16) {
        same16 = true;
    } else if (queryDtype == ge::DT_BF16 && keyDtype == ge::DT_BF16) {
        same16 = true;
    } else {
        OP_LOGE(context_, "InputDtype is not same.: queryDtype[%s], keyDtype[%s]",
                ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str());
        same16 = false;
    }
    if (weightsDtype == ge::DT_FLOAT && softmaxLseDtype == ge::DT_FLOAT && attnSoftmaxL1Dtype == ge::DT_FLOAT) {
        same32 = true;
    } else {
        OP_LOGE(context_,
                "InputDtype is fault: weightsDtype must be float32, but[%s]; softmaxLseDtype must be float32, but[%s]; "
                "attnSoftmaxL1Dtype must be float32, but[%s].",
                ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(softmaxLseDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(attnSoftmaxL1Dtype).c_str());
        same32 = false;
    }
    // 所有类型不满足返回false
    if (!same16 || !same32) {
        return false;
    }
    OP_LOGI(context_, "InputDtype: queryDtype[%s], keyDtype[%s],",
            ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str());
    OP_LOGI(context_, "weightsDtype[%s], softmaxLseDtype[%s], attnSoftmaxL1Dtype[%s].",
            ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(softmaxLseDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(attnSoftmaxL1Dtype).c_str());
    return true;
}

// 输入shape进行交叉验证，防止数据错误输入
bool DenseLightningIndexerKLLossGradTilingGeneralRegbase::CrossShapeVerify()
{
    auto queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();
    auto softmaxLseShape = context_->GetInputShape(SOFTMAX_LSE_INPUT_INDEX)->GetStorageShape();
    // 下面数字对应shape输入位置
    if (layoutQuery[0] == 'T' && layoutQuery[1] == 'N' && layoutQuery[2] == 'D') {
        int64_t t1Len = queryIndexShape[0];
        int64_t n1Len = queryIndexShape[1];
        int64_t t2Len = keyIndexShape[0];
        int64_t n2Len = keyIndexShape[1];
        // 验证T1
        OP_CHECK_IF(
            weightsShape[0] != t1Len || softmaxLseShape[1] != t1Len || attnSoftmaxL1Shape[0] != t1Len,
            OPS_REPORT_VECTOR_INNER_ERR(opName,
                                        "CrossShapeVerify T1 is failed, the value of query_index[0],"
                                        "weights[0], softmax_lse[1] and attn_softmax_l1_norm[0]"
                                        "are respectively (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                                        queryIndexShape[0], weightsShape[0], softmaxLseShape[1], attnSoftmaxL1Shape[0]),
            return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(queryIndexShape[1] < NQUERYINDEX_SIZE_1 || queryIndexShape[1] > NQUERYINDEX_SIZE_128,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query_index must be >= 1 and <= "
                                                "128, but the value of query_index[1] "
                                                "is (%ld)",
                                                queryIndexShape[1]),
                    return false);
        // 验证N Index
        OP_CHECK_IF(queryIndexShape[1] != weightsShape[1],
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify N index is failed, the value of query_index[1] and "
                                                "weights[1] are respectively (%ld), "
                                                "(%ld). Their values should be equal.",
                                                queryIndexShape[1], weightsShape[1]),
                    return false);
        // 验证N2 数字是否正确
        OP_CHECK_IF(keyIndexShape[1] != NKEYINDEX_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be 1, but the value of key_index[1] is (%ld).",
                        keyIndexShape[1]),
                    return false);
        OP_CHECK_IF(softmaxLseShape[0] != NKEYINDEX_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmaxLseShape[0] is (%ld).",
                        softmaxLseShape[0]),
                    return false);
        OP_CHECK_IF(
            attnSoftmaxL1Shape[1] != NKEYINDEX_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2 must be 1, but the value of attn_softmax_l1_norm[1] is (%ld).",
                attnSoftmaxL1Shape[1]),
            return false);
        // 验证N2
        OP_CHECK_IF(softmaxLseShape[0] != keyIndexShape[1] || attnSoftmaxL1Shape[1] != keyIndexShape[1],
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify N2 is failed, the value of key_index[1], "
                                                "softmaxLseShape[0] and attn_softmax_l1_norm[1] "
                                                "are respectively (%ld), (%ld), (%ld). Their values should be equal.",
                                                keyIndexShape[1], softmaxLseShape[0], attnSoftmaxL1Shape[1]),
                    return false);
        // 验证D 数字是否正确
        OP_CHECK_IF(
            queryIndexShape[2] != DINDEX_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[2] is (%ld)",
                queryIndexShape[2]),
            return false);
        OP_CHECK_IF(
            keyIndexShape[2] != DINDEX_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[2] is (%ld)",
                keyIndexShape[2]),
            return false);
        // 验证D
        OP_CHECK_IF(keyIndexShape[2] != queryIndexShape[2],
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify query_index-key_index shape D is failed, the value "
                                                "of query_index[2] and key_index[2] "
                                                "are respectively (%ld), (%ld). Their values should be equal.",
                                                queryIndexShape[2], keyIndexShape[2]),
                    return false);
    } else if (layoutQuery[0] == 'B' && layoutQuery[1] == 'S' && layoutQuery[2] == 'N' && layoutQuery[3] == 'D') {
        int64_t bLen = queryIndexShape[0];
        int64_t s1Len = queryIndexShape[1];
        int64_t n1Len = queryIndexShape[2];
        int64_t s2Len = keyIndexShape[1];
        int64_t n2Len = keyIndexShape[2];

        // 验证B
        OP_CHECK_IF(keyIndexShape[0] != bLen || weightsShape[0] != bLen || softmaxLseShape[0] != bLen ||
                        attnSoftmaxL1Shape[0] != bLen,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify shape B is failed, the value of query_index[0],"
                        "key_index[0], weights[0], softmaxLseShape[0] and attn_softmax_l1_norm[0]"
                        "are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                        bLen, keyIndexShape[0], weightsShape[0], softmaxLseShape[0], attnSoftmaxL1Shape[0]),
                    return false);
        // 验证s1
        OP_CHECK_IF(weightsShape[1] != s1Len || softmaxLseShape[2] != s1Len || attnSoftmaxL1Shape[1] != s1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify S1 is failed, the value of query_index[1], weights[1],"
                        "softmaxLseShape[2] and attn_softmax_l1_norm[1] are respectively (%ld), (%ld), (%ld), (%ld)."
                        "Their values should be equal.",
                        s1Len, weightsShape[1], softmaxLseShape[2], attnSoftmaxL1Shape[1]),
                    return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(n1Len < NQUERYINDEX_SIZE_1 || n1Len > NQUERYINDEX_SIZE_128,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query_index must be >= 1 and <= "
                                                "128, but the value of query_index[2]"
                                                "is (%ld)",
                                                n1Len),
                    return false);
        // 验证N Index
        OP_CHECK_IF(weightsShape[2] != n1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify N index is failed, the value of query_index[2] and "
                                                "weights[2] are respectively (%ld), "
                                                "(%ld). Their values should be equal.",
                                                n1Len, weightsShape[2]),
                    return false);
        // 验证N2 数字是否正确
        OP_CHECK_IF(keyIndexShape[2] != NKEYINDEX_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be 1, but the value of key_index[2] is (%ld).",
                        keyIndexShape[2]),
                    return false);
        OP_CHECK_IF(softmaxLseShape[1] != NKEYINDEX_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be 1, but the value of softmaxLseShape[1] is (%ld).",
                        softmaxLseShape[1]),
                    return false);
        OP_CHECK_IF(
            attnSoftmaxL1Shape[2] != NKEYINDEX_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2 must be 1, but the value of attn_softmax_l1_norm[2] is (%ld).",
                attnSoftmaxL1Shape[2]),
            return false);
        // 验证N2
        OP_CHECK_IF(softmaxLseShape[1] != n2Len || attnSoftmaxL1Shape[2] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify N2 is failed, the value of key_index[2], "
                                                "softmaxLseShape[1] and attn_softmax_l1_norm[2] "
                                                "are respectively (%ld), (%ld), (%ld). Their values should be equal.",
                                                n2Len, softmaxLseShape[1], attnSoftmaxL1Shape[2]),
                    return false);
        // 验证D 数字是否正确
        OP_CHECK_IF(
            queryIndexShape[3] != DINDEX_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[3] is (%ld)",
                queryIndexShape[3]),
            return false);
        OP_CHECK_IF(
            keyIndexShape[3] != DINDEX_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[3] is (%ld)",
                keyIndexShape[3]),
            return false);
        // 验证D
        OP_CHECK_IF(keyIndexShape[3] != queryIndexShape[3],
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify query_index-key_index shape D is failed, the value "
                                                "of query_index[3] and key_index[3] "
                                                "are respectively (%ld), (%ld). Their values should be equal.",
                                                queryIndexShape[3], keyIndexShape[3]),
                    return false);
    }
    return true;
}

bool DenseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeLayout()
{
    auto &queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto &attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();
    auto &softmaxLseShape = context_->GetInputShape(SOFTMAX_LSE_INPUT_INDEX)->GetStorageShape();

    size_t layoutLen = strlen(layoutQuery);
    OP_CHECK_IF(!CrossShapeVerify(), OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify Failed"), return false);
    OP_CHECK_IF(!AnalyzeDimLayout(queryIndexShape, keyIndexShape, weightsShape, attnSoftmaxL1Shape, layoutLen),
                OP_LOGE(opName, "Layout %s data analyze failed.", layoutQuery), return false);
    OP_CHECK_IF(gSizeQueryIndex == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSizeQueryIndex is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(dSizeQueryIndex <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "dSizeQueryIndex is not support <= 0"),
                return false);
    return true;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const DenseLightningIndexerKLLossGradCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(opName, "compileInfoPtr is null."), return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
            aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());

    if (ValidateRequiredInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
                OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
    // 将基本输入参数传给tilingdata
    dlikGradBaseParams_->set_bSize(bSize);
    dlikGradBaseParams_->set_t1Size(t1Size);
    dlikGradBaseParams_->set_t2Size(t2Size);
    dlikGradBaseParams_->set_n2Size(n2Size);
    dlikGradBaseParams_->set_gSizeQueryIndex(gSizeQueryIndex);
    dlikGradBaseParams_->set_s1Size(s1Size);
    dlikGradBaseParams_->set_s2Size(s2Size);
    dlikGradBaseParams_->set_dSizeQueryIndex(dSizeQueryIndex);
    dlikGradBaseParams_->set_sparseMode(sparseMode);
    dlikGradBaseParams_->set_cmpRatio(cmpRatio);

    OP_LOGD(context_,
            "INPUTPARAM bsize:[%d], n2Size:[%d], gSizeQueryIndex:[%d],"
            "s1Size:[%d], s2Size:[%d], dSizeQueryIndex:[%d],"
            "sparseMode:[%d], cmpRatio:[%d].",
            bSize, n2Size, gSizeQueryIndex, s1Size, s2Size, dSizeQueryIndex, sparseMode, cmpRatio);

    return ge::GRAPH_SUCCESS;
}

// 计算S1的总个数
int64_t DenseLightningIndexerKLLossGradTilingGeneralRegbase::CalcTotalSize()
{
    if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_TND) {
        return realT1Size;
    } else if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_BSND) {
        return bSize * s1Size;
    }
    return 0;
}

void DenseLightningIndexerKLLossGradTilingGeneralRegbase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(coreNum));
    dlikGradMultiCoreParams_->set_coreNum(static_cast<int32_t>(aicNum));
    dlikGradMultiCoreParams_->set_totalSize(totalSize);
    dlikGradMultiCoreParams_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
}

void DenseLightningIndexerKLLossGradTilingGeneralRegbase::InitOutputSplit()
{
    DLIKGradInitOutputParamsRegbase *initoutput = &tilingData->initOutputParams;
    auto &dKeyIndexShape = context_->GetOutputShape(D_KEY_OUTPUT_INDEX)->GetStorageShape();
    int64_t totalsize = 0;
    uint32_t singlecoresize = 0;
    if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_TND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(2); // TND场景下D为第二个维度
        int64_t totalT2Size = dKeyIndexShape.GetDim(0);    // T为第一个维度
        totalsize = totalT2Size * static_cast<int64_t>(dsizeDkeyindex);
    } else if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_BSND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(3); // BSND场景下D为第三个维度
        int64_t totals2Size = dKeyIndexShape.GetDim(1);    // s为第二个维度
        totalsize = static_cast<int64_t>(bSize) * totals2Size * dsizeDkeyindex;
    }
    // 单个核均分元素数量
    singlecoresize = static_cast<uint32_t>(
        CeilDivision(totalsize, static_cast<int64_t>(aivNum))); // 输出k-index总大小TD或者BSD 除以 总的aiv核数 向上取整
    initoutput->set_singleCoreSize(singlecoresize);
    initoutput->set_totalOutputSize(totalsize);
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::DoCastTiling()
{
    // softmaxOut、dw、dq
    int64_t dAlign16 = (dlikGradBaseParams_->get_dSizeQueryIndex() + 15) >> 4 << 4;
    // query
    int64_t allNumQuery = dlikGradBaseParams_->get_bSize() * dlikGradBaseParams_->get_n2Size() *
                          dlikGradBaseParams_->get_gSizeQueryIndex() * dlikGradBaseParams_->get_s1Size() * dAlign16;
    int64_t allNumS2 =
        dlikGradBaseParams_->get_bSize() * dlikGradBaseParams_->get_n2Size() * dlikGradBaseParams_->get_s2Size();
    // TND时候要按照真实的query的num数计算
    if (dlikGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        allNumQuery = dlikGradBaseParams_->get_t1Size() * dlikGradBaseParams_->get_n2Size() *
                      dlikGradBaseParams_->get_gSizeQueryIndex() * dAlign16;
        allNumS2 = dlikGradBaseParams_->get_t2Size() * dlikGradBaseParams_->get_n2Size();
    }
    int64_t allNumKey = allNumS2 * dAlign16;
    // weight
    int64_t allNumWeight = dlikGradBaseParams_->get_bSize() * dlikGradBaseParams_->get_n2Size() *
                           dlikGradBaseParams_->get_s1Size() * dlikGradBaseParams_->get_gSizeQueryIndex();
    // TND时候要按照真实的cmp_softmax_l1的num数计算
    if (dlikGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        allNumWeight = dlikGradBaseParams_->get_t1Size() * dlikGradBaseParams_->get_n2Size() *
                       dlikGradBaseParams_->get_gSizeQueryIndex();
    }

    // softmaxOut
    int64_t allNumSoftmaxOut = dlikGradBaseParams_->get_bSize() * dlikGradBaseParams_->get_s1Size() *
                               dlikGradBaseParams_->get_n2Size() * dlikGradBaseParams_->get_maxSeqlenK();
    // TND时候要按照真实的softmaxOut的num数计算
    if (dlikGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        allNumSoftmaxOut = dlikGradBaseParams_->get_t1Size() * dlikGradBaseParams_->get_n2Size() *
                           dlikGradBaseParams_->get_maxSeqlenK();
    }
    uint32_t typeSize = B16;
    uint32_t usedCoreNum = aivNum;
    uint32_t coreNum = aicNum;
    constexpr uint32_t postNzCoexNode = 12;
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t postNzReservedN = 1;

    uint32_t postUbBaseSize = 0;
    int64_t nzReservedSize = 0;
    int64_t curPostCoexNode = postNzCoexNode;
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    OP_LOGI(context_, "DoCastTiling postUbBaseSize: %ld.", postUbBaseSize);

    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);

    int64_t qPreBlockFactor = (allNumQuery + usedCoreNum - 1) / usedCoreNum;
    int64_t qPreBlockTotal = (allNumQuery + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreBlockTailTmp = allNumQuery % qPreBlockFactor;
    int64_t qPreBlockTail = qPreBlockTailTmp == 0 ? qPreBlockFactor : qPreBlockTailTmp;

    int64_t kPreBlockFactor = (allNumKey + usedCoreNum - 1) / usedCoreNum;
    int64_t kPreBlockTotal = (allNumKey + kPreBlockFactor - 1) / kPreBlockFactor;
    int64_t kPreBlockTailTmp = allNumKey % kPreBlockFactor;
    int64_t kPreBlockTail = kPreBlockTailTmp == 0 ? kPreBlockFactor : kPreBlockTailTmp;

    int64_t weightPreBlockFactor = 0;
    int64_t weightPreBlockTotal = 0;
    int64_t weightPreBlockTailTmp = 0;
    int64_t weightPreBlockTail = 0;
    if (allNumWeight != 0) {
        weightPreBlockFactor = (allNumWeight + usedCoreNum - 1) / usedCoreNum;
        weightPreBlockTotal = (allNumWeight + weightPreBlockFactor - 1) / weightPreBlockFactor;
        weightPreBlockTailTmp = allNumWeight % weightPreBlockFactor;
        weightPreBlockTail = weightPreBlockTailTmp == 0 ? weightPreBlockFactor : weightPreBlockTailTmp;
    }

    int64_t softmaxOutPreBlockFactor = 0;
    int64_t softmaxOutPreBlockTotal = 0;
    int64_t softmaxOutPreBlockTailTmp = 0;
    int64_t softmaxOutPreBlockTail = 0;
    if (allNumSoftmaxOut != 0) {
        softmaxOutPreBlockFactor = (allNumSoftmaxOut + usedCoreNum - 1) / usedCoreNum;
        softmaxOutPreBlockTotal = (allNumSoftmaxOut + softmaxOutPreBlockFactor - 1) / softmaxOutPreBlockFactor;
        softmaxOutPreBlockTailTmp = allNumSoftmaxOut % softmaxOutPreBlockFactor;
        softmaxOutPreBlockTail = softmaxOutPreBlockTailTmp == 0 ? softmaxOutPreBlockFactor : softmaxOutPreBlockTailTmp;
    }

    dlikGradMultiCoreParams_->set_postUbBaseSize(postUbBaseSize);

    dlikGradMultiCoreParams_->set_qPreBlockFactor(qPreBlockFactor);
    dlikGradMultiCoreParams_->set_qPreBlockTotal(qPreBlockTotal);
    dlikGradMultiCoreParams_->set_qPreBlockTail(qPreBlockTail);

    dlikGradMultiCoreParams_->set_kPreBlockFactor(kPreBlockFactor);
    dlikGradMultiCoreParams_->set_kPreBlockTotal(kPreBlockTotal);
    dlikGradMultiCoreParams_->set_kPreBlockTail(kPreBlockTail);

    dlikGradMultiCoreParams_->set_weightPreBlockFactor(weightPreBlockFactor);
    dlikGradMultiCoreParams_->set_weightPreBlockTotal(weightPreBlockTotal);
    dlikGradMultiCoreParams_->set_weightPreBlockTail(weightPreBlockTail);

    dlikGradMultiCoreParams_->set_softmaxOutPreBlockFactor(softmaxOutPreBlockFactor);
    dlikGradMultiCoreParams_->set_softmaxOutPreBlockTotal(softmaxOutPreBlockTotal);
    dlikGradMultiCoreParams_->set_softmaxOutPreBlockTail(softmaxOutPreBlockTail);

    int64_t kPostBlockFactor = (allNumS2 + usedCoreNum - 1) / usedCoreNum;
    int64_t kPostBlockTotal = (allNumS2 + kPostBlockFactor - 1) / kPostBlockFactor;
    int64_t kPostBlockTailTmp = allNumS2 % kPostBlockFactor;
    int64_t kPostBlockTail = kPostBlockTailTmp == 0 ? kPostBlockFactor : kPostBlockTailTmp;

    dlikGradMultiCoreParams_->set_kPostBlockFactor(kPostBlockFactor);
    dlikGradMultiCoreParams_->set_kPostBlockTotal(kPostBlockTotal);
    dlikGradMultiCoreParams_->set_kPostBlockTail(kPostBlockTail);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::DoOpTiling()
{
    context_->SetScheduleMode(1);
    OP_LOGD(context_, "try template[%s]", templateName);
    int64_t totalSize = CalcTotalSize();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    context_->SetBlockDim(aicNum); // 使用的核数确定

    auto status = DoCastTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    OP_LOGD(context_, "ending template[%s]", templateName);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t DenseLightningIndexerKLLossGradTilingGeneralRegbase::GetTilingKey() const
{
    bool hasCuSeqlensQ = context_->GetOptionalInputTensor(CU_SEQLENS_QUERY_INPUT_INDEX) != nullptr;
    bool hasCuSeqlensK = context_->GetOptionalInputTensor(CU_SEQLENS_KEY_INPUT_INDEX) != nullptr;
    bool hasSequsedQ = context_->GetOptionalInputTensor(SEQUSED_QUERY_INPUT_INDEX) != nullptr;
    bool hasSequsedK = context_->GetOptionalInputTensor(SEQUSED_KEY_INPUT_INDEX) != nullptr;
    bool hasCmpResidualK = context_->GetOptionalInputTensor(CMP_RESIDUAL_KEY_INPUT_INDEX) != nullptr;
    return GET_TPL_TILING_KEY(static_cast<uint8_t>(topKRange), static_cast<uint8_t>(tilingKeyLayout),
                              static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(sparseMode),
                              static_cast<uint8_t>(hasCuSeqlensQ), static_cast<uint8_t>(hasCuSeqlensK),
                              static_cast<uint8_t>(hasSequsedQ), static_cast<uint8_t>(hasSequsedK),
                              static_cast<uint8_t>(hasCmpResidualK), static_cast<uint8_t>(deterministic));
}

ge::graphStatus DenseLightningIndexerKLLossGradTilingGeneralRegbase::GetWorkspaceSize()
{
    int64_t coreNum = dlikGradMultiCoreParams_->get_coreNum();
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t s2BlockSize = 8192;
    int64_t bmm3Size = dlikGradBaseParams_->get_bSize() * dlikGradBaseParams_->get_n2Size() *
                       dlikGradBaseParams_->get_s2Size() * dSizeQueryIndex * sizeof(float);
    // TND时候要按照真实的query的num数计算
    if (dlikGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        bmm3Size =
            dlikGradBaseParams_->get_t2Size() * dlikGradBaseParams_->get_n2Size() * dSizeQueryIndex * sizeof(float);
    }
    int64_t bmm3WorkSpace = s2BlockSize * dlikGradBaseParams_->get_n2Size() * dSizeQueryIndex * sizeof(float);
    int64_t reduceSumOffset = BLOCK_SIZE * sizeof(float) * 2;
    int64_t singlecoreTotalSize = THREE_BUFFER_VALUE * reduceSumOffset;
    int64_t multicoreTotalsize = 0;
    if (deterministic) {
        singlecoreTotalSize = singlecoreTotalSize + THREE_BUFFER_VALUE * bmm3WorkSpace;
    }
    multicoreTotalsize = singlecoreTotalSize * static_cast<int64_t>(coreNum) + bmm3Size;
    int64_t workspaceOffsets = singlecoreTotalSize * coreNum;
    dlikGradBaseParams_->set_mm3ResWorkSpaceOffset(workspaceOffsets);
    workspaces[0] = static_cast<size_t>(multicoreTotalsize) + WORK_SPACE_RESERVE_SIZE; // 预留16M空间必须加;
    OP_LOGW(context_, "workspace size:[%ld], multicoreTotalsize:[%ld]", workspaces[0], multicoreTotalsize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(DenseLightningIndexerKLLossGrad, DenseLightningIndexerKLLossGradTilingGeneralRegbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace optiling
