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
 * \file sparse_lightning_indexer_kl_loss_grad_tiling_general_regbase.cpp
 * \brief
 */

#include "sparse_lightning_indexer_kl_loss_grad_tiling_general_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include <tiling/tiling_api.h>

using namespace ge;
using namespace AscendC;

namespace optiling {
static const int64_t PING_PONG_VALUE = 2L;
static const int64_t GM_ALIGN = 512;
static const int32_t FRACTAL_NUM = 16L;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t B32 = 4; // 4B
static constexpr uint32_t B16 = 2;
static constexpr uint32_t BASE_LEN_256 = 256;

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
static auto AlignDown(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
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

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::CheckOutShape(gert::Shape &inputshape,
                                                                                    const char *inputName,
                                                                                    gert::Shape &outputshape)
{
    if (layoutQuery[DIM_NUM_0] == 'T' && layoutQuery[DIM_NUM_1] == 'N' && layoutQuery[DIM_NUM_2] == 'D') {
        if (inputshape.GetDim(DIM_NUM_0) != outputshape.GetDim(DIM_NUM_0) ||
            inputshape.GetDim(DIM_NUM_1) != outputshape.GetDim(DIM_NUM_1) ||
            inputshape.GetDim(DIM_NUM_2) != outputshape.GetDim(DIM_NUM_2)) {
            OP_LOGE(
                context_,
                "SparseLightningIndexerKLLossGrad Input %s [%ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, %ld]",
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
                "SparseLightningIndexerKLLossGrad Input %s [%ld, %ld, %ld, %ld] is not equal to Output d_%s [%ld, %ld, "
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
ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::CheckOutPut()
{
    auto queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto topKShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();
    auto attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();
    auto dQueryIndexShape = context_->GetOutputShape(D_QUERY_OUTPUT_INDEX)->GetStorageShape();
    auto dKeyIndexShape = context_->GetOutputShape(D_KEY_OUTPUT_INDEX)->GetStorageShape();
    auto dWeightsShape = context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX)->GetStorageShape();
    auto softmaxOutShape = context_->GetOutputShape(SOFTMAX_OUT_OUTPUT_INDEX)->GetStorageShape();

    size_t layoutLen = strlen(layoutQuery);
    OP_CHECK_IF(queryIndexShape.GetDimNum() != layoutLen || keyIndexShape.GetDimNum() != layoutLen ||
                    topKShape.GetDimNum() != layoutLen || attnSoftmaxL1Shape.GetDimNum() != layoutLen,
                OP_LOGE(opName, "Invalid data, inputdata shapelen [%ld] is not equal to layoutQuery [%s] len[%ld].",
                        queryIndexShape.GetDimNum(), layoutQuery, layoutLen),
                return false);

    auto status = CheckOutShape(queryIndexShape, "query", dQueryIndexShape);
    if (status == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOutShape(keyIndexShape, "key", dKeyIndexShape);
    if (status == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOutShape(attnSoftmaxL1Shape, "attn_softmax_l1_norm", softmaxOutShape);
    if (status == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (layoutQuery[DIM_NUM_0] == 'B' && layoutQuery[DIM_NUM_1] == 'S' && layoutQuery[DIM_NUM_2] == 'N' &&
        layoutQuery[DIM_NUM_3] == 'D') {
        if (dWeightsShape.GetDim(DIM_NUM_0) != weightsShape.GetDim(DIM_NUM_0) ||
            dWeightsShape.GetDim(DIM_NUM_1) != weightsShape.GetDim(DIM_NUM_1) ||
            dWeightsShape.GetDim(DIM_NUM_2) != weightsShape.GetDim(DIM_NUM_2)) {
            OP_LOGE(context_, "The input weights shape is [%ld, %ld, %ld], but d_weights got [%ld, %ld, %ld].",
                    weightsShape.GetDim(DIM_NUM_0), weightsShape.GetDim(DIM_NUM_1), weightsShape.GetDim(DIM_NUM_2),
                    dWeightsShape.GetDim(DIM_NUM_0), dWeightsShape.GetDim(DIM_NUM_1), dWeightsShape.GetDim(DIM_NUM_2));
            return GRAPH_FAILED;
        }
    } else if (layoutQuery[DIM_NUM_0] == 'T' && layoutQuery[DIM_NUM_1] == 'N' && layoutQuery[DIM_NUM_2] == 'D') {
        if (dWeightsShape.GetDim(DIM_NUM_0) != weightsShape.GetDim(DIM_NUM_0) ||
            dWeightsShape.GetDim(DIM_NUM_1) != weightsShape.GetDim(DIM_NUM_1)) {
            OP_LOGE(context_, "The input weights shape is [%ld, %ld], but d_weights got [%ld, %ld].",
                    weightsShape.GetDim(DIM_NUM_0), weightsShape.GetDim(DIM_NUM_1), dWeightsShape.GetDim(DIM_NUM_0),
                    dWeightsShape.GetDim(DIM_NUM_1));
            return GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::CheckContext()
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
    auto sparseIndicesShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    auto attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX);
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
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseIndicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attnSoftmaxL1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, metadataShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dQueryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dKeyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, softmaxOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeAttrs()
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
                OP_LOGE(opName, "Layout only support TND or BSND, now layout is %s.", layoutQuery), return false);
    OP_CHECK_IF(strcmp(layoutKey, "TND") != 0 && strcmp(layoutKey, "BSND") != 0,
                OP_LOGE(opName, "Layout only support TND or BSND, now layout is %s.", layoutKey), return false);
    OP_CHECK_IF(
        strcmp(layoutQuery, layoutKey) != 0,
        OP_LOGE(opName,
                "Layout of Query and Key/Value need to be consistent, but now layoutQuery is %s and layoutKey is %s.",
                layoutQuery, layoutKey),
        return false);
    OP_CHECK_IF(
        (sparseMode != SPARSE_MODE_SIZE_3 && sparseMode != SPARSE_MODE_SIZE_0),
        OP_LOGE(opName, " The value of sparse_mode is [%ld], but currently only supports mode [0,3].", sparseMode),
        return false);
    OP_CHECK_IF((cmpRatio < CMP_RATIO_1 || cmpRatio > CMP_RATIO_128),
                OP_LOGE(opName, " The value of cmpRatio is [%ld], but currently only supports ranging from 1 to 128.",
                        cmpRatio),
                return false);
    OP_CHECK_IF((sparseMode == SPARSE_MODE_SIZE_3 && cmpRatio != 1 && cmpResidualK == nullptr),
                OP_LOGE(opName, " cmp_residual_k is required when mask_mode is 3 with cmp_ratio not equal to 1!"),
                return false);

    OP_LOGD(context_, "attrs: layout_query[%s] layout_key[%s] sparse_mode[%ld] cmp_ratio[%ld].", layoutQuery, layoutKey,
            sparseMode, cmpRatio);

    if (CheckOutPut() == ge::GRAPH_FAILED) {
        return false;
    }
    return true;
}

bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeDimLayout(const gert::Shape &queryIndexShape,
                                                                            const gert::Shape &keyIndexShape,
                                                                            const gert::Shape &weightsShape,
                                                                            const gert::Shape &topKShape,
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
                        OP_LOGE(opName, "Inputshape N Size of Query should be range in 1~128, but got %ld.", n1Size),
                        return false);
            OP_CHECK_IF(n2Size != 1, OP_LOGE(opName, "Inputshape N Size of Key should be 1, but got %ld.", n2Size),
                        return false);
            gSizeQuery = queryIndexShape.GetDim(1) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(1) / n2Size;
            dSizeQuery = queryIndexShape.GetDim(2);
            dSizeQueryIndex = queryIndexShape.GetDim(2);
            OP_CHECK_IF(dSizeQueryIndex != 128,
                        OP_LOGE(opName, "Inputshape D Size should be 128, but got %ld.", dSizeQueryIndex),
                        return false);
            kSize = topKShape.GetDim(2);
            OP_CHECK_IF(kSize < 1 || kSize > BUFFER_SIZE_BYTE_2K,
                        OP_LOGE(opName, "topK (%ld) should be range in 1~2048.", kSize), return false);
            topKRange = (kSize <= BUFFER_SIZE_BYTE_2K) ? TopKRangeRegbase::RANGE_0_2K : TopKRangeRegbase::RANGE_2K_8K;
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
                        OP_LOGE(opName, "Inputshape B Size should be greater than 0, but got %ld.", bSize),
                        return false);
            OP_CHECK_IF(s1Size < SIZE_1, OP_LOGE(opName, "Query s1Size should be greater than 0, but got %ld.", s1Size),
                        return false);
            OP_CHECK_IF(s2Size < SIZE_1, OP_LOGE(opName, "Query s2Size should be greater than 0, but got %ld.", s2Size),
                        return false);
            OP_CHECK_IF(n1Size < 1 || n1Size > 128,
                        OP_LOGE(opName, "Inputshape N Size of Query should be range in 1~128, but got %ld.", n1Size),
                        return false);
            OP_CHECK_IF(n2Size != 1, OP_LOGE(opName, "Inputshape N Size of Key should be 1, but got %ld.", n2Size),
                        return false);
            gSizeQueryIndex = queryIndexShape.GetDim(2) / n2Size;
            dSizeQueryIndex = queryIndexShape.GetDim(3);
            OP_CHECK_IF(dSizeQueryIndex != 128,
                        OP_LOGE(opName, "Inputshape D Size should be 128, but got %ld.", dSizeQueryIndex),
                        return false);
            kSize = topKShape.GetDim(3);
            OP_CHECK_IF(kSize < 1 || kSize > BUFFER_SIZE_BYTE_2K,
                        OP_LOGE(opName, "topK (%ld) should be range in 1~2048.", kSize), return false);
            topKRange = (kSize <= BUFFER_SIZE_BYTE_2K) ? TopKRangeRegbase::RANGE_0_2K : TopKRangeRegbase::RANGE_2K_8K;
            tilingData->baseParams.set_layoutType(static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_BSND));
            tilingKeyLayout = LayoutTypeRegbase::LAYOUT_BSND;
        }
    } else {
        return false;
    }

    return true;
}

// 对每一个输入参数的变量类型进行对比校验
bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeDtype()
{
    // 对5个必须输入的参数进行参数类型判断
    // 输入空指针校验
    auto queryDesc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    auto keyDesc = context_->GetInputDesc(KEY_INPUT_INDEX);
    auto weightsDesc = context_->GetInputDesc(WEIGHT_INPUT_INDEX);
    auto sparseIndicesDesc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    auto attnSoftmaxL1Desc = context_->GetInputDesc(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseIndicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attnSoftmaxL1Desc);

    // 以下2个为16类型保持一致
    auto queryDtype = queryDesc->GetDataType();
    auto keyDtype = keyDesc->GetDataType();

    // 以下3个为32类型
    auto weightsDtype = weightsDesc->GetDataType();
    auto sparseIndicesDtype = sparseIndicesDesc->GetDataType();
    auto attnSoftmaxL1Dtype = attnSoftmaxL1Desc->GetDataType();

    bool same16 = false; // 判断输入为fp16或者部分16的是否一致
    bool same32 = false; // 判断输入为int32或者fp32的类型是否正确
    if (queryDtype == ge::DT_FLOAT16 && keyDtype == ge::DT_FLOAT16) {
        same16 = true;
    } else if (queryDtype == ge::DT_BF16 && keyDtype == ge::DT_BF16) {
        same16 = true;
    } else {
        OP_LOGW(context_, "InputDtype is not same.: queryDtype[%s], keyDtype[%s]",
                ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str());
        same16 = false;
    }
    if (weightsDtype == ge::DT_FLOAT && sparseIndicesDtype == ge::DT_INT32 && attnSoftmaxL1Dtype == ge::DT_FLOAT) {
        same32 = true;
    } else {
        OP_LOGW(context_,
                "Input dtype is invalid: weightsDtype must be float32, but got [%s]; sparseIndicesDtype must be int32, "
                "but got [%s]; attnSoftmaxL1Dtype must be float32, but got [%s].",
                ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(attnSoftmaxL1Dtype).c_str());
        same32 = false;
    }
    // 所有类型不满足返回false
    if (same16 == false || same32 == false) {
        return false;
    }
    OP_LOGW(context_, "InputDtype: queryDtype[%s], keyDtype[%s],",
            ge::TypeUtils::DataTypeToSerialString(queryDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(keyDtype).c_str());
    OP_LOGW(context_, "weightsDtype[%s], sparseIndicesDtype[%s], attnSoftmaxL1Dtype[%s].",
            ge::TypeUtils::DataTypeToSerialString(weightsDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(sparseIndicesDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(attnSoftmaxL1Dtype).c_str());
    return true;
}

// 输入shape进行交叉验证，防止数据错误输入
bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::CrossShapeVerify()
{
    auto queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto sparseIndicesShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();
    auto attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();
    // 下面数字对应shape输入位置
    if (layoutQuery[0] == 'T' && layoutQuery[1] == 'N' && layoutQuery[2] == 'D') {
        int64_t t1Len = queryIndexShape[0];
        int64_t n1Len = queryIndexShape[1];
        int64_t t2Len = keyIndexShape[0];
        int64_t n2Len = keyIndexShape[1];
        // 验证T1
        OP_CHECK_IF(weightsShape[0] != t1Len || sparseIndicesShape[0] != t1Len || attnSoftmaxL1Shape[0] != t1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify T1 is failed, the value of query_index[0], "
                        "weights[0], sparse_indices[0] and attn_softmax_l1_norm[0] "
                        "are respectively (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                        queryIndexShape[0], weightsShape[0], sparseIndicesShape[0], attnSoftmaxL1Shape[0]),
                    return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(queryIndexShape[1] < NQUERYINDEX_SIZE_1 && queryIndexShape[1] > NQUERYINDEX_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query_index must be >= 1 and <= "
                                                "64, but the value of query_index[1] "
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
        OP_CHECK_IF(sparseIndicesShape[1] != NKEYINDEX_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be 1, but the value of sparse_indices[1] is (%ld).",
                        sparseIndicesShape[1]),
                    return false);
        OP_CHECK_IF(
            attnSoftmaxL1Shape[1] != NKEYINDEX_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2 must be 1, but the value of attn_softmax_l1_norm[1] is (%ld).",
                attnSoftmaxL1Shape[1]),
            return false);
        // 验证N2
        OP_CHECK_IF(sparseIndicesShape[1] != keyIndexShape[1] || attnSoftmaxL1Shape[1] != keyIndexShape[1],
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify N2 is failed, the value of key_index[1], "
                                                "sparse_indices[1] and attn_softmax_l1_norm[1] "
                                                "are respectively (%ld), (%ld), (%ld). Their values should be equal.",
                                                keyIndexShape[1], sparseIndicesShape[1], attnSoftmaxL1Shape[1]),
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
        // 验证K
        OP_CHECK_IF(sparseIndicesShape[2] != attnSoftmaxL1Shape[2],
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify failed, shape K of sparse_indices and attn_softmax_l1_norm must be equal, "
                        "but the value "
                        "of sparse_indices[2] and attn_softmax_l1_norm[2] are respectively (%ld), (%ld).",
                        sparseIndicesShape[2], attnSoftmaxL1Shape[2]),
                    return false);
    } else if (layoutQuery[0] == 'B' && layoutQuery[1] == 'S' && layoutQuery[2] == 'N' && layoutQuery[3] == 'D') {
        int64_t bLen = queryIndexShape[0];
        int64_t s1Len = queryIndexShape[1];
        int64_t n1Len = queryIndexShape[2];
        int64_t s2Len = keyIndexShape[1];
        int64_t n2Len = keyIndexShape[2];

        // 验证B
        OP_CHECK_IF(keyIndexShape[0] != bLen || weightsShape[0] != bLen || sparseIndicesShape[0] != bLen ||
                        attnSoftmaxL1Shape[0] != bLen,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify shape B is failed, the value of query_index[0], "
                        "key_index[0], weights[0], sparse_indices[0] and attn_softmax_l1_norm[0] "
                        "are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                        bLen, keyIndexShape[0], weightsShape[0], sparseIndicesShape[0], attnSoftmaxL1Shape[0]),
                    return false);
        // 验证s1
        OP_CHECK_IF(weightsShape[1] != s1Len || sparseIndicesShape[1] != s1Len || attnSoftmaxL1Shape[1] != s1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify S1 is failed, the value of query_index[1], weights[1],"
                        "sparse_indices[1] and attn_softmax_l1_norm[1] are respectively (%ld), (%ld), (%ld), (%ld). "
                        "Their values should be equal.",
                        s1Len, weightsShape[1], sparseIndicesShape[1], attnSoftmaxL1Shape[1]),
                    return false);
        // 验证N Query数字是否正确
        OP_CHECK_IF(n1Len < NQUERYINDEX_SIZE_1 && n1Len > NQUERYINDEX_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query_index must be >= 1 and <= "
                                                "64, but the value of query_index[2]"
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
        OP_CHECK_IF(sparseIndicesShape[2] != NKEYINDEX_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be 1, but the value of sparse_indices[2] is (%ld).",
                        sparseIndicesShape[2]),
                    return false);
        OP_CHECK_IF(
            attnSoftmaxL1Shape[2] != NKEYINDEX_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2 must be 1, but the value of attn_softmax_l1_norm[2] is (%ld).",
                attnSoftmaxL1Shape[2]),
            return false);
        // 验证N2
        OP_CHECK_IF(sparseIndicesShape[2] != n2Len || attnSoftmaxL1Shape[2] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify N2 is failed, the value of key_index[2], "
                                                "sparse_indices[2] and attn_softmax_l1_norm[2] "
                                                "are respectively (%ld), (%ld), (%ld). Their values should be equal.",
                                                n2Len, sparseIndicesShape[2], attnSoftmaxL1Shape[2]),
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
        // 验证K
        OP_CHECK_IF(
            sparseIndicesShape[3] != attnSoftmaxL1Shape[3],
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape K of sparse_indices and attn_softmax_l1_norm must be equal, but the "
                "value of sparse_indices[2] and attn_softmax_l1_norm[2] are respectively (%ld), (%ld).",
                sparseIndicesShape[3], attnSoftmaxL1Shape[3]),
            return false);
    }
    return true;
}

bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::AnalyzeLayout()
{
    auto &queryIndexShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyIndexShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto &topKShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();
    auto &attnSoftmaxL1Shape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();

    size_t layoutLen = strlen(layoutQuery);
    OP_CHECK_IF(!CrossShapeVerify(), OPS_REPORT_VECTOR_INNER_ERR(opName, "CrossShapeVerify Failed"), return false);
    OP_CHECK_IF(!AnalyzeDimLayout(queryIndexShape, keyIndexShape, weightsShape, topKShape, layoutLen),
                OP_LOGE(opName, "Layout %s data analyze failed.", layoutQuery), return false);
    OP_CHECK_IF(gSizeQueryIndex == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSizeQueryIndex is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(dSizeQueryIndex <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "dSizeQueryIndex must be greater than 0"),
                return false);
    return true;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const SparseLightningIndexerGradCompileInfo *>(context_->GetCompileInfo());
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

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
                OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
    // 将基本输入参数传给tilingdata
    sliGradBaseParams_->set_bSize(bSize);
    sliGradBaseParams_->set_t1Size(t1Size);
    sliGradBaseParams_->set_t2Size(t2Size);
    sliGradBaseParams_->set_n2Size(n2Size);
    sliGradBaseParams_->set_gSizeQueryIndex(gSizeQueryIndex);
    sliGradBaseParams_->set_s1Size(s1Size);
    sliGradBaseParams_->set_s2Size(s2Size);
    sliGradBaseParams_->set_dSizeQueryIndex(dSizeQueryIndex);
    sliGradBaseParams_->set_kSize(kSize);
    sliGradBaseParams_->set_sparseMode(sparseMode);
    sliGradBaseParams_->set_cmpRatio(cmpRatio);

    OP_LOGW(context_,
            "INPUTPARAM bsize:[%ld], n2Size:[%ld], gSizeQueryIndex:[%ld],"
            "s1Size:[%ld], s2Size:[%ld], dSizeQueryIndex:[%ld],"
            "kSize:[%ld], sparseMode:[%ld], cmpRatio:[%ld].",
            bSize, n2Size, gSizeQueryIndex, s1Size, s2Size, dSizeQueryIndex, kSize, sparseMode, cmpRatio);

    return ge::GRAPH_SUCCESS;
}

int64_t SparseLightningIndexerKLLossGradTilingGeneralRegbase::GetS2RealSize(int32_t sparseMode, int32_t s1Size,
                                                                            int32_t s2Size, int32_t s1Idx)
{
    int64_t s2RealSize = 0;
    // sparsemode = 3 的情形
    if (sparseMode == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = static_cast<int64_t>((s2Size - s1Size) + s1Idx + 1);
        if (s2RealSize <= 0) {
            s2RealSize = static_cast<int64_t>(s2Size);
        }
    }
    s2RealSize = AlignUp(s2RealSize, 512L);
    return std::min(s2RealSize, (int64_t)sliGradBaseParams_->get_kSize());
}

bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray)
{
    OP_CHECK_IF(sparseValidArray.size() == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."), return false);

    uint32_t sparseMode = sliGradBaseParams_->get_sparseMode();
    if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_TND) {
        return true;
    } else if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_BSND) {
        int64_t accum = 0;
        for (int32_t i = 0; i < bSize; i++) {
            for (int64_t j = 0; j < s1Size; j++) {
                int64_t s2RealSize = GetS2RealSize(sparseMode, s1Size, s2Size, j);
                sparseValidArray[accum] = s2RealSize;
                accum++;
            }
        }
    }

    return true;
}

inline bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::InitLoadValue(
    const std::vector<int64_t> &sparseValidArray, int64_t validAicNum, int64_t totalSize,
    const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
{
    for (int64_t idx = 0; idx < validAicNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAicNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
        } else {
            break;
        }
    }
    return true;
}

bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                                                                       std::vector<int64_t> &localValue,
                                                                       std::vector<int64_t> &sparseStartIdx)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAicNum =
        std::min(static_cast<int64_t>(sliGradMultiCoreParams_->get_coreNum()), static_cast<int64_t>(aicNum));
    int64_t totalSize = sliGradMultiCoreParams_->get_totalSize();
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAicNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAicNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::Balance4DLoad(std::vector<int64_t> &tmpSparseValue,
                                                                         const std::vector<int64_t> sparseValidArray,
                                                                         const int64_t balanceNum)
{
    int64_t tmpIndex = 0;
    tmpSparseValue[tmpIndex] = 0;
    int64_t sumTmpArray = 0;
    int64_t sumTmpArrayLast = 0; // 记录上次的总和值
    for (int64_t idx = 0; idx < sparseValidArray.size(); ++idx) {
        // 第一次分到最后一块后就没必须计算，剩余的直接分给最后一个核即可
        if (tmpIndex == static_cast<int64_t>(tmpSparseValue.size()) - 1) {
            break;
        }

        sumTmpArrayLast = sumTmpArray;
        sumTmpArray += sparseValidArray[idx];
        if (sumTmpArray == balanceNum) {
            tmpIndex = (tmpIndex + 1 < tmpSparseValue.size()) ? tmpIndex + 1 : tmpSparseValue.size() - 1;
            tmpSparseValue[tmpIndex] = idx + 1; // 刚好等于均值时核的末尾为idx
            sumTmpArray = 0;                    // 重新计算总和
        } else if (sumTmpArray > balanceNum) {
            tmpIndex = (tmpIndex + 1 < tmpSparseValue.size()) ? tmpIndex + 1 : tmpSparseValue.size() - 1;
            // 第一次总和大于均值，判断前面一次和当前哪个更接近均值
            if (balanceNum - sumTmpArrayLast >= sumTmpArray - balanceNum) {
                // 当前更接近，取当前值
                tmpSparseValue[tmpIndex] = idx + 1; // 刚好等于均值时核的末尾为idx
            } else {
                // 上一次更接近， 取上一次值
                tmpSparseValue[tmpIndex] = idx; // 刚好等于均值时核的末尾为idx
                idx--; // 记录的上一个值，重新计算时也需要从上一个值开始计算
            }
            sumTmpArray = 0; // 重新计算总和
        }
    }
    return true;
}

// 负载均衡
bool SparseLightningIndexerKLLossGradTilingGeneralRegbase::SetSparseStartIdx(
    const std::vector<int64_t> &sparseValidArray, int64_t maxCoreNum)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAicNum = static_cast<int64_t>(sliGradMultiCoreParams_->get_coreNum());
    int64_t totalSize = sliGradMultiCoreParams_->get_totalSize();
    int64_t *sparseStartIdx = sliGradMultiCoreParams_->get_bS1Ptr();
    int64_t splitFactorSize = sliGradMultiCoreParams_->get_splitFactorSize();

    OP_CHECK_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "totalSize should be larger than 0."),
                return false);
    if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_TND) {
        return true;
    } else if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_BSND) {
        int64_t sparseArraySum =
            std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有BS上的S2的总数量
        int64_t balanceNum = CeilDivision(sparseArraySum, validAicNum);
        std::vector<int64_t> tmpSparseValue(validAicNum, 0);
        Balance4DLoad(tmpSparseValue, sparseValidArray, balanceNum);
        for (int64_t idx = 0; idx < static_cast<int64_t>(validAicNum); ++idx) {
            sparseStartIdx[idx] = tmpSparseValue[idx];
        }
    }

    for (int64_t idx = 1; idx < static_cast<int64_t>(MAX_CORE_NUM_REGBASE); ++idx) {
        if (sparseStartIdx[idx] == 0) {
            sparseStartIdx[idx] = static_cast<int64_t>(sparseValidArray.size()); // 赋值为s1最大值
        }
    }
    return true;
}

void SparseLightningIndexerKLLossGradTilingGeneralRegbase::SetSparseParamsRegbase(int64_t maxCoreNum)
{
    std::vector<int64_t> sparseValidArray(sliGradMultiCoreParams_->get_totalSize(), 0);
    InitSparseValidArray(sparseValidArray);
    SetSparseStartIdx(sparseValidArray, maxCoreNum);
}

// 计算S1的总个数
int64_t SparseLightningIndexerKLLossGradTilingGeneralRegbase::CalcTotalSize()
{
    if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_TND) {
        return realT1Size;
    } else if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_BSND) {
        return bSize * s1Size;
    }
    return 0; // 什么也不走就返回0
}

void SparseLightningIndexerKLLossGradTilingGeneralRegbase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(coreNum));
    sliGradMultiCoreParams_->set_coreNum(static_cast<int32_t>(aicNum));
    sliGradMultiCoreParams_->set_totalSize(totalSize);
    sliGradMultiCoreParams_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
}

void SparseLightningIndexerKLLossGradTilingGeneralRegbase::InitOutputSplit()
{
    SLIGradInitOutputParamsRegbase *initoutput = &tilingData->initOutputParams;
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

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::DoCastTiling()
{
    // softmaxOut、dw、dq
    int64_t dAlign16 = (sliGradBaseParams_->get_dSizeQueryIndex() + 15) >> 4 << 4;
    // query
    int64_t allNumQuery = sliGradBaseParams_->get_bSize() * sliGradBaseParams_->get_n2Size() *
                          sliGradBaseParams_->get_gSizeQueryIndex() * sliGradBaseParams_->get_s1Size() * dAlign16;
    // TND时候要按照真实的query的num数计算
    if (sliGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        allNumQuery = sliGradBaseParams_->get_t1Size() * sliGradBaseParams_->get_n2Size() *
                      sliGradBaseParams_->get_gSizeQueryIndex() * dAlign16;
    }

    // weight
    int64_t allNumWeight = sliGradBaseParams_->get_bSize() * sliGradBaseParams_->get_n2Size() *
                           sliGradBaseParams_->get_s1Size() * sliGradBaseParams_->get_gSizeQueryIndex();
    // TND时候要按照真实的cmp_softmax_l1的num数计算
    if (sliGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        allNumWeight = sliGradBaseParams_->get_t1Size() * sliGradBaseParams_->get_n2Size() *
                       sliGradBaseParams_->get_gSizeQueryIndex();
    }

    // softmaxOut
    int64_t allNumSoftmaxOut = sliGradBaseParams_->get_bSize() * sliGradBaseParams_->get_s1Size() *
                               sliGradBaseParams_->get_n2Size() * sliGradBaseParams_->get_kSize();
    // TND时候要按照真实的softmaxOut的num数计算
    if (sliGradBaseParams_->get_layoutType() == static_cast<uint8_t>(LayoutTypeRegbase::LAYOUT_TND)) {
        allNumSoftmaxOut =
            sliGradBaseParams_->get_t1Size() * sliGradBaseParams_->get_n2Size() * sliGradBaseParams_->get_kSize();
    }
    uint32_t typeSize = B16;
    uint32_t usedCoreNum = aivNum;
    uint32_t coreNum = aicNum;
    constexpr uint32_t postNzCoexNode = 12;
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t postNzReservedN = 1;

    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t nzReservedSize = 0;
    int64_t curPostCoexNode = postNzCoexNode;
    postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / // 开DB预留2份nzReservedSize
                     BASE_LEN_256 * BASE_LEN_256;
    OP_LOGI(context_, "DoCastTiling postUbBaseSize: %ld.", postUbBaseSize);
    qPostBaseNum = 128 * 128;
    OP_LOGI(context_, "DoCastTiling qPostBaseNum: %ld.", qPostBaseNum);

    OP_CHECK_IF(qPostBaseNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "castUsedCoreNum is 0."),
                return ge::GRAPH_FAILED);

    int64_t qPreBlockFactor = (allNumQuery + usedCoreNum - 1) / usedCoreNum;
    int64_t qPreBlockTotal = (allNumQuery + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreBlockTailTmp = allNumQuery % qPreBlockFactor;
    int64_t qPreBlockTail = qPreBlockTailTmp == 0 ? qPreBlockFactor : qPreBlockTailTmp;

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

    sliGradMultiCoreParams_->set_postUbBaseSize(postUbBaseSize);

    sliGradMultiCoreParams_->set_qPreBlockFactor(qPreBlockFactor);
    sliGradMultiCoreParams_->set_qPreBlockTotal(qPreBlockTotal);
    sliGradMultiCoreParams_->set_qPreBlockTail(qPreBlockTail);

    sliGradMultiCoreParams_->set_weightPreBlockFactor(weightPreBlockFactor);
    sliGradMultiCoreParams_->set_weightPreBlockTotal(weightPreBlockTotal);
    sliGradMultiCoreParams_->set_weightPreBlockTail(weightPreBlockTail);

    sliGradMultiCoreParams_->set_softmaxOutPreBlockFactor(softmaxOutPreBlockFactor);
    sliGradMultiCoreParams_->set_softmaxOutPreBlockTotal(softmaxOutPreBlockTotal);
    sliGradMultiCoreParams_->set_softmaxOutPreBlockTail(softmaxOutPreBlockTail);

    dqWorkspaceLen = (allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    dwWorkspaceLen = (allNumWeight * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    softmaxOutWorkspaceLen = (allNumSoftmaxOut * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::DoOpTiling()
{
    context_->SetScheduleMode(1);
    OP_LOGD(context_, "try template[%s]", templateName);
    // 无多余操作，分核，目前只实现TND场景分核
    int64_t totalSize = CalcTotalSize();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    context_->SetBlockDim(aicNum); // 使用的核数确定

    std::vector<int64_t> shapeVec = {1, kSize};
    ge::Shape srcShape(shapeVec);
    int64_t softmaxTmpBufferSize = BUFFER_SIZE_BYTE_32K; // 需要32KB
    if (kSize > BUFFER_SIZE_BYTE_2K) {
        softmaxTmpBufferSize = BUFFER_SIZE_BYTE_33K; // kSize >2048 需要33kB
    }
    SoftMaxTilingFunc(srcShape, 4, softmaxTmpBufferSize, tilingData->vectorParams.softmaxYTilingData);

    SetSparseParamsRegbase(static_cast<int64_t>(aicNum));
    InitOutputSplit(); // output分核
    auto status = DoCastTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    OP_LOGD(context_, "ending template[%s]", templateName);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::PostTiling()
{
    OP_LOGI(context_, "====================================== BASE PARAMS ===========================================");
    OP_LOGI(context_, "sliGradBaseParams_.bSize = [%ld]", sliGradBaseParams_->get_bSize());
    OP_LOGI(context_, "sliGradBaseParams_.cmpRatio = [%ld]", sliGradBaseParams_->get_cmpRatio());
    OP_LOGI(context_, "sliGradBaseParams_.dSizeQueryIndex = [%ld]", sliGradBaseParams_->get_dSizeQueryIndex());
    OP_LOGI(context_, "sliGradBaseParams_.gSizeQueryIndex = [%ld]", sliGradBaseParams_->get_gSizeQueryIndex());
    OP_LOGI(context_, "sliGradBaseParams_.kSize = [%ld]", sliGradBaseParams_->get_kSize());
    OP_LOGI(context_, "sliGradBaseParams_.layoutType = [%ld]", sliGradBaseParams_->get_layoutType());
    OP_LOGI(context_, "sliGradBaseParams_.n2Size = [%ld]", sliGradBaseParams_->get_n2Size());
    OP_LOGI(context_, "sliGradBaseParams_.s1Size = [%ld]", sliGradBaseParams_->get_s1Size());
    OP_LOGI(context_, "==================================== MULTI CORE PARAMS =======================================");
    OP_LOGI(context_, "sliGradMultiCoreParams_.coreNum = [%ld]", sliGradMultiCoreParams_->get_coreNum());
    OP_LOGI(context_, "sliGradMultiCoreParams_.splitFactorSize = [%ld]",
            sliGradMultiCoreParams_->get_splitFactorSize());
    OP_LOGI(context_, "sliGradMultiCoreParams_.totalSize = [%ld]", sliGradMultiCoreParams_->get_totalSize());
    OP_LOGI(context_, "===================================== INIT OUTPUT PARAMS =====================================");
    SLIGradInitOutputParamsRegbase *initoutput = &tilingData->initOutputParams;
    OP_LOGI(context_, "initoutput.singleCoreSize = [%ld]", initoutput->get_singleCoreSize());
    OP_LOGI(context_, "initoutput.totalOutputSize = [%ld]", initoutput->get_totalOutputSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SparseLightningIndexerKLLossGradTilingGeneralRegbase::GetTilingKey() const
{
    bool hasCuSeqlensQ = true;
    bool hasCuSeqlensK = true;
    bool hasSequsedQ = true;
    bool hasSequsedK = true;
    bool hasCmpResidualK = true;
    if (context_->GetOptionalInputTensor(CU_SEQLENS_QUERY_INPUT_INDEX) == nullptr) {
        hasCuSeqlensQ = false;
    }
    if (context_->GetOptionalInputTensor(CU_SEQLENS_KEY_INPUT_INDEX) == nullptr) {
        hasCuSeqlensK = false;
    }
    if (context_->GetOptionalInputTensor(SEQUSED_QUERY_INPUT_INDEX) == nullptr) {
        hasSequsedQ = false;
    }
    if (context_->GetOptionalInputTensor(SEQUSED_KEY_INPUT_INDEX) == nullptr) {
        hasSequsedK = false;
    }
    if (context_->GetOptionalInputTensor(CMP_RESIDUAL_KEY_INPUT_INDEX) == nullptr) {
        hasCmpResidualK = false;
    }
    return GET_TPL_TILING_KEY(static_cast<uint8_t>(topKRange), static_cast<uint8_t>(tilingKeyLayout),
                              static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(sparseMode),
                              static_cast<uint8_t>(hasCuSeqlensQ), static_cast<uint8_t>(hasCuSeqlensK),
                              static_cast<uint8_t>(hasSequsedQ), static_cast<uint8_t>(hasSequsedK),
                              static_cast<uint8_t>(hasCmpResidualK), static_cast<uint8_t>(deterministic));
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingGeneralRegbase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int32_t kSizeAlign128 = (kSize + 127) >> 7 << 7;
    int64_t reduceSumOffset = kSizeAlign128 * sizeof(float);
    int64_t reluOffset = gSizeQueryIndex * kSizeAlign128 * sizeof(float);
    int64_t gatherSYOffset = kSizeAlign128 * dSizeQueryIndex * 2; // 2代表输入D类型大小
    int64_t bmm3Size = kSize * dSizeQueryIndex * sizeof(float) * PING_PONG_VALUE;

    int64_t scatterAddOutSize;
    if (tilingKeyLayout == LayoutTypeRegbase::LAYOUT_TND) {
        scatterAddOutSize = accumS2 * dSizeQueryIndex * sizeof(float); // batch
    } else {
        scatterAddOutSize = bSize * s2Size * dSizeQueryIndex * sizeof(float); // batch
    }

    int64_t singlecoreTotalSize = PING_PONG_VALUE * reduceSumOffset + reluOffset + gatherSYOffset;
    int64_t multicoreTotalsize = 0;
    if (deterministic) {
        multicoreTotalsize =
            (singlecoreTotalSize + bmm3Size) * static_cast<int64_t>(sliGradMultiCoreParams_->get_coreNum()) +
            scatterAddOutSize * 2;
    } else {
        multicoreTotalsize =
            singlecoreTotalSize * static_cast<int64_t>(sliGradMultiCoreParams_->get_coreNum()) + scatterAddOutSize;
    }
    workspaces[0] = static_cast<size_t>(multicoreTotalsize) + WORK_SPACE_RESERVE_SIZE; // 预留16M空间必须加;
    OP_LOGW(context_, "workspace size:[%ld], multicoreTotalsize:[%ld]", workspaces[0], multicoreTotalsize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(SparseLightningIndexerKLLossGrad,
                                   SparseLightningIndexerKLLossGradTilingGeneralRegbase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace optiling
