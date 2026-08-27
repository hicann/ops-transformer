/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_grad_kl_loss_tiling.cpp
 * \brief
 */
#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include <limits>
#include <tiling/tiling_api.h>
using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus DenseLightningIndexerGradKLLossTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    // 输入属性参数验证
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    // 输入shape验证
    auto queryShape = context_->GetInputShape(0);
    auto keyShape = context_->GetInputShape(1);
    auto queryIndexShape = context_->GetInputShape(2);
    auto keyIndexShape = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyIndexShape);

    // 输出shape验证
    auto dQueryIndexShape = context_->GetOutputShape(D_QUERY_INDEX_OUTPUT_INDEX);
    auto dkeyIndexShape = context_->GetOutputShape(D_KEY_INDEX_OUTPUT_INDEX);
    auto dWeightsShape = context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX);
    auto lossShape = context_->GetOutputShape(LOSS_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dQueryIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dkeyIndexShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dWeightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, lossShape);

    // RawTilingData验证
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());

    return ge::GRAPH_SUCCESS;
}

bool DenseLightningIndexerGradKLLossTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    scaleValue = *scaleValuePtr;
    inputLayout = inputLayoutPtr;
    sparseMode = *sparseModePtr;
    deterministic = (context_->GetDeterministic() == 1);
    OP_LOGD(context_->GetNodeName(), "attrs: scaleValue[%f] input_layout[%s] sparse_mode[%ld] deterministic[%d].",
            scaleValue, inputLayout, sparseMode, deterministic);
    return true;
}

bool DenseLightningIndexerGradKLLossTilingBase::AnalyzeDtype()
{
    // 以下5个保持一致
    auto queryDtype = context_->GetInputDesc(QUERY_INPUT_INDEX)->GetDataType();
    auto keyDtype = context_->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    auto queryIndexDtype = context_->GetInputDesc(QUERY_INDEX_INPUT_INDEX)->GetDataType();
    auto keyIndexDtype = context_->GetInputDesc(KEY_INDEX_INPUT_INDEX)->GetDataType();
    auto weightsDtype = context_->GetInputDesc(WEIGHT_INPUT_INDEX)->GetDataType();
    if (queryDtype == ge::DT_FLOAT16 && keyDtype == ge::DT_FLOAT16 && queryIndexDtype == ge::DT_FLOAT16 &&
        keyIndexDtype == ge::DT_FLOAT16 && (weightsDtype == ge::DT_FLOAT16 || weightsDtype == ge::DT_FLOAT)) {
        OP_LOGD(opName, "q/k/weight inputDtype is fp16.");
    } else if (queryDtype == ge::DT_BF16 && keyDtype == ge::DT_BF16 && queryIndexDtype == ge::DT_BF16 &&
               keyIndexDtype == ge::DT_BF16 && (weightsDtype == ge::DT_BF16 || weightsDtype == ge::DT_FLOAT)) {
        OP_LOGD(opName, "q/k/weight inputDtype is bf16.");
    } else {
        OP_LOGE(opName, "q/k/weight inputDtype is not same.");
        return false;
    }

    // 以下4个为32类型
    auto softmaxMaxDtype = context_->GetInputDesc(SOFTMAX_MAX_INPUT_INDEX)->GetDataType();
    auto softmaxSumDtype = context_->GetInputDesc(SOFTMAX_SUM_INPUT_INDEX)->GetDataType();
    auto softmaxMaxIndexDtype = context_->GetInputDesc(SOFTMAX_MAX_INDEX_INPUT_INDEX)->GetDataType();
    auto softmaxSumIndexDtype = context_->GetInputDesc(SOFTMAX_SUM_INDEX_INPUT_INDEX)->GetDataType();
    if (softmaxMaxDtype == ge::DT_FLOAT && softmaxSumDtype == ge::DT_FLOAT && softmaxMaxIndexDtype == ge::DT_FLOAT &&
        softmaxSumIndexDtype == ge::DT_FLOAT) {
        OP_LOGD(opName, "softmax inputDtype is fp32.");
    } else {
        OP_LOGE(opName, "softmax inputDtype is not same.");
        return false;
    }

    return true;
}

bool DenseLightningIndexerGradKLLossTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto &keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX)->GetStorageShape();

    auto queryRope = context_->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX);
    bool hasQueryRope = queryRope != nullptr && queryRope->GetStorageShape().GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX);
    bool hasKeyRope = keyRope != nullptr && keyRope->GetStorageShape().GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time, check this.");
        return false;
    }
    // TND和BSND的dim位置不同来赋值
    hasRope = hasQueryRope && hasKeyRope;
    const gert::Shape emptyRopeShape;
    const auto &queryRopeShape = hasQueryRope ? queryRope->GetStorageShape() : emptyRopeShape;
    const auto &keyRopeShape = hasKeyRope ? keyRope->GetStorageShape() : emptyRopeShape;

    size_t layoutLen = strlen(inputLayout);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen ||
                    queryIndexShape.GetDimNum() != layoutLen || keyIndexShape.GetDimNum() != layoutLen,
                OP_LOGE(opName, "Invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!CrossShapeVerify(queryRopeShape, keyRopeShape),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Dense train CrossShapeVerify Failed"), return false);
    OP_CHECK_IF(!Analyze3DimLayout(queryShape, keyShape, queryIndexShape, keyIndexShape, layoutLen, queryRopeShape,
                                   keyRopeShape),
                OP_LOGE(opName, "Layout: %s, Run Failed", inputLayout), return false);
    OP_CHECK_IF(gSizeQuery == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSizeQuery is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(n2IndexSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2IndexSize is zero"), return false);
    OP_CHECK_IF(dSizeQuery <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "dSizeQuery  is not support <= 0"), return false);
    return true;
}

// 输入shape进行交叉验证，防止数据错误输入
bool DenseLightningIndexerGradKLLossTilingBase::CrossShapeVerify(const gert::Shape &queryRopeShape,
                                                                 const gert::Shape &keyRopeShape)
{
    auto queryShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto keyShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto queryIndexShape = context_->GetInputShape(QUERY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto keyIndexShape = context_->GetInputShape(KEY_INDEX_INPUT_INDEX)->GetStorageShape();
    auto weightsShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto softmaxMaxShape = context_->GetInputShape(SOFTMAX_MAX_INPUT_INDEX)->GetStorageShape();
    auto softmaxSumShape = context_->GetInputShape(SOFTMAX_SUM_INPUT_INDEX)->GetStorageShape();
    auto softmaxMaxIndexShape = context_->GetInputShape(SOFTMAX_MAX_INDEX_INPUT_INDEX)->GetStorageShape();
    auto softmaxSumIndexShape = context_->GetInputShape(SOFTMAX_SUM_INDEX_INPUT_INDEX)->GetStorageShape();

    if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
        int64_t t1Len = queryShape[0];
        int64_t t2Len = keyShape[0];
        int64_t n1Len = queryShape[1];
        int64_t n2Len = keyShape[1];
        int64_t n1IndexLen = queryIndexShape[1];
        int64_t n2IndexLen = keyIndexShape[1];

        // 数值验证N1 是否在支持的范围内
        OP_CHECK_IF(n1Len != NQUERY_SIZE_32 && n1Len != NQUERY_SIZE_64 && n1Len != NQUERY_SIZE_128,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query must be one of the {32, 64, "
                                                "128}, but the value of query[1] is (%ld)",
                                                n1Len),
                    return false);
        // 数值验证N2=N1
        OP_CHECK_IF(
            n2Len != n1Len,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2 must be same with N1, but the value of key[1] is (%ld).", n2Len),
            return false);
        // 数值验证N1_Index 是否在支持的范围内
        OP_CHECK_IF(n1IndexLen != NQUERYINDEX_SIZE_8 && n1IndexLen != NQUERYINDEX_SIZE_16 &&
                        n1IndexLen != NQUERYINDEX_SIZE_32 && n1IndexLen != NQUERYINDEX_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query_index must be one of the "
                                                "{8, 16, 32, 64}, but the value of query_index[1] is (%ld).",
                                                n1IndexLen),
                    return false);
        // 数值验证N2_INDEX=1
        OP_CHECK_IF(n2IndexLen != N2_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2_IDX must be 1, but the value of key_index[1] is (%ld).",
                        n2IndexLen),
                    return false);
        // 数值验证D 是否在支持的范围内
        OP_CHECK_IF(
            queryShape[2] != D_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, shape D of query must be 128, but the value of query[2] is (%ld)",
                queryShape[2]),
            return false);
        OP_CHECK_IF(
            queryIndexShape[2] != D_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[2] is (%ld)",
                queryIndexShape[2]),
            return false);

        // 交叉验证T1
        OP_CHECK_IF(
            queryIndexShape[0] != t1Len || weightsShape[0] != t1Len || softmaxMaxShape[1] != t1Len ||
                softmaxSumShape[1] != t1Len || softmaxMaxIndexShape[1] != t1Len || softmaxSumIndexShape[1] != t1Len,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify T1 is failed, the value of query[0], query_index[0], weights[0], softmax_max[1] \
                softmax_sum[1], softmax_max_index[1] and softmax_sum_index[1] are respectively (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                queryShape[0], queryIndexShape[0], weightsShape[0], softmaxMaxShape[1], softmaxSumShape[1],
                softmaxMaxIndexShape[1], softmaxSumIndexShape[1]),
            return false);
        // 交叉验证T2
        OP_CHECK_IF(keyIndexShape[0] != t2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify T2 is failed, the value of key[0] and key_index[0] "
                                                "are respectively (%ld), (%ld). Their values should be equal.",
                                                t2Len, keyIndexShape[0]),
                    return false);
        // 交叉验证N1_Index
        OP_CHECK_IF(
            weightsShape[1] != n1IndexLen,
            OPS_REPORT_VECTOR_INNER_ERR(opName,
                                        "CrossShapeVerify N index is failed, the value of query_index[1] and "
                                        "weights[1] are respectively (%ld), (%ld). Their values should be equal.",
                                        n1IndexLen, weightsShape[1]),
            return false);
        // 交叉验证N2
        OP_CHECK_IF(softmaxMaxShape[0] != n2Len || softmaxSumShape[0] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify failed, the value of key[1], softmax_max[0] and softmax_sum[0] are "
                        "respectively (%ld), (%ld), (%ld). Their values should be equal.",
                        n2Len, softmaxMaxShape[0], softmaxSumShape[0]),
                    return false);
        // 交叉验证N2_INDEX
        OP_CHECK_IF(softmaxMaxIndexShape[0] != n2IndexLen || softmaxSumIndexShape[0] != n2IndexLen,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify N2_INDEX is failed, the value of key_index[1], softmax_max_index[0] and "
                        "softmax_sum_index[0] are respectively (%ld), (%ld), (%ld). Their values should be equal.",
                        n2IndexLen, softmaxMaxIndexShape[0], softmaxSumIndexShape[0]),
                    return false);
        // 交叉验证D
        OP_CHECK_IF(
            keyShape[2] != queryShape[2],
            OPS_REPORT_VECTOR_INNER_ERR(opName,
                                        "CrossShapeVerify query-key shape D is failed, the value of query[2] and "
                                        "key[2] are respectively (%ld), (%ld). Their values should be equal.",
                                        queryShape[2], keyShape[2]),
            return false);
        OP_CHECK_IF(queryIndexShape[2] != keyIndexShape[2],
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify query-key_index shape D is failed, the value of query_index[2] and "
                        "key_index[2] are respectively (%ld), (%ld). Their values should be equal.",
                        queryIndexShape[2], keyIndexShape[2]),
                    return false);

        // 验证ROPE
        if (hasRope) {
            // 验证queryrope
            OP_CHECK_IF(
                queryRopeShape[0] != t1Len || queryRopeShape[1] != n1Len,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "CrossShapeVerify query_rope is failed, the value of query_rope[0] is (%ld), it should be equal to tlSize(%ld). \
                    the value of query_rope[1] is (%ld), it should be equal to nQuerySize(%ld).",
                    queryRopeShape[0], t1Len, queryRopeShape[1], n1Len),
                return false);
            // 验证keyrope
            OP_CHECK_IF(
                keyRopeShape[0] != t2Len || keyRopeShape[1] != n2Len,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "CrossShapeVerify key_rope is failed, the value of key_rope[0] is (%ld), it should be equal to t2Size(%ld). \
                    the value of key_rope[1] is (%ld), it should be equal to n2Size(%ld)",
                    keyRopeShape[0], t2Len, keyRopeShape[1], n2Len),
                return false);
            // 验证rope D 数字是否正确
            OP_CHECK_IF(queryRopeShape[2] != DROPE_SIZE_64,
                        OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                    "CrossShapeVerify failed, shape D of query_rope must be 64, but "
                                                    "the value of query_rope[2] is (%ld).",
                                                    queryRopeShape[2]),
                        return false);
            // 验证rope D
            OP_CHECK_IF(queryRopeShape[2] != keyRopeShape[2],
                        OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                    "CrossShapeVerify query_rope D is not equal to key_rope D, the "
                                                    "value of query_rope[2] and key_rope[2] is (%ld), (%ld).",
                                                    queryRopeShape[2], keyRopeShape[2]),
                        return false);
        }
    } else if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
        int64_t bLen = queryShape[0];
        int64_t s1Len = queryShape[1];
        int64_t s2Len = keyShape[1];
        int64_t n1Len = queryShape[2];
        int64_t n2Len = keyShape[2];
        int64_t n1IndexLen = queryIndexShape[2];
        int64_t n2IndexLen = keyIndexShape[2];

        // 数值验证N1 是否在支持的范围内
        OP_CHECK_IF(n1Len != NQUERY_SIZE_32 && n1Len != NQUERY_SIZE_64 && n1Len != NQUERY_SIZE_128,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query must be one of the {32, 64, "
                                                "128}, but the value of query[2] is (%ld)",
                                                n1Len),
                    return false);
        // 数值验证N2=N1
        OP_CHECK_IF(n2Len != n1Len,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2 must be same with N1, but N2 is (%ld), N1 is (%ld).",
                        n2Len, n1Len),
                    return false);
        // 数值验证N1_Index 是否在支持的范围内
        OP_CHECK_IF(n1IndexLen != NQUERYINDEX_SIZE_8 && n1IndexLen != NQUERYINDEX_SIZE_16 &&
                        n1IndexLen != NQUERYINDEX_SIZE_32 && n1IndexLen != NQUERYINDEX_SIZE_64,
                    OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                "CrossShapeVerify failed, shape N of query_index must be one of the "
                                                "{8, 16, 32, 64}, but the value of query_index[2] is (%ld).",
                                                n1IndexLen),
                    return false);
        // 数值验证N2_INDEX=1
        OP_CHECK_IF(n2IndexLen != N2_SIZE_1,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, N2_IDX must be 1, but the value of key_index[2] is (%ld).",
                        n2IndexLen),
                    return false);

        // 交叉验证B
        OP_CHECK_IF(
            queryIndexShape[0] != bLen || weightsShape[0] != bLen || softmaxMaxShape[0] != bLen ||
                softmaxSumShape[0] != bLen || keyShape[0] != bLen || keyIndexShape[0] != bLen ||
                softmaxMaxIndexShape[0] != bLen || softmaxSumIndexShape[0] != bLen,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify shape B is failed, the value of query[0], key[0], query_index[0], key_index[0], weights[0], softmax_max[0], softmax_sum[0], softmax_max_index[0] and softmax_sum_index[0] are respectively (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld).\
                Their values should be equal.",
                queryShape[0], keyShape[0], queryIndexShape[0], keyIndexShape[0], weightsShape[0], softmaxMaxShape[0],
                softmaxSumShape[0], softmaxMaxIndexShape[0], softmaxSumIndexShape[0]),
            return false);
        // 交叉验证s1
        OP_CHECK_IF(
            queryIndexShape[1] != s1Len || weightsShape[1] != s1Len || softmaxMaxShape[2] != s1Len ||
                softmaxSumShape[2] != s1Len || softmaxMaxIndexShape[2] != s1Len || softmaxSumIndexShape[2] != s1Len,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify S1 is failed, the value of query[1], query_index[1], weights[1], softmax_max[2],  softmax_sum[2], softmax_max_index[2]\
                and softmax_sum_index[2] are respectively (%ld), (%ld), (%ld), (%ld), (%ld), (%ld), (%ld). Their values should be equal.",
                queryShape[1], queryIndexShape[1], weightsShape[1], softmaxMaxShape[2], softmaxSumShape[2],
                softmaxMaxIndexShape[2], softmaxSumIndexShape[2]),
            return false);
        // 交叉验证s2
        OP_CHECK_IF(
            keyIndexShape[1] != s2Len,
            OPS_REPORT_VECTOR_INNER_ERR(opName,
                                        "CrossShapeVerify shape S2 is failed, the value of key[1] and key_index[1] are "
                                        "respectively (%ld), (%ld). Their values should be equal.",
                                        keyShape[1], keyIndexShape[1]),
            return false);
        // 交叉验证N1_Index
        OP_CHECK_IF(
            queryIndexShape[2] != weightsShape[2],
            OPS_REPORT_VECTOR_INNER_ERR(opName,
                                        "CrossShapeVerify N index is failed, the value of query_index[2] and "
                                        "weights[2] are respectively (%ld), (%ld). Their values should be equal.",
                                        queryIndexShape[2], weightsShape[2]),
            return false);
        // 交叉验证N2
        OP_CHECK_IF(softmaxMaxShape[1] != n2Len || softmaxSumShape[1] != n2Len,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName,
                        "CrossShapeVerify N2 is failed, the value of key[2], softmax_max[1] and softmax_sum[1] are "
                        "respectively (%ld), (%ld), (%ld). Their values should be equal.",
                        keyShape[2], softmaxMaxShape[1], softmaxSumShape[1]),
                    return false);
        // 交叉验证N2IDX=1
        OP_CHECK_IF(
            softmaxMaxIndexShape[1] != N2_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2_INDEX must be 1, but the value of softmax_max_index[2] is (%ld).",
                softmaxMaxIndexShape[1]),
            return false);
        OP_CHECK_IF(
            softmaxSumIndexShape[1] != N2_SIZE_1,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, N2_INDEX must be 1, but the value of softmax_sum_index[2] is (%ld).",
                softmaxSumIndexShape[1]),
            return false);
        // 交叉验证D=128
        OP_CHECK_IF(
            queryShape[3] != D_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName, "CrossShapeVerify failed, shape D of query must be 128, but the value of query[3] is (%ld)",
                queryShape[3]),
            return false);
        OP_CHECK_IF(keyShape[3] != D_SIZE_128,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        opName, "CrossShapeVerify failed, shape D of key must be 128, but the value of key[3] is (%ld)",
                        keyShape[3]),
                    return false);
        OP_CHECK_IF(
            queryIndexShape[3] != D_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of query_index must be 128, but the value of query_index[3] is (%ld)",
                queryIndexShape[3]),
            return false);
        OP_CHECK_IF(
            keyIndexShape[3] != D_SIZE_128,
            OPS_REPORT_VECTOR_INNER_ERR(
                opName,
                "CrossShapeVerify failed, shape D of key_index must be 128, but the value of key_index[3] is (%ld)",
                keyIndexShape[3]),
            return false);

        // 验证ROPE相关参数
        if (hasRope) {
            // 验证queryrope
            OP_CHECK_IF(
                queryRopeShape[0] != bLen || queryRopeShape[1] != s1Len || queryRopeShape[2] != n1Len,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "CrossShapeVerify query_rope is failed, the value of query_rope[0] is (%ld), it should be equal to bSize(%ld). \
                    the value of query_rope[1] is (%ld), it should be equal to s1Size(%ld), the value of query_rope[2] is (%ld), it should be equal to nQuerySize(%ld).",
                    queryRopeShape[0], bLen, queryRopeShape[1], s1Len, queryRopeShape[2], n1Len),
                return false);
            // 验证keyrope
            OP_CHECK_IF(
                keyRopeShape[0] != bLen || keyRopeShape[1] != s2Len || keyRopeShape[2] != n2Len,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "CrossShapeVerify key_rope is failed, the value of key_rope[0] is (%ld), it should be equal to bSize(%ld), the value of key_rope[1] is (%ld), it should be equal to s2Size(%ld). \
                    the value of key_rope[2] is (%ld), it should be equal to n2Size(%ld)",
                    keyRopeShape[0], bLen, keyRopeShape[1], s2Len, keyRopeShape[2], n2Len),
                return false);
            // 验证rope D 数字是否正确
            OP_CHECK_IF(queryRopeShape[3] != DROPE_SIZE_64,
                        OPS_REPORT_VECTOR_INNER_ERR(opName,
                                                    "CrossShapeVerify failed, shape D of query_rope  must be 64, but "
                                                    "the value of query_rope[3] is (%ld).",
                                                    queryRopeShape[3]),
                        return false);
            OP_CHECK_IF(
                keyRopeShape[3] != DROPE_SIZE_64,
                OPS_REPORT_VECTOR_INNER_ERR(
                    opName,
                    "CrossShapeVerify failed, shape D of key_rope must be 64, but the value of key_rope[3] is (%ld).",
                    keyRopeShape[3]),
                return false);
        }
    }
    return true;
}

bool DenseLightningIndexerGradKLLossTilingBase::Analyze3DimLayout(const gert::Shape &queryShape,
                                                                  const gert::Shape &keyShape,
                                                                  const gert::Shape &queryIndexShape,
                                                                  const gert::Shape &keyIndexShape, size_t layoutLen,
                                                                  const gert::Shape &queryRopeShape,
                                                                  const gert::Shape &keyRopeShape)
{
    // dRopeSize的确定，有queryRopeShape 和 keyRopeShape
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKLen = 0;
            int64_t t1Size = queryShape.GetDim(0); // TND只有三个数值 T:0 N:1 D:2
            int64_t t2Size = keyShape.GetDim(0);
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKData.begin(), actualSeqLenKData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTHS_QUERY_INPUT_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTHS_KEY_INPUT_INDEX, actualSeqLenKData, actualSeqKLen);
            OP_CHECK_IF(actualSeqQLen != actualSeqKLen, OP_LOGE(opName, "VarLen scene, q is not equal k."),
                        return false);
            // 校验actualQ 对应每一个元素是否大于 actualK ，大于则拦截
            for (int i = 0; i < actualSeqQLen; i++) {
                OP_CHECK_IF(
                    actualSeqLenData[i] > actualSeqLenKData[i],
                    OP_LOGE(opName,
                            "Every element of actualSeqLenData must be less than every element of actualSeqLenKData,\
                        but actualSeqLenData[%d]:[%d] is larger than actualSeqLenKData[%d]:[%d].",
                            i, actualSeqLenData[i], i, actualSeqLenKData[i]),
                    return false);
            }
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.begin() + actualSeqQLen, 0LL);
            accumS2 = std::accumulate(actualSeqLenKData.begin(), actualSeqLenKData.begin() + actualSeqKLen, 0LL);
            realT1Size = accumS1;
            realT2Size = accumS2;
            padT1Size = t1Size;
            padT2Size = t2Size;

            OP_CHECK_IF(s1Size > s2Size || t1Size > t2Size || accumS1 > accumS2,
                        OP_LOGE(opName,
                                "Query s1Size(%ld), t1Size(%ld) and the sum of seqQLen(%ld) must be small than Key "
                                "s2Size(%ld), t2Size(%ld) and seqkLen(%ld), respectively.",
                                s1Size, t1Size, accumS1, s2Size, t2Size, accumS2),
                        return false);
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKData.begin(), actualSeqLenKData.end());
            OP_CHECK_IF(!CheckSeqLenLimit(maxS1Val, maxS2Val), OP_LOGE(opName, "TND sequence length check failed."),
                        return false);
            s1Size = maxS1Val;
            s2Size = maxS2Val;
            n2Size = keyShape.GetDim(1);
            n2IndexSize = keyIndexShape.GetDim(1);
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n2IndexSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2Index is zero."), return false);
            gSizeQuery = queryShape.GetDim(1) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(1) / n2IndexSize;
            dSizeQuery = queryShape.GetDim(2);
            dSizeQueryIndex = queryIndexShape.GetDim(2);
            if (hasRope) {
                dQueryRopeSize = queryRopeShape.GetDim(2);
                dKeyRopeSize = keyRopeShape.GetDim(2);
            }
            tilingData->baseParams.set_layoutType(LAYOUT_TND);
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        }
    } else if (layoutLen == 4UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            OP_CHECK_IF(!CheckSeqLenLimit(queryShape.GetDim(1), keyShape.GetDim(1)),
                        OP_LOGE(opName, "BSND sequence length check failed."), return false);
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2);
            n2IndexSize = keyIndexShape.GetDim(2);
            OP_CHECK_IF(s1Size > s2Size,
                        OP_LOGE(opName, "Query s1Size(%ld) must be small than Key s2Size(%ld).", s1Size, s2Size),
                        return false);
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n2IndexSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2Index is zero."), return false);
            gSizeQuery = queryShape.GetDim(2) / n2Size;
            gSizeQueryIndex = queryIndexShape.GetDim(2) / n2IndexSize;
            dSizeQuery = queryShape.GetDim(3);
            dSizeQueryIndex = queryIndexShape.GetDim(3);
            if (hasRope) {
                dQueryRopeSize = queryRopeShape.GetDim(3);
                dKeyRopeSize = keyRopeShape.GetDim(3);
            }
            tilingData->baseParams.set_layoutType(LAYOUT_BSND);
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        }
    } else {
        return false;
    }

    return true;
}

bool DenseLightningIndexerGradKLLossTilingBase::CheckSeqLenLimit(int64_t querySeqLength, int64_t keySeqLength) const
{
    OP_CHECK_IF(querySeqLength < 0 || querySeqLength > MAX_SEQ_LENGTH,
                OP_LOGE(opName, "Query sequence length[%ld] exceeds max supported sequence length[%ld].",
                        querySeqLength, MAX_SEQ_LENGTH),
                return false);
    OP_CHECK_IF(keySeqLength < 0 || keySeqLength > MAX_SEQ_LENGTH,
                OP_LOGE(opName, "Key sequence length[%ld] exceeds max supported sequence length[%ld].", keySeqLength,
                        MAX_SEQ_LENGTH),
                return false);
    return true;
}

void DenseLightningIndexerGradKLLossTilingBase::GetActualSeqLenData(int64_t inputIdx,
                                                                    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res,
                                                                    int64_t &actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return;
    }
    res[0] = value[0];
    actualLen++;
    for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
        auto qLen = value[i] - value[i - 1]; // value[i]代表偏移位置的索引
        res[i] = qLen < 0 ? 0 : qLen;
        actualLen++;
    }
}

int64_t DenseLightningIndexerGradKLLossTilingBase::GetS2RealSize(int32_t sparseMode, int32_t s1Size, int32_t s2Size,
                                                                 int32_t s1Idx)
{
    int64_t s2RealSize = 0;
    // sparsemode = 3 的情形
    if (sparseMode == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = static_cast<int64_t>((s2Size - s1Size) + s1Idx + 1);
        if (s2RealSize <= 0) {
            s2RealSize = static_cast<int64_t>(s2Size);
        }
    }
    return CeilDivision(s2RealSize, C0_SIZE) * C0_SIZE;
}

bool DenseLightningIndexerGradKLLossTilingBase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray)
{
    OP_CHECK_IF(sparseValidArray.size() == 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."), return false);

    uint32_t sparseMode = dliGradkllossBaseParams_->get_sparseMode();
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        int64_t totalSize = 0;
        int64_t accumS1 = 0;
        for (int32_t i = 0; i < bSize; i++) {
            int64_t s1Size = actualSeqLenData[i];
            int64_t s2Size = actualSeqLenKData[i];
            for (int64_t j = 0; j < s1Size; j++) {
                int64_t s2RealSize = GetS2RealSize(sparseMode, s1Size, s2Size, j);
                sparseValidArray[accumS1] = s2RealSize;
                accumS1++;
            }
        }
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
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

inline bool DenseLightningIndexerGradKLLossTilingBase::InitLoadValue(const std::vector<int64_t> &sparseValidArray,
                                                                     int64_t validAicNum, int64_t totalSize,
                                                                     const std::vector<int64_t> &sparseStartIdx,
                                                                     std::vector<int64_t> &localValue)
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

bool DenseLightningIndexerGradKLLossTilingBase::BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                                                            std::vector<int64_t> &localValue,
                                                            std::vector<int64_t> &sparseStartIdx)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAicNum =
        std::min(static_cast<int64_t>(dliGradkllossMultiCoreParams_->get_coreNum()), static_cast<int64_t>(aicNum));
    int64_t totalSize = dliGradkllossMultiCoreParams_->get_totalSize();
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

bool DenseLightningIndexerGradKLLossTilingBase::Balance4DLoad(std::vector<int64_t> &tmpSparseValue,
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
bool DenseLightningIndexerGradKLLossTilingBase::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray)
{
    int64_t validAicNum = static_cast<int64_t>(dliGradkllossMultiCoreParams_->get_coreNum());
    int64_t totalSize = dliGradkllossMultiCoreParams_->get_totalSize();
    int64_t *sparseStartIdx = dliGradkllossMultiCoreParams_->get_bS1Ptr();
    int64_t splitFactorSize = dliGradkllossMultiCoreParams_->get_splitFactorSize();

    OP_CHECK_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "totalSize should be larger than 0."),
                return false);
    OP_CHECK_IF(validAicNum <= 0 || validAicNum > static_cast<int64_t>(MAX_CORE_NUM),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "validAicNum[%ld] is invalid.", validAicNum), return false);
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        // initLoad: 使用均分策略, 保证后续不会比均分差
        std::vector<int64_t> localSparseStartIdx(validAicNum, totalSize);
        for (int64_t idx = 0; idx < validAicNum; ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAicNum, 0);
        // 得到了每个核需要计算的S2数量，在T方向平均分配，但是有于有Sparse的存在，localValue中的数值，每个核之间应该差距很大
        InitLoadValue(sparseValidArray, validAicNum, totalSize, localSparseStartIdx, localValue);

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAicNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(validAicNum, totalSize);
        int64_t sparseArraySum =
            std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有T上的S2的总数量
        int64_t avgVal = CeilDivision(sparseArraySum, validAicNum); // 平均每个核需要处理的S2的总数量

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < validAicNum; ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                             (tmpsparseStartIdx[idx] - 1) :
                                             (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                  singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAicNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAicNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, tmpLocalValue, tmpsparseStartIdx)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }
        for (int64_t idx = 0; idx < validAicNum; ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t sparseArraySum =
            std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL); // 得到所有BS上的S2的总数量
        int64_t balanceNum = CeilDivision(sparseArraySum, validAicNum);
        std::vector<int64_t> tmpSparseValue(validAicNum, 0);
        Balance4DLoad(tmpSparseValue, sparseValidArray, balanceNum);
        for (int64_t idx = 0; idx < static_cast<int64_t>(validAicNum); ++idx) {
            sparseStartIdx[idx] = tmpSparseValue[idx];
        }
    }

    for (int64_t idx = validAicNum; idx < static_cast<int64_t>(MAX_CORE_NUM); ++idx) {
        sparseStartIdx[idx] = totalSize;
    }
    return true;
}

// 计算S1的总个数
int64_t DenseLightningIndexerGradKLLossTilingBase::CalcTotalSize()
{
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return realT1Size;
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        return bSize * s1Size;
    }
    return 0; // 什么也不走就返回0
}

void DenseLightningIndexerGradKLLossTilingBase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min({totalSize, coreNum, static_cast<int64_t>(MAX_CORE_NUM)});
    dliGradkllossMultiCoreParams_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    dliGradkllossMultiCoreParams_->set_totalSize(totalSize);
    dliGradkllossMultiCoreParams_->set_padTotalSize(padT1Size);
    dliGradkllossMultiCoreParams_->set_splitFactorSize(
        actualUsedCoreNum > 0 ? CeilDivision(totalSize, actualUsedCoreNum) : 0);
}

bool DenseLightningIndexerGradKLLossTilingBase::SetSparseParamsRegbase()
{
    std::vector<int64_t> sparseValidArray(dliGradkllossMultiCoreParams_->get_totalSize(), 0);
    return InitSparseValidArray(sparseValidArray) && SetSparseStartIdx(sparseValidArray);
}

void DenseLightningIndexerGradKLLossTilingBase::InitOutputSplit()
{
    DLIGradKLLossInitOutputParams *initoutput = &tilingData->initOutputParams;
    auto &dKeyIndexShape = context_->GetOutputShape(1)->GetStorageShape();
    int64_t totalsize = 0;
    uint32_t singlecoresize = 0;
    int64_t usedAivNum = static_cast<int64_t>(dliGradkllossMultiCoreParams_->get_coreNum() * AIC_AIV_RATIO);
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(2); // TND场景下D为第二个维度
        int64_t totalT2Size = realT2Size;
        totalsize = totalT2Size * static_cast<int64_t>(dsizeDkeyindex);
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        int64_t dsizeDkeyindex = dKeyIndexShape.GetDim(3); // BSND场景下D为第三个维度
        int64_t totals2Size = dKeyIndexShape.GetDim(1);    // s为第二个维度
        totalsize = static_cast<int64_t>(bSize) * totals2Size * dsizeDkeyindex;
    }
    // 单个核均分元素数量
    singlecoresize = static_cast<uint32_t>(
        CeilDivision(totalsize, usedAivNum)); // 输出k-index总大小TD或者BSD 除以 总的aiv核数 向上取整
    initoutput->set_singleCoreSize(singlecoresize);
    initoutput->set_totalOutputSize(totalsize);

    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        // 只有TND场景存在需要pad T的场景
        int64_t t1PadSingleCoreSize = (padT1Size - realT1Size) / usedAivNum;
        int64_t t1PadRemainder = (padT1Size - realT1Size) % usedAivNum;
        int64_t t2PadSingleCoreSize = (padT2Size - realT2Size) / usedAivNum;
        int64_t t2PadRemainder = (padT2Size - realT2Size) % usedAivNum;

        initoutput->set_t1PadSingleCoreSize(t1PadSingleCoreSize);
        initoutput->set_t1PadRemainderSize(t1PadRemainder);
        initoutput->set_t2PadSingleCoreSize(t2PadSingleCoreSize);
        initoutput->set_t2PadRemainderSize(t2PadRemainder);
    }
}

int32_t DenseLightningIndexerGradKLLossTilingBase::GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ,
                                                                  int32_t actualSeqLensK, int32_t sparseMode)
{
    if (sparseMode == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
    } else {
        return 0;
    }
}

int64_t DenseLightningIndexerGradKLLossTilingBase::GetEndS1Etx(
    int32_t bIdx, int32_t defaultLens, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &actualSeqLenData, LayoutType layout)
{
    if (bSize <= 0 || bIdx >= bSize) {
        return defaultLens;
    }

    if (layout == LayoutType::LAYOUT_TND) {
        return actualSeqLenData[bIdx];
    } else {
        return 0;
    }
}

int32_t DenseLightningIndexerGradKLLossTilingBase::GetActualSeqLens(
    int32_t bIdx, int32_t defaultLens, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &actualSeqLenData, LayoutType layout)
{
    if (bSize <= 0 || bIdx >= bSize) {
        return defaultLens;
    }

    if (layout == LayoutType::LAYOUT_TND) {
        return actualSeqLenData[bIdx];
    } else {
        return defaultLens;
    }
}

int64_t DenseLightningIndexerGradKLLossTilingBase::FindBIndex(int64_t bIndex, int64_t curBs1Index,
                                                              int64_t &accumulateLen)
{
    for (int index = bIndex; index < bSize; index++) {
        int64_t actualLen = actualSeqLenData[index];
        if (curBs1Index < actualLen) {
            return index;
        }
        accumulateLen = actualLen;
    }
    return dliGradkllossMultiCoreParams_->get_totalSize() >= curBs1Index ? bSize : -1;
}

void DenseLightningIndexerGradKLLossTilingBase::CalcMultiCoreOffset(int64_t &bStartIdx, int64_t &s1StartIdx,
                                                                    int64_t &bEndIdx, int64_t &s1EndIdx,
                                                                    int64_t &aicIdx)
{
    int64_t actualSum = 0;

    int64_t *sparseStartIdx = dliGradkllossMultiCoreParams_->get_bS1Ptr();
    if (sparseStartIdx == nullptr || aicIdx >= static_cast<int64_t>(dliGradkllossMultiCoreParams_->get_coreNum())) {
        return;
    }
    int64_t bS1Index = sparseStartIdx[aicIdx];
    int64_t bS1EndIndex = aicIdx + 1 < dliGradkllossMultiCoreParams_->get_coreNum() ?
                              sparseStartIdx[aicIdx + 1] :
                              dliGradkllossMultiCoreParams_->get_totalSize();
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        bStartIdx = FindBIndex(0, bS1Index, actualSum);
        s1StartIdx = bS1Index - actualSum;
        bEndIdx = FindBIndex(bStartIdx, bS1EndIndex - 1, actualSum);
        s1EndIdx = bS1EndIndex - actualSum;
    } else {
        bStartIdx = bS1Index / s1Size;
        bEndIdx = (bS1EndIndex - 1) / s1Size;
        s1StartIdx = bS1Index - bStartIdx * s1Size;
        s1EndIdx = bS1EndIndex - bEndIdx * s1Size;
    }
}

void DenseLightningIndexerGradKLLossTilingBase::CalcMaxLoop()
{
    if (!deterministic) {
        return;
    }
    int64_t maxLoopSize = 0;
    for (int64_t aicIdx = 0; aicIdx < dliGradkllossMultiCoreParams_->get_coreNum(); aicIdx++) {
        int64_t bStartIdx;
        int64_t bEndIdx;
        int64_t s1StartIdx;
        int64_t s1EndIdx;
        CalcMultiCoreOffset(bStartIdx, s1StartIdx, bEndIdx, s1EndIdx, aicIdx);
        int64_t taskId = 0;
        uint32_t s2PreloadTail = 0;

        for (int64_t bIdx = bStartIdx; bIdx <= bEndIdx; bIdx++) {
            bool lastB = (bIdx == bEndIdx);
            int64_t s1StartIdxThisBatch = 0;
            int64_t s1EndIdxThisBatch = 0;
            if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
                s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
                s1EndIdxThisBatch = (!lastB) ? GetEndS1Etx(bIdx, s1Size, actualSeqLenData, tilingKeyLayout) : s1EndIdx;
            } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
                s1StartIdxThisBatch = (bIdx == bStartIdx) ? s1StartIdx : 0;
                s1EndIdxThisBatch = (!lastB) ? s1Size : s1EndIdx;
            }

            for (int64_t s1Idx = s1StartIdxThisBatch; s1Idx < s1EndIdxThisBatch; s1Idx += S1_BASE_STEP) {
                bool lastS1 = ((s1Idx + S1_BASE_STEP) >= s1EndIdxThisBatch);
                uint32_t actualSeqLensQ = GetActualSeqLens(bIdx, s1Size, actualSeqLenData, tilingKeyLayout);
                uint32_t actualSeqLensK = GetActualSeqLens(bIdx, s2Size, actualSeqLenKData, tilingKeyLayout);
                int64_t s1EndIdxThisLoop = lastS1 ? s1EndIdxThisBatch : (s1Idx + S1_BASE_STEP);
                int32_t s2EndIdx = GetS2SparseLen(s1EndIdxThisLoop, actualSeqLensQ, actualSeqLensK, sparseMode) - 1;
                if (lastB && lastS1) {
                    s2PreloadTail = S2_BASE_STEP * 1;
                }
                taskId += CeilDivision(s2EndIdx + s2PreloadTail, S2_BASE_STEP);
            }
        }
        maxLoopSize = Max(maxLoopSize, taskId);
    }
    dliGradkllossBaseParams_->set_maxLoopSize(maxLoopSize);
    OP_LOGD(context_->GetNodeName(), "CalcMaxLoop maxLoopSize = %d.", maxLoopSize);
}

ge::graphStatus DenseLightningIndexerGradKLLossTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());

    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
                OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
    // 将基本输入参数传给tilingdata
    dliGradkllossBaseParams_->set_bSize(bSize);
    dliGradkllossBaseParams_->set_n2Size(n2Size);
    dliGradkllossBaseParams_->set_n2IndexSize(n2IndexSize);
    dliGradkllossBaseParams_->set_gSizeQuery(gSizeQuery);
    dliGradkllossBaseParams_->set_gSizeQueryIndex(gSizeQueryIndex);
    dliGradkllossBaseParams_->set_s1Size(s1Size);
    dliGradkllossBaseParams_->set_s2Size(s2Size);
    dliGradkllossBaseParams_->set_dSizeQuery(dSizeQuery);
    dliGradkllossBaseParams_->set_dSizeQueryIndex(dSizeQueryIndex);
    dliGradkllossBaseParams_->set_kSize(2048); // TODO: dense这里应该要改
    dliGradkllossBaseParams_->set_sparseMode(sparseMode);
    dliGradkllossBaseParams_->set_scaleValue(scaleValue);

    OP_LOGD(opName,
            "INPUTPARAM bsize:[%d], n2Size:[%d], gSizeQuery:[%d], dSizeQueryIndex:[%d], s1Size:[%d], s2Size:[%d], "
            "dSizeQuery:[%d], dSizeQueryIndex:[%d], kSize:[%d], sparseMode:[%d], scaleValue:[%f] .",
            bSize, n2Size, gSizeQuery, gSizeQueryIndex, s1Size, s2Size, dSizeQuery, dSizeQueryIndex, kSize, sparseMode,
            scaleValue);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerGradKLLossTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const DenseLightningIndexerGradKLLossCompileInfo *>(context_->GetCompileInfo());
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
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(opName, "invalid core count, aicNum[%u], aivNum[%u].", aicNum, aivNum),
                return ge::GRAPH_FAILED);
    OP_LOGD(opName, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).", aivNum,
            aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerGradKLLossTilingBase::DoOpTiling()
{
    OP_LOGD(opName, "try template[%s]", templateName);
    // 无多余操作，分核，目前只实现TND场景分核
    int64_t totalSize = CalcTotalSize();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    OP_CHECK_IF(dliGradkllossMultiCoreParams_->get_coreNum() == 0, OP_LOGE(opName, "no valid AIC task is available."),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(dliGradkllossMultiCoreParams_->get_coreNum()); // 使用的核数确定
    context_->SetScheduleMode(SCHEDULE_MODE_ALL_CORE);

    OP_CHECK_IF(!SetSparseParamsRegbase(), OP_LOGE(opName, "failed to set sparse multi-core parameters."),
                return ge::GRAPH_FAILED);
    InitOutputSplit(); // output分核
    // 确定性场景需要计算最大轮次
    CalcMaxLoop();
    OP_LOGD(opName, "ending template[%s]", templateName);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DenseLightningIndexerGradKLLossTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    uint64_t bmm1Size = static_cast<uint64_t>(n2Size) * S1_BASE_STEP * S2_BASE_STEP * sizeof(float);
    uint64_t bmm2Size = static_cast<uint64_t>(gSizeQueryIndex) * S1_BASE_STEP * S2_BASE_STEP * sizeof(float);
    uint64_t reluGradSize = static_cast<uint64_t>(gSizeQueryIndex) * S1_BASE_STEP * S2_BASE_STEP * sizeof(float);
    uint64_t psySyncSize = static_cast<uint64_t>(AIC_AIV_RATIO) * S1_VEC_SIZE_8 * S2_BASE_STEP * 2 * sizeof(float);

    uint64_t dWeightFloatSize = static_cast<uint64_t>(S1_BASE_STEP) * n2IndexSize * gSizeQueryIndex * sizeof(float);
    auto weightsDtype = context_->GetInputDesc(WEIGHT_INPUT_INDEX)->GetDataType();
    if (weightsDtype == ge::DT_FLOAT) {
        dWeightFloatSize = 0;
    }

    uint64_t dQueryIndexFloatSize =
        static_cast<uint64_t>(S1_BASE_STEP) * gSizeQueryIndex * dSizeQueryIndex * sizeof(float);
    uint64_t dKeyIndexGmSize = 0;
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        dKeyIndexGmSize = static_cast<uint64_t>(accumS2) * n2IndexSize * dSizeQueryIndex * sizeof(float);
    } else {
        dKeyIndexGmSize = static_cast<uint64_t>(bSize) * s2Size * n2IndexSize * dSizeQueryIndex * sizeof(float);
    }

    uint64_t dKeyIndexDeterGmSize = 0;
    uint64_t lossDeterGmSize = 0;
    uint64_t coreRuninfoDeterGmSize = 0;
    if (deterministic) {
        uint64_t coreNum = dliGradkllossMultiCoreParams_->get_coreNum();
        dKeyIndexDeterGmSize = coreNum * S2_BASE_STEP * n2IndexSize * dSizeQueryIndex * sizeof(float);
        lossDeterGmSize = coreNum * AIC_AIV_RATIO * optiling::DETER_LOSS_TMP_GM_NUM * sizeof(float);
        coreRuninfoDeterGmSize = coreNum * DETER_CORE_INFO_TMP_GM_NUM * sizeof(int64_t);
    }

    uint64_t singlecoreTotalSize =
        PING_PONG_VALUE * (bmm1Size + bmm2Size + reluGradSize + psySyncSize) + dWeightFloatSize + dQueryIndexFloatSize;
    uint64_t multicoreTotalSize = singlecoreTotalSize * dliGradkllossMultiCoreParams_->get_coreNum() + dKeyIndexGmSize +
                                  dKeyIndexDeterGmSize + lossDeterGmSize + coreRuninfoDeterGmSize;
    OP_CHECK_IF(multicoreTotalSize > std::numeric_limits<size_t>::max() - WORK_SPACE_RESERVE_SIZE,
                OP_LOGE(opName, "workspace size overflow."), return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(multicoreTotalSize + WORK_SPACE_RESERVE_SIZE);
    OP_LOGW(context_, "workspace size:[%zu], multicoreTotalSize:[%lu]", workspaces[0], multicoreTotalSize);
    return ge::GRAPH_SUCCESS;
}

uint64_t DenseLightningIndexerGradKLLossTilingBase::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint8_t>(hasRope), static_cast<uint8_t>(tilingKeyLayout),
                              static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(sparseMode),
                              static_cast<uint8_t>(deterministic));
}

} // namespace optiling
