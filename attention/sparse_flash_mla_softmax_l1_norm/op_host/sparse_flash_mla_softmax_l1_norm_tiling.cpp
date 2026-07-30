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
 * \file sparse_flash_mla_softmax_l1_norm_tiling.cpp
 * \brief 补充完整的参数校验，支持 BSND 和 TND 布局
 */
#include <string>
#include <set>
#include "sparse_flash_mla_softmax_l1_norm_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;
using namespace std;

namespace optiling {
namespace smlasoftmaxl1norm {

constexpr int64_t QUERY_IDX = 0;
constexpr int64_t KEY_IDX = 1;
constexpr int64_t LSE_IDX = 2;
constexpr int64_t SPARSE_INDICES_IDX = 3;
constexpr int64_t CU_SEQLENS_Q_IDX = 4;
constexpr int64_t CU_SEQLENS_K_IDX = 5;
constexpr int64_t SEQUSED_Q_IDX = 6;
constexpr int64_t SEQUSED_K_IDX = 7;
constexpr int64_t CMP_RESIDUAL_K_IDX = 8;
constexpr int64_t TOPK_LENGTH_IDX = 9;
constexpr int64_t METADATA_IDX = 10;
constexpr int64_t SOFTMAX_L1_NORM_IDX = 0;

constexpr int64_t DIM_LIMIT = 512;
constexpr int64_t N1_MIN = 1;
constexpr int64_t N1_MAX = 128;
constexpr int64_t N2_VALUE = 1;
constexpr int64_t CMP_RATIO_MIN = 1;
constexpr int64_t CMP_RATIO_MAX = 128;
constexpr int64_t DIM_NUM_THREE = 3;
constexpr int64_t DIM_NUM_FOUR = 4;
constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t DIM_NUM_TWO = 2;
constexpr int64_t B_ADD_ONE = 1;
constexpr int64_t SCALE_VALUE_ATTR_IDX = 0;
constexpr int64_t MAX_SEQLEN_K_ATTR_IDX = 1;
constexpr int64_t CMP_RATIO_ATTR_IDX = 2;
constexpr int64_t MASK_MODE_ATTR_IDX = 3;
constexpr int64_t LAYOUT_Q_ATTR_IDX = 4;
constexpr int64_t LAYOUT_K_ATTR_IDX = 5;

struct TilingShapeInfo {
    int64_t bSize = 0;
    int64_t sqSize = 0;
    int64_t skSize = 0;
    int64_t gSize = 0;
    int64_t dSize = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
};

static string DataTypeToStr(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BFLOAT16";
        case ge::DT_FLOAT:
            return "FLOAT32";
        case ge::DT_INT32:
            return "INT32";
        case ge::DT_INT64:
            return "INT64";
        default:
            return "UNKNOWN(" + to_string(static_cast<int>(dtype)) + ")";
    }
}

static int64_t AlignData(const int64_t a, const int64_t b)
{
    if (b == 0U) {
        return a;
    }
    return (a + b - 1U) / b * b;
}

static ge::graphStatus ParseAttrs(gert::TilingContext *context, float &scaleValue, int64_t &maxSeqlenK,
                                  int64_t &cmpRatio, int64_t &maskMode, string &layoutQ, string &layoutK)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const float *scaleValuePtr = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_IDX);
    const int64_t *maxSeqlenKPtr = attrs->GetAttrPointer<int64_t>(MAX_SEQLEN_K_ATTR_IDX);
    const int64_t *cmpRatioPtr = attrs->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_IDX);
    const int64_t *maskModePtr = attrs->GetAttrPointer<int64_t>(MASK_MODE_ATTR_IDX);
    const char *layoutQPtr = attrs->GetAttrPointer<char>(LAYOUT_Q_ATTR_IDX);
    const char *layoutKPtr = attrs->GetAttrPointer<char>(LAYOUT_K_ATTR_IDX);

    scaleValue = (scaleValuePtr != nullptr) ? *scaleValuePtr : 1.0;
    maxSeqlenK = (maxSeqlenKPtr != nullptr) ? *maxSeqlenKPtr : 0;
    cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : 1;
    maskMode = (maskModePtr != nullptr) ? *maskModePtr : 0;
    layoutQ = (layoutQPtr != nullptr) ? string(layoutQPtr) : "BSND";
    layoutK = (layoutKPtr != nullptr) ? string(layoutKPtr) : "BSND";
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateRequiredInputs(gert::TilingContext *context, const string &opName)
{
    auto queryDesc = context->GetInputDesc(QUERY_IDX);
    auto keyDesc = context->GetInputDesc(KEY_IDX);
    auto lseDesc = context->GetInputDesc(LSE_IDX);
    auto outputDesc = context->GetOutputDesc(SOFTMAX_L1_NORM_IDX);

    OP_CHECK_IF(queryDesc == nullptr, OP_LOGE(opName, "query must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDesc == nullptr, OP_LOGE(opName, "key must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(lseDesc == nullptr, OP_LOGE(opName, "lse must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc == nullptr, OP_LOGE(opName, "softmaxL1Norm must be provided."), return ge::GRAPH_FAILED);

    auto metadataTensor = context->GetOptionalInputTensor(METADATA_IDX);
    auto metadataDesc = context->GetOptionalInputDesc(METADATA_IDX);
    OP_CHECK_IF(metadataTensor == nullptr, OP_LOGE(opName, "metadata must be provided."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataDesc == nullptr, OP_LOGE(opName, "metadata desc must be provided."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateLayout(const string &layoutQ, const string &layoutK, bool &isTnd, const string &opName)
{
    const set<string> validLayouts = {"BSND", "TND"};
    OP_CHECK_IF(validLayouts.find(layoutQ) == validLayouts.end(),
                OP_LOGE(opName, "layoutQ only supports BSND/TND, but got %s.", layoutQ.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validLayouts.find(layoutK) == validLayouts.end(),
                OP_LOGE(opName, "layoutK only supports BSND/TND, but got %s.", layoutK.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(layoutQ != layoutK,
                OP_LOGE(opName, "layoutQ(%s) must be the same as layoutK(%s).", layoutQ.c_str(), layoutK.c_str()),
                return ge::GRAPH_FAILED);
    isTnd = (layoutQ == "TND");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateDtypes(gert::TilingContext *context, const string &opName)
{
    auto queryDesc = context->GetInputDesc(QUERY_IDX);
    auto keyDesc = context->GetInputDesc(KEY_IDX);
    auto lseDesc = context->GetInputDesc(LSE_IDX);
    auto outputDesc = context->GetOutputDesc(SOFTMAX_L1_NORM_IDX);
    auto metadataDesc = context->GetOptionalInputDesc(METADATA_IDX);

    auto queryDtype = queryDesc->GetDataType();
    auto keyDtype = keyDesc->GetDataType();
    auto lseDtype = lseDesc->GetDataType();
    auto outputDtype = outputDesc->GetDataType();

    const set<ge::DataType> validQKDtypes = {ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(
        validQKDtypes.find(queryDtype) == validQKDtypes.end(),
        OP_LOGE(opName, "query dtype only supports FLOAT16/BFLOAT16, but got %s.", DataTypeToStr(queryDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        validQKDtypes.find(keyDtype) == validQKDtypes.end(),
        OP_LOGE(opName, "key dtype only supports FLOAT16/BFLOAT16, but got %s.", DataTypeToStr(keyDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(lseDtype != ge::DT_FLOAT,
                OP_LOGE(opName, "lse dtype must be FLOAT32, but got %s.", DataTypeToStr(lseDtype).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDtype != ge::DT_FLOAT,
                OP_LOGE(opName, "softmaxL1Norm dtype must be FLOAT32, but got %s.", DataTypeToStr(outputDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto checkInt32OptionalInput = [&](uint32_t idx, const string &name) -> ge::graphStatus {
        auto desc = context->GetOptionalInputDesc(idx);
        if (desc != nullptr) {
            OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(opName, "%s dtype must be INT32, but got %s.", name.c_str(),
                                DataTypeToStr(desc->GetDataType()).c_str()),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    };

    if (checkInt32OptionalInput(SPARSE_INDICES_IDX, "sparseIndices") != ge::GRAPH_SUCCESS ||
        checkInt32OptionalInput(CU_SEQLENS_Q_IDX, "cuSeqlensQ") != ge::GRAPH_SUCCESS ||
        checkInt32OptionalInput(CU_SEQLENS_K_IDX, "cuSeqlensK") != ge::GRAPH_SUCCESS ||
        checkInt32OptionalInput(SEQUSED_Q_IDX, "sequsedQ") != ge::GRAPH_SUCCESS ||
        checkInt32OptionalInput(SEQUSED_K_IDX, "sequsedK") != ge::GRAPH_SUCCESS ||
        checkInt32OptionalInput(CMP_RESIDUAL_K_IDX, "cmpResidualK") != ge::GRAPH_SUCCESS ||
        checkInt32OptionalInput(TOPK_LENGTH_IDX, "topkLength") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(opName, "metadata dtype must be INT32, but got %s.",
                        DataTypeToStr(metadataDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputDims(gert::TilingContext *context, const string &layoutQ, const string &layoutK,
                                         bool isTnd, const string &opName)
{
    auto queryShape = context->GetInputShape(QUERY_IDX);
    auto keyShape = context->GetInputShape(KEY_IDX);
    auto lseShape = context->GetInputShape(LSE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, lseShape);

    size_t expectedDim = isTnd ? DIM_NUM_THREE : DIM_NUM_FOUR;
    OP_CHECK_IF(queryShape->GetStorageShape().GetDimNum() != expectedDim,
                OP_LOGE(opName, "query dim num must be %zu when layout is %s, but got %zu.", expectedDim,
                        layoutQ.c_str(), queryShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape->GetStorageShape().GetDimNum() != expectedDim,
                OP_LOGE(opName, "key dim num must be %zu when layout is %s, but got %zu.", expectedDim, layoutK.c_str(),
                        keyShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != expectedDim,
                OP_LOGE(opName, "lse dim num must be %zu when layout is %s, but got %zu.", expectedDim, layoutQ.c_str(),
                        lseShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);

    auto metadataTensor = context->GetOptionalInputTensor(METADATA_IDX);
    OP_CHECK_IF(
        metadataTensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
        OP_LOGE(opName, "metadata dim num must be 1, but got %zu.", metadataTensor->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAttrRanges(int64_t cmpRatio, int64_t maskMode, const string &opName)
{
    OP_CHECK_IF(cmpRatio < CMP_RATIO_MIN || cmpRatio > CMP_RATIO_MAX,
                OP_LOGE(opName, "cmpRatio must be in [1, 128], but got %lld.", cmpRatio), return ge::GRAPH_FAILED);
    OP_CHECK_IF(maskMode != 0 && maskMode != 3,
                OP_LOGE(opName, "maskMode only supports 0 or 3, but got %lld.", maskMode), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAndExtractShapesTnd(gert::TilingContext *context, TilingShapeInfo &info,
                                                   const string &opName)
{
    auto queryShape = context->GetInputShape(QUERY_IDX);
    auto keyShape = context->GetInputShape(KEY_IDX);

    info.t1 = queryShape->GetStorageShape().GetDim(0);
    info.gSize = queryShape->GetStorageShape().GetDim(1);
    info.dSize = queryShape->GetStorageShape().GetDim(2);
    info.t2 = keyShape->GetStorageShape().GetDim(0);

    int64_t keyN2Size = keyShape->GetStorageShape().GetDim(1);
    int64_t keyDSize = keyShape->GetStorageShape().GetDim(2);
    OP_CHECK_IF(keyN2Size != N2_VALUE, OP_LOGE(opName, "key N2 must be 1, but got %lld.", keyN2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDSize != DIM_LIMIT, OP_LOGE(opName, "key D must be 512, but got %lld.", keyDSize),
                return ge::GRAPH_FAILED);

    auto cuSeqQInput = context->GetOptionalInputTensor(CU_SEQLENS_Q_IDX);
    auto cuSeqKInput = context->GetOptionalInputTensor(CU_SEQLENS_K_IDX);
    OP_CHECK_IF(cuSeqQInput == nullptr, OP_LOGE(opName, "cuSeqlensQ must be provided when layout is TND."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuSeqKInput == nullptr, OP_LOGE(opName, "cuSeqlensK must be provided when layout is TND."),
                return ge::GRAPH_FAILED);

    auto cuSeqQShape = context->GetInputShape(CU_SEQLENS_Q_IDX);
    auto cuSeqKShape = context->GetInputShape(CU_SEQLENS_K_IDX);
    OP_CHECK_IF(
        cuSeqQShape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
        OP_LOGE(opName, "cuSeqlensQ dim num must be 1, but got %zu.", cuSeqQShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        cuSeqKShape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
        OP_LOGE(opName, "cuSeqlensK dim num must be 1, but got %zu.", cuSeqKShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(cuSeqQShape->GetStorageShape().GetDim(0) < B_ADD_ONE,
                OP_LOGE(opName, "cuSeqlensQ dim[0](%lld) must be >= 1.", cuSeqQShape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    info.bSize = cuSeqQShape->GetStorageShape().GetDim(0) - B_ADD_ONE;

    OP_CHECK_IF(cuSeqKShape->GetStorageShape().GetDim(0) < B_ADD_ONE,
                OP_LOGE(opName, "cuSeqlensK dim[0](%lld) must be >= 1.", cuSeqKShape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    int64_t cuSeqKBSize = cuSeqKShape->GetStorageShape().GetDim(0) - B_ADD_ONE;
    OP_CHECK_IF(info.bSize != cuSeqKBSize,
                OP_LOGE(opName, "cuSeqlensQ B(%lld) must be equal to cuSeqlensK B(%lld).", info.bSize, cuSeqKBSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateAndExtractShapesBsnd(gert::TilingContext *context, TilingShapeInfo &info,
                                                    const string &opName)
{
    auto queryShape = context->GetInputShape(QUERY_IDX);
    auto keyShape = context->GetInputShape(KEY_IDX);

    info.bSize = queryShape->GetStorageShape().GetDim(0);
    info.sqSize = queryShape->GetStorageShape().GetDim(1);
    info.gSize = queryShape->GetStorageShape().GetDim(2);
    info.dSize = queryShape->GetStorageShape().GetDim(3);
    info.skSize = keyShape->GetStorageShape().GetDim(1);

    int64_t keyBSize = keyShape->GetStorageShape().GetDim(0);
    int64_t keyN2Size = keyShape->GetStorageShape().GetDim(2);
    int64_t keyDSize = keyShape->GetStorageShape().GetDim(3);

    OP_CHECK_IF(info.bSize != keyBSize,
                OP_LOGE(opName, "query B(%lld) must be equal to key B(%lld).", info.bSize, keyBSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyN2Size != N2_VALUE, OP_LOGE(opName, "key N2 must be 1, but got %lld.", keyN2Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDSize != DIM_LIMIT, OP_LOGE(opName, "key D must be 512, but got %lld.", keyDSize),
                return ge::GRAPH_FAILED);

    auto cuSeqQInput = context->GetOptionalInputTensor(CU_SEQLENS_Q_IDX);
    auto cuSeqKInput = context->GetOptionalInputTensor(CU_SEQLENS_K_IDX);
    OP_CHECK_IF(cuSeqQInput != nullptr, OP_LOGE(opName, "cuSeqlensQ should be null when layout is BSND."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuSeqKInput != nullptr, OP_LOGE(opName, "cuSeqlensK should be null when layout is BSND."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShapeRanges(gert::TilingContext *context, const TilingShapeInfo &info, bool isTnd,
                                           const string &opName)
{
    auto keyShape = context->GetInputShape(KEY_IDX);
    int64_t n1Size = info.gSize;
    int64_t n2SizeFromKey = isTnd ? keyShape->GetStorageShape().GetDim(1) : keyShape->GetStorageShape().GetDim(2);

    OP_CHECK_IF(n1Size < N1_MIN || n1Size > N1_MAX, OP_LOGE(opName, "N1 must be in [1, 128], but got %lld.", n1Size),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(n2SizeFromKey != N2_VALUE, OP_LOGE(opName, "N2 must be 1, but got %lld.", n2SizeFromKey),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.dSize != DIM_LIMIT, OP_LOGE(opName, "D must be 512, but got %lld.", info.dSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateLseShape(gert::TilingContext *context, const TilingShapeInfo &info, bool isTnd,
                                        const string &opName)
{
    auto lseShape = context->GetInputShape(LSE_IDX);
    if (isTnd) {
        int64_t lseN2 = lseShape->GetStorageShape().GetDim(0);
        int64_t lseT1 = lseShape->GetStorageShape().GetDim(1);
        int64_t lseG = lseShape->GetStorageShape().GetDim(2);
        OP_CHECK_IF(lseN2 != N2_VALUE, OP_LOGE(opName, "lse N2 must be 1, but got %lld.", lseN2),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseT1 != info.t1, OP_LOGE(opName, "lse T1(%lld) must be equal to query T1(%lld).", lseT1, info.t1),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseG != info.gSize,
                    OP_LOGE(opName, "lse G(%lld) must be equal to query G(%lld).", lseG, info.gSize),
                    return ge::GRAPH_FAILED);
    } else {
        int64_t lseB = lseShape->GetStorageShape().GetDim(0);
        int64_t lseN2 = lseShape->GetStorageShape().GetDim(1);
        int64_t lseS1 = lseShape->GetStorageShape().GetDim(2);
        int64_t lseG = lseShape->GetStorageShape().GetDim(3);
        OP_CHECK_IF(lseB != info.bSize,
                    OP_LOGE(opName, "lse B(%lld) must be equal to query B(%lld).", lseB, info.bSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseN2 != N2_VALUE, OP_LOGE(opName, "lse N2 must be 1, but got %lld.", lseN2),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseS1 != info.sqSize,
                    OP_LOGE(opName, "lse S1(%lld) must be equal to query S1(%lld).", lseS1, info.sqSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseG != info.gSize,
                    OP_LOGE(opName, "lse G(%lld) must be equal to query G(%lld).", lseG, info.gSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateOptionalShapeInputs(gert::TilingContext *context, int64_t bSize, int64_t maskMode,
                                                   int64_t cmpRatio, const string &opName)
{
    auto cmpResidualKInput = context->GetOptionalInputTensor(CMP_RESIDUAL_K_IDX);
    if (maskMode == 3 && cmpRatio > 1) {
        OP_CHECK_IF(cmpResidualKInput == nullptr, OP_LOGE(opName, "cmpResidualK must be provided when maskMode is 3."),
                    return ge::GRAPH_FAILED);
    }
    if (cmpResidualKInput != nullptr) {
        OP_CHECK_IF(cmpResidualKInput->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                    OP_LOGE(opName, "cmpResidualK dim num must be 1, but got %zu.",
                            cmpResidualKInput->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cmpResidualKInput->GetStorageShape().GetDim(0) != bSize,
                    OP_LOGE(opName, "cmpResidualK shape[0](%lld) must be equal to B(%lld).",
                            cmpResidualKInput->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_FAILED);
    }

    auto sequsedQInput = context->GetOptionalInputTensor(SEQUSED_Q_IDX);
    if (sequsedQInput != nullptr) {
        OP_CHECK_IF(
            sequsedQInput->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
            OP_LOGE(opName, "sequsedQ dim num must be 1, but got %zu.", sequsedQInput->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(sequsedQInput->GetStorageShape().GetDim(0) != bSize,
                    OP_LOGE(opName, "sequsedQ shape[0](%lld) must be equal to B(%lld).",
                            sequsedQInput->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_FAILED);
    }

    auto sequsedKInput = context->GetOptionalInputTensor(SEQUSED_K_IDX);
    if (sequsedKInput != nullptr) {
        OP_CHECK_IF(
            sequsedKInput->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
            OP_LOGE(opName, "sequsedK dim num must be 1, but got %zu.", sequsedKInput->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(sequsedKInput->GetStorageShape().GetDim(0) != bSize,
                    OP_LOGE(opName, "sequsedK shape[0](%lld) must be equal to B(%lld).",
                            sequsedKInput->GetStorageShape().GetDim(0), bSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateSparseInfo(gert::TilingContext *context, int64_t maskMode, bool isTnd,
                                          const string &layoutQ, const string &opName, bool &isSparse, int64_t &kLength)
{
    auto sparseIndicesInput = context->GetOptionalInputTensor(SPARSE_INDICES_IDX);
    isSparse = (sparseIndicesInput != nullptr);
    kLength = 0;

    if (isSparse) {
        auto sparseShape = context->GetInputShape(SPARSE_INDICES_IDX);
        size_t sparseDimNum = sparseShape->GetStorageShape().GetDimNum();
        size_t expectedSparseDim = isTnd ? DIM_NUM_THREE : DIM_NUM_FOUR;
        OP_CHECK_IF(sparseDimNum != expectedSparseDim,
                    OP_LOGE(opName, "sparseIndices dim num must be %zu when layout is %s, but got %zu.",
                            expectedSparseDim, layoutQ.c_str(), sparseDimNum),
                    return ge::GRAPH_FAILED);
        kLength = sparseShape->GetStorageShape().GetDim(sparseDimNum - 1);

        auto topkLengthInput = context->GetOptionalInputTensor(TOPK_LENGTH_IDX);
        if (maskMode == 0) {
            OP_CHECK_IF(topkLengthInput == nullptr,
                        OP_LOGE(opName, "topkLength must be provided when maskMode is 0 and sparseIndices exists."),
                        return ge::GRAPH_FAILED);
            size_t expectedTopkDim = isTnd ? DIM_NUM_TWO : DIM_NUM_THREE;
            OP_CHECK_IF(topkLengthInput->GetStorageShape().GetDimNum() != expectedTopkDim,
                        OP_LOGE(opName, "topkLength dim num must be %zu when layout is %s, but got %zu.",
                                expectedTopkDim, layoutQ.c_str(), topkLengthInput->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, const string &opName, AiCoreParams &aicoreParams_)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    uint64_t l2CacheSize;
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const SparseFlashMlaSoftmaxL1NormCompileInfo *>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compile_info is null."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->aivNum;
        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
        aicoreParams_.l0bSize = compileInfoPtr->l0bSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_CHECK_IF((aicoreParams_.numBlocks == 0) || (aicoreParams_.aicNum == 0),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                                            aicoreParams_.numBlocks, aicoreParams_.aicNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "ubSize or l2CacheSize is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SparseFlashMlaSoftmaxL1NormTilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    float scaleValue;
    int64_t maxSeqlenK;
    int64_t cmpRatio;
    int64_t maskMode;
    string layoutQ;
    string layoutK;
    if (ParseAttrs(context, scaleValue, maxSeqlenK, cmpRatio, maskMode, layoutQ, layoutK)) {
        return ge::GRAPH_FAILED;
    }
    auto opName = context->GetNodeName();
    if (ValidateRequiredInputs(context, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool isTnd = false;
    if (ValidateLayout(layoutQ, layoutK, isTnd, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ValidateDtypes(context, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateInputDims(context, layoutQ, layoutK, isTnd, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateAttrRanges(cmpRatio, maskMode, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    TilingShapeInfo info;
    if (isTnd) {
        if (ValidateAndExtractShapesTnd(context, info, opName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (ValidateAndExtractShapesBsnd(context, info, opName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (ValidateShapeRanges(context, info, isTnd, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateLseShape(context, info, isTnd, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateOptionalShapeInputs(context, info.bSize, maskMode, cmpRatio, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool isSparse = false;
    int64_t kLength = 0;
    if (ValidateSparseInfo(context, maskMode, isTnd, layoutQ, opName, isSparse, kLength) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool has_seqused_q = (context->GetOptionalInputTensor(SEQUSED_Q_IDX) != nullptr);
    bool has_seqused_k = (context->GetOptionalInputTensor(SEQUSED_K_IDX) != nullptr);
    bool has_topk_length = (context->GetOptionalInputTensor(TOPK_LENGTH_IDX) != nullptr);

    OpTiling *tilingData = context->GetTilingData<OpTiling>();
    tilingData->b = info.bSize;
    tilingData->sq = info.sqSize;
    tilingData->sk = info.skSize;
    tilingData->g = info.gSize;
    tilingData->d = info.dSize;
    tilingData->t1 = info.t1;
    tilingData->t2 = info.t2;
    tilingData->max_seqlen_k = maxSeqlenK;
    tilingData->k_length = kLength;
    tilingData->cmp_ratio = cmpRatio;
    tilingData->softmax_scale = scaleValue;
    tilingData->has_seqused_q = has_seqused_q;
    tilingData->has_seqused_k = has_seqused_k;
    tilingData->has_topk_length = has_topk_length;

    if (isTnd) {
        tilingData->init_total_num = info.t1 * (isSparse ? kLength : maxSeqlenK);
    } else {
        tilingData->init_total_num = info.bSize * info.sqSize * (isSparse ? kLength : info.skSize);
    }
    AiCoreParams aicoreParams_;
    if (GetPlatformInfo(context, opName, aicoreParams_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t perCoreNum = AlignData(tilingData->init_total_num, static_cast<int64_t>(aicoreParams_.numBlocks)) /
                         static_cast<int64_t>(aicoreParams_.numBlocks);
    tilingData->init_per_core_num = AlignData(perCoreNum, 8 * 1024);

    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(maskMode), static_cast<uint64_t>(isTnd ? 1 : 0),
                                            static_cast<uint64_t>(isSparse ? 1 : 0));
    context->SetTilingKey(tilingKey);

    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (!isTnd) {
        workspaceSize += static_cast<size_t>(info.gSize) * static_cast<size_t>(info.skSize) * sizeof(float);
    } else {
        workspaceSize += static_cast<size_t>(info.gSize) * static_cast<size_t>(info.t2) * sizeof(float);
    }
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "SparseFlashMlaSoftmaxL1Norm tiling2 completed: blockDim=%u, B=%lld, Sq=%lld, Sk=%lld, G=%lld, D=%lld, "
            "t1=%lld, t2=%lld, max_seqlen_k=%lld, k_length=%lld, cmp_ratio=%lld, isTnd=%d, isSparse=%d.",
            blockDim, info.bSize, info.sqSize, info.skSize, info.gSize, info.dSize, info.t1, info.t2, maxSeqlenK,
            kLength, cmpRatio, isTnd, isSparse);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSparseFlashMlaSoftmaxL1Norm([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashMlaSoftmaxL1Norm)
    .Tiling(SparseFlashMlaSoftmaxL1NormTilingFunc)
    .TilingParse<SparseFlashMlaSoftmaxL1NormCompileInfo>(TilingParseForSparseFlashMlaSoftmaxL1Norm);
} // namespace smlasoftmaxl1norm
} // namespace optiling
