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
 * \file mla_preprocess_v2_infershape.cpp
 * \brief
 */

#include <initializer_list>
#include <vector>

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;
namespace ops {

constexpr size_t ATTR_CACHE_MODE = 9;

constexpr size_t INPUT_INPUT = 0;
constexpr size_t INPUT_GAMMA0 = 1;
constexpr size_t INPUT_BETA0 = 2;
constexpr size_t INPUT_DESCALE0 = 6;
constexpr size_t INPUT_COS = 16;
constexpr size_t INPUT_SIN = 17;
constexpr size_t INPUT_WUK = 18;
constexpr size_t INPUT_KV_CACHE = 19;
constexpr size_t INPUT_KV_CACHE_ROPE = 20;
constexpr size_t INPUT_SLOT_MAPPING = 21;

constexpr size_t ATTR_Q_ROPE_DIM = 1;
constexpr size_t ATTR_K_ROPE_DIM = 2;
constexpr size_t ATTR_Q_ROTARY_COEFF = 4;
constexpr size_t ATTR_K_ROTARY_COEFF = 5;
constexpr size_t ATTR_TRANSPOSE_WDQ = 6;
constexpr size_t ATTR_TRANSPOSE_WUQ = 7;
constexpr size_t ATTR_TRANSPOSE_WUK = 8;
constexpr size_t ATTR_QUANT_MODE = 10;
constexpr size_t ATTR_WDKV_SPLIT_COUNT = 12;

constexpr size_t OUTPUT_Q_OUT = 0;
constexpr size_t OUTPUT_KV_CACHE_OUT = 1;
constexpr size_t OUTPUT_Q_ROPE_OUT = 2;
constexpr size_t OUTPUT_KR_CACHE_OUT = 3;
constexpr size_t OUTPUT_Q_DOWN_OUT = 4;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_SHAPE_Q = 512;
constexpr size_t DIM_SHAPE_QR = 64;
constexpr size_t DIM_SHAPE_Q_CACHE = 576;
constexpr size_t DIM_SHAPE_Q_DOWN = 1536;
constexpr int64_t ROPE_HEAD_DIM = 64;

static const gert::Shape* InputOriginShape(gert::InferShapeContext* context, size_t index)
{
    return context->GetRequiredInputShape(index);
}

static bool ShapeEquals(const gert::Shape& shape, const std::initializer_list<int64_t>& expect)
{
    if (shape.GetDimNum() != expect.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dim : expect) {
        if (shape.GetDim(index) != dim) {
            return false;
        }
        ++index;
    }
    return true;
}

static void ReportAclnnParamError(const char* opName, const char* message);

static ge::graphStatus CheckMlaPreprocessInferParams(gert::InferShapeContext* context)
{
    constexpr int64_t kRopeDimSupported = 64;
    constexpr int64_t kRotaryCoeffSupported = 2;
    constexpr int64_t kWdkvSplitCountSupported = 1;
    constexpr int64_t kCacheModeMax = 3;
    constexpr int64_t kTokenNumMax = 256;
    constexpr int64_t kHiddenSizeMin = 2048;
    constexpr int64_t kHiddenSizeMax = 10240;
    constexpr int64_t kHiddenSizeAlign = 256;
    constexpr int64_t kHeadNumMax = 128;
    constexpr int64_t kQLoraRankMin = 32;
    constexpr int64_t kQLoraRankMax = 4096;
    constexpr int64_t kQLoraRankAlign = 32;
    constexpr int64_t kNopeDimMin = 16;
    constexpr int64_t kNopeDimMax = 256;
    constexpr int64_t kNopeDimAlign = 16;
    constexpr int64_t kKvLastMerged = 576;
    constexpr int64_t kKvLastSplit = 512;
    constexpr int64_t kKvRopeLast = 64;
    constexpr int64_t kDim1 = 1;
    constexpr int64_t kDim2 = 2;

    const char* opName = context->GetNodeName();
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        ReportAclnnParamError(opName, "attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const auto* qRopeDim = attrs->GetAttrPointer<int64_t>(ATTR_Q_ROPE_DIM);
    const auto* kRopeDim = attrs->GetAttrPointer<int64_t>(ATTR_K_ROPE_DIM);
    const auto* qRotaryCoeff = attrs->GetAttrPointer<int64_t>(ATTR_Q_ROTARY_COEFF);
    const auto* kRotaryCoeff = attrs->GetAttrPointer<int64_t>(ATTR_K_ROTARY_COEFF);
    const auto* transposeWdq = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WDQ);
    const auto* transposeWuq = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WUQ);
    const auto* transposeWuk = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WUK);
    const auto* cacheModePtr = attrs->GetAttrPointer<int64_t>(ATTR_CACHE_MODE);
    const auto* quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
    const auto* wdkvSplitCount = attrs->GetAttrPointer<int64_t>(ATTR_WDKV_SPLIT_COUNT);
    if (qRopeDim == nullptr || kRopeDim == nullptr || qRotaryCoeff == nullptr || kRotaryCoeff == nullptr ||
        transposeWdq == nullptr || transposeWuq == nullptr || transposeWuk == nullptr || cacheModePtr == nullptr ||
        quantModePtr == nullptr || wdkvSplitCount == nullptr) {
        ReportAclnnParamError(opName, "required attributes are missing.");
        return ge::GRAPH_FAILED;
    }
    if (*qRopeDim != kRopeDimSupported || *kRopeDim != kRopeDimSupported) {
        ReportAclnnParamError(opName, "qRopeDim and kRopeDim only support 64.");
        return ge::GRAPH_FAILED;
    }
    if (*qRotaryCoeff != kRotaryCoeffSupported || *kRotaryCoeff != kRotaryCoeffSupported) {
        ReportAclnnParamError(opName, "qRotaryCoeff and kRotaryCoeff only support 2.");
        return ge::GRAPH_FAILED;
    }
    if (!(*transposeWdq) || !(*transposeWuq) || !(*transposeWuk)) {
        ReportAclnnParamError(opName, "transposeWdq, transposeWuq and transposeWuk only support true.");
        return ge::GRAPH_FAILED;
    }
    if (*wdkvSplitCount != kWdkvSplitCountSupported) {
        ReportAclnnParamError(opName, "wdkvSplitCount only supports 1.");
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheMode = *cacheModePtr;
    if (cacheMode < 0 || cacheMode > kCacheModeMax) {
        ReportAclnnParamError(opName, "cacheMode only supports 0, 1, 2 and 3.");
        return ge::GRAPH_FAILED;
    }
    if (*quantModePtr != 0) {
        ReportAclnnParamError(opName, "quantMode only supports 0.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape = InputOriginShape(context, INPUT_INPUT);
    const gert::Shape* gamma0Shape = InputOriginShape(context, INPUT_GAMMA0);
    const gert::Shape* beta0Shape = InputOriginShape(context, INPUT_BETA0);
    const gert::Shape* deScale0Shape = InputOriginShape(context, INPUT_DESCALE0);
    const gert::Shape* wukShape = InputOriginShape(context, INPUT_WUK);
    const gert::Shape* slotMappingShape = InputOriginShape(context, INPUT_SLOT_MAPPING);
    const gert::Shape* kvCacheShape = InputOriginShape(context, INPUT_KV_CACHE);
    const gert::Shape* kvCacheRopeShape = InputOriginShape(context, INPUT_KV_CACHE_ROPE);
    if (inputShape == nullptr || gamma0Shape == nullptr || beta0Shape == nullptr || deScale0Shape == nullptr ||
        wukShape == nullptr || slotMappingShape == nullptr || kvCacheShape == nullptr || kvCacheRopeShape == nullptr) {
        ReportAclnnParamError(opName, "required input shapes are missing.");
        return ge::GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 2) {
        ReportAclnnParamError(opName, "input shape must be [tokenNum, hiddenSize].");
        return ge::GRAPH_FAILED;
    }
    const int64_t tokenNum = inputShape->GetDim(0);
    const int64_t hiddenSize = inputShape->GetDim(1);
    if (tokenNum < 0 || tokenNum > kTokenNumMax) {
        ReportAclnnParamError(opName, "tokenNum must be in range [0, 256].");
        return ge::GRAPH_FAILED;
    }
    if (hiddenSize < kHiddenSizeMin || hiddenSize > kHiddenSizeMax || (hiddenSize % kHiddenSizeAlign) != 0) {
        ReportAclnnParamError(opName, "hiddenSize must be in range [2048, 10240] and a multiple of 256.");
        return ge::GRAPH_FAILED;
    }
    if (!ShapeEquals(*gamma0Shape, {hiddenSize}) || !ShapeEquals(*beta0Shape, {hiddenSize})) {
        ReportAclnnParamError(opName, "gamma0 and beta0 shape must be [hiddenSize].");
        return ge::GRAPH_FAILED;
    }
    if (!ShapeEquals(*slotMappingShape, {tokenNum})) {
        ReportAclnnParamError(opName, "slotMapping shape must be [tokenNum].");
        return ge::GRAPH_FAILED;
    }
    if (deScale0Shape->GetDimNum() < 1) {
        ReportAclnnParamError(opName, "deScale0 shape must contain qLoraRank.");
        return ge::GRAPH_FAILED;
    }
    const int64_t qLoraRank = deScale0Shape->GetDim(0) - kKvLastMerged;
    if (qLoraRank < kQLoraRankMin || qLoraRank > kQLoraRankMax || (qLoraRank % kQLoraRankAlign) != 0) {
        ReportAclnnParamError(opName, "qLoraRank must be in range [32, 4096] and aligned to 32.");
        return ge::GRAPH_FAILED;
    }
    if (wukShape->GetDimNum() < 2) {
        ReportAclnnParamError(opName, "wuk shape must be [headNum, nopeDim, ...].");
        return ge::GRAPH_FAILED;
    }
    const int64_t headNum = wukShape->GetDim(0);
    const int64_t nopeDim = wukShape->GetDim(1);
    if (headNum < 1 || headNum > kHeadNumMax) {
        ReportAclnnParamError(opName, "headNum must be in range [1, 128].");
        return ge::GRAPH_FAILED;
    }
    if (nopeDim < kNopeDimMin || nopeDim > kNopeDimMax || (nopeDim % kNopeDimAlign) != 0) {
        ReportAclnnParamError(opName, "nopeDim must be in range [16, 256] and aligned to 16.");
        return ge::GRAPH_FAILED;
    }

    if (cacheMode == 0 || cacheMode == 1) {
        const int64_t kvLast = cacheMode == 0 ? kKvLastMerged : kKvLastSplit;
        if (kvCacheShape->GetDimNum() != 4) {
            ReportAclnnParamError(opName, "kvCache shape must be [blockNum, blockSize, 1, kvDim] for ND cache.");
            return ge::GRAPH_FAILED;
        }
        const int64_t blockNum = kvCacheShape->GetDim(0);
        const int64_t blockSize = kvCacheShape->GetDim(1);
        if (blockNum <= 0 || blockSize <= 0 || !ShapeEquals(*kvCacheShape, {blockNum, blockSize, 1, kvLast})) {
            ReportAclnnParamError(opName, "kvCache shape does not match cacheMode.");
            return ge::GRAPH_FAILED;
        }
        if (cacheMode == 1 && !ShapeEquals(*kvCacheRopeShape, {blockNum, blockSize, 1, kKvRopeLast})) {
            ReportAclnnParamError(opName,
                                  "kvCacheRope shape must be [blockNum, blockSize, 1, 64] when cacheMode is 1.");
            return ge::GRAPH_FAILED;
        }
    } else {
        const bool blockSizeFromDim2 = kvCacheShape->GetDimNum() > static_cast<size_t>(kDim2) &&
                                       kvCacheShape->GetDim(static_cast<size_t>(kDim2)) != 1;
        const int64_t blockSize = blockSizeFromDim2 ? kvCacheShape->GetDim(static_cast<size_t>(kDim2)) :
                                                      (kvCacheShape->GetDimNum() > static_cast<size_t>(kDim1) ?
                                                           kvCacheShape->GetDim(static_cast<size_t>(kDim1)) :
                                                           0);
        if (blockSize <= 0) {
            ReportAclnnParamError(opName, "blockSize must be greater than 0.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static bool IsDisabledRopeShape(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == 0;
}

static void ReportAclnnParamError(const char* opName, const char* message)
{
    OP_LOGE_WITHOUT_REPORT(opName, "%s", message);
    const std::vector<const char*> msgKey = {"message"};
    const std::vector<const char*> msgValue = {message};
    REPORT_PREDEFINED_ERR_MSG("EZ1001", msgKey, msgValue);
}

static ge::graphStatus CheckRopeInputShapes(gert::InferShapeContext* context, int64_t tokenNum)
{
    const char* opName = context->GetNodeName();
    const gert::Shape* cosShape = context->GetRequiredInputShape(INPUT_COS);
    const gert::Shape* sinShape = context->GetRequiredInputShape(INPUT_SIN);
    if (cosShape == nullptr || sinShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "cos,sin");
        return ge::GRAPH_FAILED;
    }

    const bool cosDisablesRope = IsDisabledRopeShape(cosShape);
    const bool sinDisablesRope = IsDisabledRopeShape(sinShape);
    if (cosDisablesRope != sinDisablesRope) {
        ReportAclnnParamError(opName, "cos and sin must both have shape [0] or neither have shape [0].");
        return ge::GRAPH_FAILED;
    }
    if (!cosDisablesRope) {
        if (cosShape->GetDimNum() != 2 || sinShape->GetDimNum() != 2 || cosShape->GetDim(0) != tokenNum ||
            sinShape->GetDim(0) != tokenNum || cosShape->GetDim(1) != ROPE_HEAD_DIM ||
            sinShape->GetDim(1) != ROPE_HEAD_DIM) {
            ReportAclnnParamError(opName, "cos and sin must have shape [tokenNum, 64].");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeMlaPreprocessV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMlaPreprocessV2.");
    if (CheckMlaPreprocessInferParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"), return ge::GRAPH_FAILED);

    const int64_t* cacheModePtr = attrs->GetAttrPointer<int64_t>(ATTR_CACHE_MODE);
    if (cacheModePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheMode = *cacheModePtr;

    const gert::Shape* kvCacheShape = context->GetRequiredInputShape(INPUT_KV_CACHE);
    const gert::Shape* kvCacheRopeShape = context->GetRequiredInputShape(INPUT_KV_CACHE_ROPE);
    const gert::Shape* inputShape = context->GetRequiredInputShape(INPUT_INPUT);
    const gert::Shape* wukShape = context->GetRequiredInputShape(INPUT_WUK);

    const int64_t tokenNum = inputShape->GetDim(0);
    const int64_t headNum = wukShape->GetDim(0);

    if (CheckRopeInputShapes(context, tokenNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* qOutShape = context->GetOutputShape(OUTPUT_Q_OUT);
    gert::Shape* kvCacheOutShape = context->GetOutputShape(OUTPUT_KV_CACHE_OUT);
    gert::Shape* qDownOutShape = context->GetOutputShape(OUTPUT_Q_DOWN_OUT);

    *kvCacheOutShape = *kvCacheShape;

    qOutShape->SetDimNum(DIM_THREE);
    qOutShape->SetDim(0, tokenNum);
    qOutShape->SetDim(1, headNum);

    qDownOutShape->SetDimNum(DIM_TWO);
    qOutShape->SetDim(0, tokenNum);
    qOutShape->SetDim(1, DIM_SHAPE_Q_DOWN);

    if (cacheMode != 0) {
        gert::Shape* qRopeOutShape = context->GetOutputShape(OUTPUT_Q_ROPE_OUT);
        gert::Shape* krCacheOutShape = context->GetOutputShape(OUTPUT_KR_CACHE_OUT);
        *krCacheOutShape = *kvCacheRopeShape;

        qOutShape->SetDim(DIM_TWO, DIM_SHAPE_Q);

        qRopeOutShape->SetDimNum(DIM_THREE);
        qRopeOutShape->SetDim(0, tokenNum);
        qRopeOutShape->SetDim(1, headNum);
        qRopeOutShape->SetDim(DIM_TWO, DIM_SHAPE_QR);
    } else {
        qOutShape->SetDim(DIM_TWO, DIM_SHAPE_Q_CACHE);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMlaPreprocessV2.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlaPreprocessV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMlaPreprocess.");
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"), return ge::GRAPH_FAILED);

    const int64_t* cacheModePtr = attrs->GetAttrPointer<int64_t>(ATTR_CACHE_MODE);
    if (cacheModePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t cacheMode = *cacheModePtr;

    auto inputDtype = context->GetRequiredInputDataType(INPUT_INPUT);
    auto kvCacheDtype = context->GetRequiredInputDataType(INPUT_KV_CACHE);
    context->SetOutputDataType(OUTPUT_Q_OUT, kvCacheDtype);
    context->SetOutputDataType(OUTPUT_KV_CACHE_OUT, kvCacheDtype);
    context->SetOutputDataType(OUTPUT_Q_DOWN_OUT, inputDtype);
    if (cacheMode != 0) {
        context->SetOutputDataType(OUTPUT_Q_ROPE_OUT, inputDtype);
        context->SetOutputDataType(OUTPUT_KR_CACHE_OUT, inputDtype);
    }
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMlaPreprocessV2.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPreprocessV2).InferShape(InferShapeMlaPreprocessV2).InferDataType(InferDataTypeMlaPreprocessV2);

} // namespace ops
