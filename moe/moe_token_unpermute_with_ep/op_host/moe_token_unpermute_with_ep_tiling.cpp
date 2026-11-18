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
 * \file moe_token_unpermute_with_ep_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include <string>
#include "graph/utils/type_utils.h"
#include "moe_token_unpermute_with_ep_tiling.h"

namespace optiling {

const static int64_t UNPERMUTE_WITH_EP_INPUT_TOKENS = 0;
const static int64_t UNPERMUTE_WITH_EP_INPUT_IDX = 1;
const static int64_t UNPERMUTE_WITH_EP_INPUT_PROBS = 2;
const static int64_t UNPERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
const static int64_t UNPERMUTE_WITH_EP_ARRT_TOPK = 0;
const static int64_t UNPERMUTE_WITH_EP_ARRT_RANGE = 1;
const static int64_t UNPERMUTE_WITH_EP_ARRT_PADDED_MODE = 2;
const static int64_t UNPERMUTE_WITH_EP_ARRT_RESTORE_SHAPE = 3;
const static int64_t RANGE_SIZE = 2;
const static int64_t MAX_NUM_TOPK = 512;

ge::graphStatus TilingMoeTokenUnpermuteWithEp(gert::TilingContext* context);

ge::graphStatus TilingMoeTokenUnpermuteWithEp(gert::TilingContext* context)
{
    return UnpermuteWithEpTilingCompute(context, -1, true);
}

static inline int64_t AlignN(const int64_t x, const int64_t N)
{
    return (x + N - 1) & ~(N - 1);
}

static inline int64_t GetLengthByType(const int32_t dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            return sizeof(int16_t);
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return sizeof(int32_t);
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

static inline bool IsSupportedFloatDtype(const ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static inline std::string DtypeStr(const ge::DataType dtype)
{
    return ge::TypeUtils::DataTypeToSerialString(dtype);
}

static inline int64_t safeMod(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a % b;
}

static inline int64_t safeDiv(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a / b;
}

static inline bool isFloatDtype(const int64_t inputDtypeSize)
{
    return inputDtypeSize == GetLengthByType(ge::DT_FLOAT);
}

/*
 * 计算hidden_size=1所需要的Btye
 */
static inline int64_t ComputeUnitHSpace(const int64_t inputDtypeSize, const int64_t bufferNum)
{
    int64_t castNum = isFloatDtype(inputDtypeSize) ? 0 : CAST_NUM;
    return inputDtypeSize * (QUE_NUM + bufferNum - 1) + FLOAT_DATA_SIZE * castNum;
}

static inline int64_t ComputeMaxHiddenSize(MoeTokenUnpermuteWithEpParam& param, int64_t bufferNum)
{
    // sorted_indices和probs的预留空间；topK_num为最大值512时，至少需要5120 Btye。
    const int64_t reserveSpace = 5120;
    int64_t maxHiddenSize =
        safeDiv((param.core.maxCoreMemory - reserveSpace), ComputeUnitHSpace(param.input.tokensDtypeSize, bufferNum));

    return AlignN(maxHiddenSize - ALIGN_512, ALIGN_512);
}

static inline ge::graphStatus CheckDtypes(const gert::TilingContext* context, const bool haveProbs)
{
    auto nodeName = context->GetNodeName();
    const auto* tokensDesc = context->GetInputDesc(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    OP_CHECK_IF(tokensDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "permuted_tokens", "desc is nullptr."),
                return ge::GRAPH_FAILED);
    const ge::DataType tokensDtype = tokensDesc->GetDataType();
    const auto* indicesDesc = context->GetInputDesc(UNPERMUTE_WITH_EP_INPUT_IDX);
    OP_CHECK_IF(indicesDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sorted_indices", "desc is nullptr."),
                return ge::GRAPH_FAILED);
    const ge::DataType indicesDtype = indicesDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedFloatDtype(tokensDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "permuted_tokens", DtypeStr(tokensDtype).c_str(),
                                                      "permuted_tokens only supports float32/float16/bfloat16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "sorted_indices", DtypeStr(indicesDtype).c_str(),
                                                      "sorted_indices only supports int32."),
                return ge::GRAPH_FAILED);
    if (haveProbs) {
        const auto* probsDesc = context->GetOptionalInputDesc(UNPERMUTE_WITH_EP_INPUT_PROBS);
        OP_CHECK_IF(probsDesc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "probs", "probs desc is nullptr."),
                    return ge::GRAPH_FAILED);
        const ge::DataType probsDtype = probsDesc->GetDataType();
        OP_CHECK_IF(!IsSupportedFloatDtype(probsDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "probs", DtypeStr(probsDtype).c_str(),
                                                          "probs only supports float32/float16/bfloat16."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus MoeTokenUnpermuteWithEpInputParamCheck(const gert::TilingContext* context,
                                                                     const bool isUnpermute)
{
    const gert::StorageShape* tokensShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    const gert::StorageShape* indicesShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
    const gert::StorageShape* probsShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);
    auto dataTensor0 = context->GetInputTensor(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    auto dataTensor1 = context->GetInputTensor(UNPERMUTE_WITH_EP_INPUT_IDX);
    auto nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "attrs", "attrs is nullptr."),
                return ge::GRAPH_FAILED);
    const int64_t* tmpTopK = attrs->GetAttrPointer<int64_t>(UNPERMUTE_WITH_EP_ARRT_TOPK);
    OP_CHECK_IF(tmpTopK == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "num_topk", "num_topk is nullptr."),
                return ge::GRAPH_FAILED);
    const int64_t inputTopK = *tmpTopK;

    OP_CHECK_IF(inputTopK < 1,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "num_topk", "input num TopK cannot less than 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputTopK > MAX_NUM_TOPK,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "num_topk",
                                                 "input num TopK(" + std::to_string(inputTopK) +
                                                     ") can not larger than " + std::to_string(MAX_NUM_TOPK) + "."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokensShape == nullptr || indicesShape == nullptr || dataTensor0 == nullptr || dataTensor1 == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(nodeName),
                                                         "tokensShape, indicesShape, dataTensor0, dataTensor1",
                                                         "permutedTokens or sortedIndices is nullptr."),
                return ge::GRAPH_FAILED);
    if (CheckDtypes(context, probsShape != nullptr) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    const bool* paddedModePtr = attrs->GetAttrPointer<bool>(UNPERMUTE_WITH_EP_ARRT_PADDED_MODE);
    OP_CHECK_IF(
        paddedModePtr != nullptr && *paddedModePtr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "padded_mode", "true", "padded_mode only supports false."),
        return ge::GRAPH_FAILED);
    // restore_shape is an unsupported (must be empty) optional list attribute.
    // The generated aclnn wrapper encodes a null restoreShapeOptional as a
    // single-element array, but the encoded value is not stable: it is [0] on
    // the plain path while it can be [1] when AutoContiguous nodes are inserted
    // for non-contiguous inputs. A single-element list cannot be a valid shape,
    // so only a list with two or more elements is treated as genuinely
    // provided; the read is additionally guarded by the real attribute count
    // because this last attribute may be absent when no default is materialized.
    bool restoreShapeProvided = false;
    const size_t attrNum = attrs->GetAttrNum();
    if (attrNum > static_cast<size_t>(UNPERMUTE_WITH_EP_ARRT_RESTORE_SHAPE)) {
        const gert::ContinuousVector* restoreShapePtr =
            attrs->GetAttrPointer<gert::ContinuousVector>(UNPERMUTE_WITH_EP_ARRT_RESTORE_SHAPE);
        restoreShapeProvided = (restoreShapePtr != nullptr && restoreShapePtr->GetSize() > 1);
    }
    OP_CHECK_IF(restoreShapeProvided,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "restore_shape", "non-empty",
                                                      "restore_shape is not supported and must be empty."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        tokensShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "permuted_tokens", "permutedTokens's shape is not 2D."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokensShape->GetStorageShape().GetDim(0) == 0 || tokensShape->GetStorageShape().GetDim(1) == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "permuted_tokens",
                                                         "permutedTokens does not support empty tensor."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "sorted_indices",
                                                         std::to_string(indicesShape->GetStorageShape().GetDimNum()),
                                                         "sortedIndices's shape is not 1D."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDim(0) == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sorted_indices",
                                                         "sortedIndices does not support empty tensor."),
                return ge::GRAPH_FAILED);

    const int64_t totalLength = indicesShape->GetStorageShape().GetDim(0);
    if (probsShape != nullptr) {
        int64_t probsDimNum = probsShape->GetStorageShape().GetDimNum();
        if (isUnpermute) {
            OP_CHECK_IF(probsDimNum != 2,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "probs", std::to_string(probsDimNum),
                            "probs's shape(" + std::to_string(probsDimNum) + ") is not 2D."),
                        return ge::GRAPH_FAILED);

            int64_t tokensNum = probsShape->GetStorageShape().GetDim(0);
            int64_t topK = probsShape->GetStorageShape().GetDim(1);

            OP_CHECK_IF(
                topK != inputTopK,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    nodeName, "probs", "input topK(" + std::to_string(inputTopK) + ") can not match probs' dim(1)."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(totalLength != tokensNum * topK,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            nodeName, "permuted_tokens, probs",
                            "permutedTokens's dim(0) is not equal to probs's dim(0) * probs's dim(1)."),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(probsDimNum != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "probs", std::to_string(probsDimNum),
                            "permutedProbsGrad's shape(" + std::to_string(probsDimNum) + ") is not 1D."),
                        return ge::GRAPH_FAILED);
        }
    } else if (isUnpermute) {
        OP_CHECK_IF(safeMod(totalLength, inputTopK) != 0 || safeDiv(totalLength, inputTopK) < 1,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sorted_indices",
                                                             "sortedIndices length(" + std::to_string(totalLength) +
                                                                 ") must be a positive multiple of num_topk(" +
                                                                 std::to_string(inputTopK) + ")."),
                    return ge::GRAPH_FAILED);
    }

    if (isUnpermute) {
        const gert::StorageShape* outShape = context->GetOutputShape(UNPERMUTE_WITH_EP_OUTPUT_TOKENS);
        if (outShape != nullptr) {
            const gert::Shape& outStorageShape = outShape->GetStorageShape();
            const int64_t hiddenSize = tokensShape->GetStorageShape().GetDim(1);
            const int64_t expectTokens =
                (probsShape != nullptr) ? probsShape->GetStorageShape().GetDim(0) : safeDiv(totalLength, inputTopK);
            OP_CHECK_IF(outStorageShape.GetDimNum() != 2,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "unpermuted_tokens",
                                                                 std::to_string(outStorageShape.GetDimNum()),
                                                                 "unpermuted_tokens's shape is not 2D."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(outStorageShape.GetDim(0) >= 0 && outStorageShape.GetDim(0) != expectTokens,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            nodeName, "unpermuted_tokens",
                            "unpermuted_tokens dim(0)(" + std::to_string(outStorageShape.GetDim(0)) +
                                ") is not equal to num_tokens(" + std::to_string(expectTokens) + ")."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(outStorageShape.GetDim(1) >= 0 && outStorageShape.GetDim(1) != hiddenSize,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            nodeName, "unpermuted_tokens",
                            "unpermuted_tokens dim(1)(" + std::to_string(outStorageShape.GetDim(1)) +
                                ") is not equal to hidden_size(" + std::to_string(hiddenSize) + ")."),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus Init(const gert::TilingContext* context, const int64_t topK,
                                   MoeTokenUnpermuteWithEpParam& param, const bool isUnpermute)
{
    auto nodeName = context->GetNodeName();
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "platform_info", "platform info is nullptr."),
                return ge::GRAPH_FAILED);
    auto ascendPlaform = platform_ascendc::PlatformAscendC(platformInfo);
    param.core.maxCoreNum = static_cast<int64_t>(ascendPlaform.GetCoreNumAiv());
    OP_CHECK_IF(param.core.maxCoreNum <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "max_core_num", "AIV core num must be positive."),
                return ge::GRAPH_FAILED);
    uint64_t maxCoreMemory = 0;
    ascendPlaform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxCoreMemory);
    param.core.maxCoreMemory = static_cast<int64_t>(maxCoreMemory);
    OP_CHECK_IF(param.core.maxCoreMemory == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "max_core_memory", "UB size must be positive."),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* tokensShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    const gert::StorageShape* sortedIndicesShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
    const gert::StorageShape* probsShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);

    const auto* tokensDesc = context->GetInputDesc(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    OP_CHECK_IF(tokensDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "permuted_tokens", "desc is nullptr."),
                return ge::GRAPH_FAILED);
    const auto* indicesDesc = context->GetInputDesc(UNPERMUTE_WITH_EP_INPUT_IDX);
    OP_CHECK_IF(indicesDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sorted_indices", "desc is nullptr."),
                return ge::GRAPH_FAILED);
    param.input.tokensDtypeSize = GetLengthByType(tokensDesc->GetDataType());
    param.input.indicesDtypeSize = GetLengthByType(indicesDesc->GetDataType());
    param.input.isUnpermute = isUnpermute;

    auto attrPtr = context->GetAttrs();
    const int64_t* tmpTopK = attrPtr->GetAttrPointer<int64_t>(UNPERMUTE_WITH_EP_ARRT_TOPK);
    const int64_t inputTopK = *tmpTopK;
    auto rangePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(UNPERMUTE_WITH_EP_ARRT_RANGE);
    if (rangePtr != nullptr && rangePtr->GetSize() != 0) {
        OP_CHECK_IF(rangePtr->GetSize() != RANGE_SIZE,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "range", "the size of range only support 2"),
                    return ge::GRAPH_FAILED);
        const int64_t* rangeList = reinterpret_cast<const int64_t*>(rangePtr->GetData());
        param.input.start = rangeList[0];
        param.input.end = rangeList[1];
    } else {
        param.input.start = 0;
        param.input.end = 0;
    }

    param.input.numOutTokens = tokensShape->GetStorageShape().GetDim(0); // numOutTokens根据tokens第0维获取；
    param.input.hiddenSize = tokensShape->GetStorageShape().GetDim(1);
    param.input.totalLength = sortedIndicesShape->GetStorageShape().GetDim(
        0); // tokens可能存在numOutTokens，因此totalLength从sortedIndices获取。

    OP_CHECK_IF(param.input.start < 0 || param.input.start > param.input.end,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "range",
                                                         "range must be [start, end] with 0 <= start <= end, got [" +
                                                             std::to_string(param.input.start) + ", " +
                                                             std::to_string(param.input.end) + "]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        param.input.end - param.input.start > param.input.numOutTokens,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "range",
                                                 "range span(" + std::to_string(param.input.end - param.input.start) +
                                                     ") can not be larger than permuted_tokens dim(0)(" +
                                                     std::to_string(param.input.numOutTokens) + ")."),
        return ge::GRAPH_FAILED);

    if (isUnpermute) {
        if (probsShape != nullptr) {
            const auto* probsDesc = context->GetOptionalInputDesc(UNPERMUTE_WITH_EP_INPUT_PROBS);
            OP_CHECK_IF(probsDesc == nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "probs", "probs desc is nullptr."),
                        return ge::GRAPH_FAILED);
            param.input.probsDtypeSize = GetLengthByType(probsDesc->GetDataType());
            param.input.haveProbs = true;
            param.input.tokensNum = probsShape->GetStorageShape().GetDim(0);
            param.input.topK = probsShape->GetStorageShape().GetDim(1);
            param.input.PermuteProbGradSize = 0;
        } else {
            param.input.topK = inputTopK;
            param.input.PermuteProbGradSize = 1;
            param.input.haveProbs = false;
            param.input.tokensNum = safeDiv(param.input.totalLength, param.input.topK);
        }
    } else {
        param.input.topK = (inputTopK == topK) ? topK : 1;
        param.input.PermuteProbGradSize = 1;
        param.input.haveProbs = (probsShape != nullptr) ? true : false;
        param.input.tokensNum = safeDiv(param.input.totalLength, param.input.topK);
    }
    return ge::GRAPH_SUCCESS;
}

static void SetCoreNum(MoeTokenUnpermuteWithEpParam& param)
{
    if (param.input.tokensNum < param.core.maxCoreNum) {
        param.core.usedCoreNum = param.input.tokensNum;
    } else {
        param.core.usedCoreNum = param.core.maxCoreNum;
    }
}

static inline ge::graphStatus TilingHiddenSize(MoeTokenUnpermuteWithEpParam& param)
{
    int64_t maxHiddenSize = ComputeMaxHiddenSize(param, MIN_BUFFER_NUM);
    OP_CHECK_IF(maxHiddenSize <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    "MoeTokenUnpermuteWithEp", "hidden_size",
                    "UB is too small to hold any hidden_size block for the current dtype."),
                return ge::GRAPH_FAILED);
    if (AlignN(param.input.hiddenSize, ALIGN_512) <= maxHiddenSize) {
        param.hiddenTiling.length = param.input.hiddenSize;
        param.hiddenTiling.remain = 0;
        param.hiddenTiling.num = 1;
    } else {
        param.hiddenTiling.length = maxHiddenSize;
        param.hiddenTiling.remain = safeMod(param.input.hiddenSize, maxHiddenSize);
        param.hiddenTiling.num = safeDiv(param.input.hiddenSize, maxHiddenSize);
    }
    return ge::GRAPH_SUCCESS;
}

static inline void SetBufferNum(MoeTokenUnpermuteWithEpParam& param)
{
    const int64_t maxBufferNum = 4;
    int64_t bufferNum = maxBufferNum;
    while (bufferNum > MIN_BUFFER_NUM && param.hiddenTiling.length > ComputeMaxHiddenSize(param, bufferNum)) {
        bufferNum--;
    }
    param.core.bufferNum = bufferNum;
}

static inline void ComputeRemainMemerySpace(MoeTokenUnpermuteWithEpParam& param)
{
    param.core.remainMemorySpace =
        param.core.maxCoreMemory - AlignN(param.hiddenTiling.length, ALIGN_512) *
                                       ComputeUnitHSpace(param.input.tokensDtypeSize, param.core.bufferNum);
    param.core.remainMemorySpace -= ALIGN_256;
}

static inline ge::graphStatus TilingToken(MoeTokenUnpermuteWithEpParam& param)
{
    param.tokenPerCore.length = safeDiv(param.input.tokensNum, param.core.usedCoreNum);
    param.tokenPerCore.num = param.core.usedCoreNum;
    param.tokenPerCore.remain = safeMod(param.input.tokensNum, param.core.usedCoreNum);

    int64_t unitTokenSpace = param.input.indicesDtypeSize;
    if (param.input.haveProbs && param.input.isUnpermute) {
        unitTokenSpace += param.input.probsDtypeSize;
        if (!isFloatDtype(param.input.probsDtypeSize)) {
            unitTokenSpace += static_cast<int64_t>(sizeof(float));
        }
    }

    int64_t probIndiceSpace = param.tokenPerCore.length * param.input.topK * unitTokenSpace;

    if (param.core.remainMemorySpace >= probIndiceSpace) {
        param.tokenTiling.length = param.tokenPerCore.length;
        param.tokenTiling.remain = 0;
        param.tokenTiling.num = 1;
    } else {
        int64_t maxTokenSize = safeDiv(param.core.remainMemorySpace, (param.input.topK * unitTokenSpace));
        OP_CHECK_IF(
            param.input.topK <= 0 || unitTokenSpace <= 0 || param.core.remainMemorySpace <= 0 || maxTokenSize <= 0,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                "MoeTokenUnpermuteWithEp", "num_topk",
                "num_topk(" + std::to_string(param.input.topK) +
                    ") is too large for the available UB to hold sorted_indices/probs."),
            return ge::GRAPH_FAILED);
        param.tokenTiling.length = maxTokenSize;
        param.tokenTiling.remain = safeMod(param.tokenPerCore.length, maxTokenSize);
        param.tokenTiling.num = safeDiv(param.tokenPerCore.length, maxTokenSize);
    }
    OP_CHECK_IF(param.tokenTiling.length <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("MoeTokenUnpermuteWithEp", "token_tiling",
                                                         "token tiling length must be positive."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
/*
  tilingKey计算规则
    0表示probs为None，
    1 2 3 表示 DT_FLOAT DT_FLOAT16 DT_BF16
 */
static inline ge::graphStatus SetTilingKey(const gert::TilingContext* context, const bool isUnpermute,
                                           MoeTokenUnpermuteWithEpParam& param)
{
    if (param.input.haveProbs) {
        // 存在probs
        if (isUnpermute == true) {
            // 在unpermute的条件下
            const auto* probsDesc = context->GetOptionalInputDesc(UNPERMUTE_WITH_EP_INPUT_PROBS);
            OP_CHECK_IF(
                probsDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "probs", "probs desc is nullptr."),
                return ge::GRAPH_FAILED);
            int64_t dataTensor2Type = probsDesc->GetDataType();
            if (dataTensor2Type == ge::DT_FLOAT) {
                param.core.tilingKey = TILINGKEY_FLOAT;
            } else if (dataTensor2Type == ge::DT_FLOAT16) {
                param.core.tilingKey = TILINGKEY_FLOAT16;
            } else if (dataTensor2Type == ge::DT_BF16) {
                param.core.tilingKey = TILINGKEY_BF16;
            } else {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "probs",
                                                      DtypeStr(static_cast<ge::DataType>(dataTensor2Type)).c_str(),
                                                      "the data type of probs is incorrect");
                return ge::GRAPH_FAILED;
            }
        } else {
            // 在permute的条件下
            param.core.tilingKey = 1;
        }
    } else {
        // 不存在probs
        param.core.tilingKey = 0;
    }
    return ge::GRAPH_SUCCESS;
}

static inline void SetTilingData(gert::TilingContext* context, const MoeTokenUnpermuteWithEpParam& param)
{
    MoeTokenUnpermuteWithEpTilingData tilingData;
    tilingData.set_hidden_size(param.input.hiddenSize);
    tilingData.set_top_k(param.input.topK);
    tilingData.set_start(param.input.start);
    tilingData.set_end(param.input.end);
    tilingData.set_permuted_probs_grad_length(param.input.PermuteProbGradSize);
    tilingData.set_num_out_tokens(param.input.numOutTokens);
    tilingData.set_hidden_splited_length(param.hiddenTiling.length);
    tilingData.set_hidden_splited_num(param.hiddenTiling.num);
    tilingData.set_hidden_splited_remain(param.hiddenTiling.remain);
    tilingData.set_tokens_core_length(param.tokenPerCore.length);
    tilingData.set_tokens_core_remain(param.tokenPerCore.remain);
    tilingData.set_tokens_splited_length(param.tokenTiling.length);
    tilingData.set_tokens_splited_remain(param.tokenTiling.remain);
    tilingData.set_tokens_splited_num(param.tokenTiling.num);
    tilingData.set_buffer_num(param.core.bufferNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(param.core.usedCoreNum);
}

static inline void DebugPrint(const gert::TilingContext* context, const MoeTokenUnpermuteWithEpParam& param)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print MoeTokenUnpermuteWithEp tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "coreNum: %ld", param.core.usedCoreNum);
    OP_LOGD(nodeName, "maxCoreMemory: %ld", param.core.maxCoreMemory);
    OP_LOGD(nodeName, "maxCoreNum: %ld", param.core.maxCoreNum);
    OP_LOGD(nodeName, "remainMemorySpace: %ld", param.core.remainMemorySpace);
    OP_LOGD(nodeName, "tilingKey: %ld", param.core.tilingKey);
    OP_LOGD(nodeName, "bufferNum: %ld", param.core.bufferNum);

    OP_LOGD(nodeName, "totalLength: %ld", param.input.totalLength);
    OP_LOGD(nodeName, "tokensNum: %ld", param.input.tokensNum);
    OP_LOGD(nodeName, "topK: %ld", param.input.topK);
    OP_LOGD(nodeName, "hiddenSize: %ld", param.input.hiddenSize);
    OP_LOGD(nodeName, "haveProbs: %d", param.input.haveProbs);
    OP_LOGD(nodeName, "tokensDtypeSize: %ld", param.input.tokensDtypeSize);
    OP_LOGD(nodeName, "indicesDtypeSize: %ld", param.input.indicesDtypeSize);
    OP_LOGD(nodeName, "probsDtypeSize: %ld", param.input.probsDtypeSize);
    OP_LOGD(nodeName, "start: %ld", param.input.start);
    OP_LOGD(nodeName, "end: %ld", param.input.end);
    OP_LOGD(nodeName, "numOutTokens: %ld", param.input.numOutTokens);

    OP_LOGD(nodeName, "hiddenTilingParam.length: %ld", param.hiddenTiling.length);
    OP_LOGD(nodeName, "hiddenTilingParam.num: %ld", param.hiddenTiling.num);
    OP_LOGD(nodeName, "hiddenTilingParam.remain: %ld", param.hiddenTiling.remain);

    OP_LOGD(nodeName, "tokenTilingParam.length: %ld", param.tokenTiling.length);
    OP_LOGD(nodeName, "tokenTilingParam.num: %ld", param.tokenTiling.num);
    OP_LOGD(nodeName, "tokenTilingParam.remain: %ld", param.tokenTiling.remain);

    OP_LOGD(nodeName, "tokenPerCore.length: %ld", param.tokenPerCore.length);
    OP_LOGD(nodeName, "tokenPerCore.num: %ld", param.tokenPerCore.num);
    OP_LOGD(nodeName, "tokenPerCore.remain: %ld", param.tokenPerCore.remain);

    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print MoeTokenUnpermuteWithEp tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus UnpermuteWithEpTilingCompute(gert::TilingContext* context, const int64_t topK, const bool isUnpermute)
{
    MoeTokenUnpermuteWithEpParam param;

    if (MoeTokenUnpermuteWithEpInputParamCheck(context, isUnpermute) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (Init(context, topK, param, isUnpermute) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetCoreNum(param);
    if (TilingHiddenSize(param) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetBufferNum(param);
    ComputeRemainMemerySpace(param);
    if (TilingToken(param) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (SetTilingKey(context, isUnpermute, param) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingData(context, param);
    DebugPrint(context, param);
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = 0;
    return context->SetTilingKey(param.core.tilingKey);
}

static ge::graphStatus TilingPrepareForMoeTokenUnpermuteWithEp(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MoeTokenUnpermuteWithEpCompileInfo {};

IMPL_OP_OPTILING(MoeTokenUnpermuteWithEp)
    .Tiling(TilingMoeTokenUnpermuteWithEp)
    .TilingParse<MoeTokenUnpermuteWithEpCompileInfo>(TilingPrepareForMoeTokenUnpermuteWithEp);
} // namespace optiling
