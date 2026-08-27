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
 * \file fused_gdn_decode_tiling.cpp
 * \brief
 */

#include "fused_gdn_decode_tiling.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "err/ops_err.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_gdn_decode_tiling_data.h"

using namespace FusedGdnDecode;

namespace optiling {
namespace {
constexpr size_t MIXED_INDEX = 0;
constexpr size_t A_INDEX = 1;
constexpr size_t B_INDEX = 2;
constexpr size_t A_LOG_INDEX = 3;
constexpr size_t DT_BIAS_INDEX = 4;
constexpr size_t STATE_INDEX = 5;
constexpr size_t STATE_INDICES_INDEX = 6;
constexpr uint64_t TILING_BF16_STATE_FP32 = 1;
constexpr uint64_t TILING_FP16_STATE_FP32 = 2;
constexpr uint64_t TILING_BF16_STATE_BF16 = 3;
constexpr uint64_t TILING_FP16_STATE_FP16 = 4;
constexpr uint64_t UB_ALIGN_BYTES = 256;
constexpr uint64_t SCALAR_UB_ELEMS = 192;
constexpr uint32_t BLOCK_ELEMS_INT32 = 8;
constexpr uint32_t K_ALIGN_ELEMS = 16;
constexpr uint32_t FP32_ELEMS_PER_BLOCK = 8;
constexpr uint32_t MAX_DATA_COPY_PAD_BYTES = 32;
constexpr uint32_t MIN_SUPPORTED_K = 64;
constexpr uint32_t MAX_SUPPORTED_K =
    (std::numeric_limits<uint8_t>::max() * FP32_ELEMS_PER_BLOCK / K_ALIGN_ELEMS) * K_ALIGN_ELEMS;

uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return x / y + static_cast<uint32_t>(x % y != 0);
}

uint32_t Align(uint32_t x, uint32_t y)
{
    return CeilDiv(x, y) * y;
}

bool CheckRank(const gert::Shape &shape, size_t expectedRank, const char *name)
{
    if (shape.GetDimNum() != expectedRank) {
        OP_LOGE("FusedGdnDecode", "%s rank must be %zu, but got %zu", name, expectedRank, shape.GetDimNum());
        return false;
    }
    return true;
}

bool CheckPositiveDims(const gert::Shape &shape, const char *name)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            OP_LOGE("FusedGdnDecode", "%s dimension %zu must be positive, but got %ld", name, i, shape.GetDim(i));
            return false;
        }
    }
    return true;
}

bool CheckU32Dim(int64_t dim, const char *name)
{
    if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE("FusedGdnDecode", "%s must be in (0, UINT32_MAX], but got %ld", name, dim);
        return false;
    }
    return true;
}

bool CheckMulU64(uint64_t lhs, uint64_t rhs, const char *name)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        OP_LOGE("FusedGdnDecode", "%s overflows uint64", name);
        return false;
    }
    return true;
}

uint64_t EstimateQueueBytes(uint32_t k, uint32_t v, uint32_t bv, uint32_t stateBytes, uint32_t indexBufferElems)
{
    const uint32_t alignK = Align(k, K_ALIGN_ELEMS);
    const uint32_t alignBV = Align(bv, 16);
    const uint64_t qkBytes = 2ULL * alignK * sizeof(uint16_t);
    const uint64_t indexBytes = static_cast<uint64_t>(indexBufferElems) * sizeof(int32_t);
    const uint64_t stateSlotBytes =
        static_cast<uint64_t>(alignK) * alignBV * stateBytes + static_cast<uint64_t>(alignBV) * sizeof(uint16_t);
    const uint64_t stateBufferNum = bv >= v ? 1ULL : 2ULL;
    return qkBytes + indexBytes + 2ULL * stateBufferNum * stateSlotBytes;
}

uint64_t EstimateComputeBytes(uint32_t k, uint32_t bv, uint32_t stateBytes)
{
    const uint32_t alignK = Align(k, K_ALIGN_ELEMS);
    const uint32_t alignBV = Align(bv, 16);
    const uint64_t computeMatrixBytes = stateBytes == sizeof(float) ? 0ULL : 2ULL * alignK * alignBV * sizeof(float);
    return (3ULL * alignK + 3ULL * alignBV + SCALAR_UB_ELEMS) * sizeof(float) + computeMatrixBytes +
           9ULL * UB_ALIGN_BYTES;
}

uint64_t GetReduceTmpBytes(uint32_t k, uint32_t bv)
{
    const uint32_t alignK = Align(k, K_ALIGN_ELEMS);
    uint32_t maxTmpBytes = 0;
    uint32_t minTmpBytes = 0;
    const ge::Shape shape({static_cast<int64_t>(bv), static_cast<int64_t>(alignK)});
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, true,
                                       maxTmpBytes, minTmpBytes);
    return minTmpBytes;
}

uint64_t GetSigmoidTmpBytes()
{
    uint32_t maxTmpBytes = 0;
    uint32_t minTmpBytes = 0;
    const ge::Shape shape({static_cast<int64_t>(FP32_ELEMS_PER_BLOCK)});
    AscendC::GetSigmoidMaxMinTmpSize(shape, sizeof(float), false, maxTmpBytes, minTmpBytes);
    return minTmpBytes;
}

uint64_t GetStackTmpBytes(uint32_t k, uint32_t bv)
{
    const uint64_t reduceTmpBytes = GetReduceTmpBytes(k, bv);
    const uint64_t sigmoidTmpBytes = GetSigmoidTmpBytes();
    return reduceTmpBytes > sigmoidTmpBytes ? reduceTmpBytes : sigmoidTmpBytes;
}

uint64_t EstimateUbBytes(uint32_t k, uint32_t v, uint32_t bv, uint32_t stateBytes, uint32_t indexBufferElems)
{
    return EstimateQueueBytes(k, v, bv, stateBytes, indexBufferElems) + EstimateComputeBytes(k, bv, stateBytes) +
           GetStackTmpBytes(k, bv);
}

uint32_t SelectBv(uint32_t k, uint32_t v, uint32_t stateBytes, uint32_t indexBufferElems, uint64_t ubSize)
{
    constexpr uint32_t candidates[] = {128, 64, 32, 16, 8};
    for (uint32_t candidate : candidates) {
        if (candidate <= v && EstimateUbBytes(k, v, candidate, stateBytes, indexBufferElems) <= ubSize) {
            return candidate;
        }
    }
    const uint32_t tailCandidate = v < 8 ? v : 8;
    if (EstimateUbBytes(k, v, tailCandidate, stateBytes, indexBufferElems) <= ubSize) {
        return tailCandidate;
    }
    return 0;
}

bool GetPlatformInfo(const gert::TilingContext *context, uint32_t &aivNum, uint64_t &ubSize)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        aivNum = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        return aivNum > 0 && ubSize > 0;
    }
    auto compileInfo = reinterpret_cast<const FusedGdnDecodeCompileInfo *>(context->GetCompileInfo());
    if (compileInfo == nullptr) {
        return false;
    }
    aivNum = compileInfo->aivNum;
    ubSize = compileInfo->ubSize;
    return aivNum > 0 && ubSize > 0;
}
} // namespace

ge::graphStatus FusedGdnDecodeTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("FusedGdnDecode", "tiling context is null"), return ge::GRAPH_FAILED);

    auto mixedShapePtr = context->GetInputShape(MIXED_INDEX);
    auto aShapePtr = context->GetInputShape(A_INDEX);
    auto bShapePtr = context->GetInputShape(B_INDEX);
    auto aLogShapePtr = context->GetInputShape(A_LOG_INDEX);
    auto dtBiasShapePtr = context->GetInputShape(DT_BIAS_INDEX);
    auto stateShapePtr = context->GetInputShape(STATE_INDEX);
    auto stateIndicesShapePtr = context->GetInputShape(STATE_INDICES_INDEX);
    OP_CHECK_IF(mixedShapePtr == nullptr || aShapePtr == nullptr || bShapePtr == nullptr || aLogShapePtr == nullptr ||
                    dtBiasShapePtr == nullptr || stateShapePtr == nullptr || stateIndicesShapePtr == nullptr,
                OP_LOGE("FusedGdnDecode", "input shape is null"), return ge::GRAPH_FAILED);

    const auto &mixedShape = mixedShapePtr->GetOriginShape();
    const auto &aShape = aShapePtr->GetOriginShape();
    const auto &bShape = bShapePtr->GetOriginShape();
    const auto &aLogShape = aLogShapePtr->GetOriginShape();
    const auto &dtBiasShape = dtBiasShapePtr->GetOriginShape();
    const auto &stateShape = stateShapePtr->GetOriginShape();
    const auto &stateIndicesShape = stateIndicesShapePtr->GetOriginShape();
    OP_CHECK_IF(!CheckRank(mixedShape, 2, "mixed_qkv") || !CheckRank(aShape, 2, "a") || !CheckRank(bShape, 2, "b") ||
                    !CheckRank(aLogShape, 1, "a_log") || !CheckRank(dtBiasShape, 1, "dt_bias") ||
                    !CheckRank(stateShape, 4, "state") || !CheckRank(stateIndicesShape, 1, "ssm_state_indices"),
                OP_LOGE("FusedGdnDecode", "input rank validation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckPositiveDims(mixedShape, "mixed_qkv") || !CheckPositiveDims(aShape, "a") ||
                    !CheckPositiveDims(bShape, "b") || !CheckPositiveDims(aLogShape, "a_log") ||
                    !CheckPositiveDims(dtBiasShape, "dt_bias") || !CheckPositiveDims(stateShape, "state") ||
                    !CheckPositiveDims(stateIndicesShape, "ssm_state_indices"),
                OP_LOGE("FusedGdnDecode", "input dimension validation failed"), return ge::GRAPH_FAILED);

    const int64_t batchI64 = mixedShape.GetDim(0);
    const int64_t mixedDimI64 = mixedShape.GetDim(1);
    const int64_t slotsI64 = stateShape.GetDim(0);
    const int64_t hvI64 = stateShape.GetDim(1);
    const int64_t vI64 = stateShape.GetDim(2);
    const int64_t kI64 = stateShape.GetDim(3);
    OP_CHECK_IF(!CheckU32Dim(batchI64, "B") || !CheckU32Dim(mixedDimI64, "mixedDim") ||
                    !CheckU32Dim(slotsI64, "slots") || !CheckU32Dim(hvI64, "HV") || !CheckU32Dim(vI64, "V") ||
                    !CheckU32Dim(kI64, "K"),
                OP_LOGE("FusedGdnDecode", "dimension exceeds supported range"), return ge::GRAPH_FAILED);

    const uint32_t batch = static_cast<uint32_t>(batchI64);
    const uint32_t mixedDim = static_cast<uint32_t>(mixedDimI64);
    const uint32_t hv = static_cast<uint32_t>(hvI64);
    const uint32_t v = static_cast<uint32_t>(vI64);
    const uint32_t k = static_cast<uint32_t>(kI64);
    OP_CHECK_IF(k < MIN_SUPPORTED_K || k > MAX_SUPPORTED_K,
                OP_LOGE("FusedGdnDecode", "K must be in [%u, %u], but got %u", MIN_SUPPORTED_K, MAX_SUPPORTED_K, k),
                return ge::GRAPH_FAILED);
    const uint64_t valueDim = static_cast<uint64_t>(hv) * v;
    OP_CHECK_IF(valueDim >= mixedDim || valueDim > std::numeric_limits<uint32_t>::max(),
                OP_LOGE("FusedGdnDecode", "mixed_qkv last dimension is too small"), return ge::GRAPH_FAILED);

    const uint32_t qkDim = mixedDim - static_cast<uint32_t>(valueDim);
    OP_CHECK_IF(qkDim % (2 * k) != 0, OP_LOGE("FusedGdnDecode", "mixed_qkv is inconsistent with K/HV/V"),
                return ge::GRAPH_FAILED);
    const uint32_t h = qkDim / (2 * k);
    OP_CHECK_IF(h == 0 || hv % h != 0,
                OP_LOGE("FusedGdnDecode", "HV must be divisible by derived H, H=%u HV=%u", h, hv),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(aShape.GetDim(0) != batchI64 || aShape.GetDim(1) != hvI64 || bShape.GetDim(0) != batchI64 ||
                    bShape.GetDim(1) != hvI64 || aLogShape.GetDim(0) != hvI64 || dtBiasShape.GetDim(0) != hvI64 ||
                    stateIndicesShape.GetDim(0) != batchI64,
                OP_LOGE("FusedGdnDecode", "input shapes are inconsistent"), return ge::GRAPH_FAILED);

    auto mixedDesc = context->GetInputDesc(MIXED_INDEX);
    auto aDesc = context->GetInputDesc(A_INDEX);
    auto bDesc = context->GetInputDesc(B_INDEX);
    auto aLogDesc = context->GetInputDesc(A_LOG_INDEX);
    auto dtBiasDesc = context->GetInputDesc(DT_BIAS_INDEX);
    auto stateDesc = context->GetInputDesc(STATE_INDEX);
    auto stateIndicesDesc = context->GetInputDesc(STATE_INDICES_INDEX);
    OP_CHECK_IF(mixedDesc == nullptr || aDesc == nullptr || bDesc == nullptr || aLogDesc == nullptr ||
                    dtBiasDesc == nullptr || stateDesc == nullptr || stateIndicesDesc == nullptr,
                OP_LOGE("FusedGdnDecode", "input desc is null"), return ge::GRAPH_FAILED);

    const ge::DataType mixedDtype = mixedDesc->GetDataType();
    const ge::DataType stateDtype = stateDesc->GetDataType();
    OP_CHECK_IF((mixedDtype != ge::DT_BF16 && mixedDtype != ge::DT_FLOAT16) || aDesc->GetDataType() != mixedDtype ||
                    bDesc->GetDataType() != mixedDtype || dtBiasDesc->GetDataType() != mixedDtype ||
                    aLogDesc->GetDataType() != ge::DT_FLOAT || stateIndicesDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE("FusedGdnDecode", "input dtype validation failed"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    uint32_t stateBytes = 0;
    if (mixedDtype == ge::DT_BF16 && stateDtype == ge::DT_FLOAT) {
        tilingKey = TILING_BF16_STATE_FP32;
        stateBytes = sizeof(float);
    } else if (mixedDtype == ge::DT_FLOAT16 && stateDtype == ge::DT_FLOAT) {
        tilingKey = TILING_FP16_STATE_FP32;
        stateBytes = sizeof(float);
    } else if (mixedDtype == ge::DT_BF16 && stateDtype == ge::DT_BF16) {
        tilingKey = TILING_BF16_STATE_BF16;
        stateBytes = sizeof(uint16_t);
    } else if (mixedDtype == ge::DT_FLOAT16 && stateDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_FP16_STATE_FP16;
        stateBytes = sizeof(uint16_t);
    } else {
        OP_LOGE("FusedGdnDecode", "unsupported dtype combination: mixed=%d state=%d", mixedDtype, stateDtype);
        return ge::GRAPH_FAILED;
    }
    const uint32_t alignK = Align(k, K_ALIGN_ELEMS);
    const uint32_t statePaddingBytes = (alignK - k) * stateBytes;
    OP_CHECK_IF(statePaddingBytes > MAX_DATA_COPY_PAD_BYTES,
                OP_LOGE("FusedGdnDecode", "state row padding %u bytes exceeds DataCopyPad limit %u", statePaddingBytes,
                        MAX_DATA_COPY_PAD_BYTES),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("FusedGdnDecode", "attrs is null"), return ge::GRAPH_FAILED);
    const float *scaleAttr = attrs->GetAttrPointer<float>(0);
    const float *thresholdAttr = attrs->GetAttrPointer<float>(1);
    const float scale = scaleAttr == nullptr ? 1.0f : *scaleAttr;
    const float threshold = thresholdAttr == nullptr ? 20.0f : *thresholdAttr;
    OP_CHECK_IF(!std::isfinite(scale) || !std::isfinite(threshold),
                OP_LOGE("FusedGdnDecode", "scale and softplusThreshold must be finite"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckMulU64(hv, v, "state head product") ||
                    !CheckMulU64(static_cast<uint64_t>(hv) * v, k, "state slot stride") ||
                    !CheckMulU64(batch, hv, "task count"),
                OP_LOGE("FusedGdnDecode", "shape product overflow"), return ge::GRAPH_FAILED);
    const uint64_t totalTasksU64 = static_cast<uint64_t>(batch) * hv;
    OP_CHECK_IF(totalTasksU64 > std::numeric_limits<uint32_t>::max(),
                OP_LOGE("FusedGdnDecode", "task count exceeds UINT32_MAX"), return ge::GRAPH_FAILED);

    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    OP_CHECK_IF(!GetPlatformInfo(context, aivNum, ubSize),
                OP_LOGE("FusedGdnDecode", "failed to get AIV count or UB size"), return ge::GRAPH_FAILED);
    const uint32_t totalTasks = static_cast<uint32_t>(totalTasksU64);
    const uint32_t blockDim = totalTasks < aivNum ? totalTasks : aivNum;
    const uint32_t maxTasksPerBlock = CeilDiv(totalTasks, blockDim);
    const uint32_t maxBatchesPerBlock = 1 + (maxTasksPerBlock - 1) / hv;
    OP_CHECK_IF(maxBatchesPerBlock > std::numeric_limits<uint32_t>::max() - (BLOCK_ELEMS_INT32 - 1),
                OP_LOGE("FusedGdnDecode", "state index buffer alignment overflows uint32"), return ge::GRAPH_FAILED);
    const uint32_t indexBufferElems = Align(maxBatchesPerBlock, BLOCK_ELEMS_INT32);

    const uint32_t bv = SelectBv(k, v, stateBytes, indexBufferElems, ubSize);
    OP_CHECK_IF(bv == 0, OP_LOGE("FusedGdnDecode", "available UB is insufficient"), return ge::GRAPH_FAILED);
    const uint64_t requiredUbBytes = EstimateUbBytes(k, v, bv, stateBytes, indexBufferElems);
    OP_CHECK_IF(requiredUbBytes > ubSize,
                OP_LOGE("FusedGdnDecode", "required UB %llu exceeds available UB %llu",
                        static_cast<unsigned long long>(requiredUbBytes), static_cast<unsigned long long>(ubSize)),
                return ge::GRAPH_FAILED);

    FusedGdnDecodeTilingData td{};
    td.b = batch;
    td.h = h;
    td.hv = hv;
    td.k = k;
    td.v = v;
    td.bv = bv;
    td.vTiles = CeilDiv(v, bv);
    td.stateBufferNum = bv >= v ? 1 : 2;
    td.totalTasks = totalTasks;
    td.indexBufferElems = indexBufferElems;
    td.mixedStride = mixedDim;
    td.stateSlotStride = static_cast<uint64_t>(hv) * v * k;
    td.stateHeadStride = static_cast<uint64_t>(v) * k;
    td.outBatchStride = static_cast<uint64_t>(hv) * v;
    td.scale = scale;
    td.softplusThreshold = threshold;
    const uint64_t computeBytes = EstimateComputeBytes(k, bv, stateBytes);
    OP_CHECK_IF(computeBytes > std::numeric_limits<uint32_t>::max(),
                OP_LOGE("FusedGdnDecode", "compute UB size exceeds UINT32_MAX"), return ge::GRAPH_FAILED);
    td.ubRestBytes = static_cast<uint32_t>(computeBytes);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr || rawTilingData->GetCapacity() < sizeof(FusedGdnDecodeTilingData),
                OP_LOGE("FusedGdnDecode", "raw tiling buffer is invalid"), return ge::GRAPH_FAILED);
    const errno_t ret =
        memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td, sizeof(FusedGdnDecodeTilingData));
    OP_CHECK_IF(ret != EOK, OP_LOGE("FusedGdnDecode", "copy tiling data failed, ret=%d", ret), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(FusedGdnDecodeTilingData));

    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE("FusedGdnDecode", "workspace size buffer is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = 0;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedGdnDecode(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("FusedGdnDecode", "tiling parse context is null"), return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<FusedGdnDecodeCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(compileInfo == nullptr || platformInfo == nullptr,
                OP_LOGE("FusedGdnDecode", "compile or platform info is null"), return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->aivNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE("FusedGdnDecode", "invalid platform info: aiv=%u ub=%llu", compileInfo->aivNum,
                        static_cast<unsigned long long>(compileInfo->ubSize)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedGdnDecode)
    .Tiling(FusedGdnDecodeTilingFunc)
    .TilingParse<FusedGdnDecodeCompileInfo>(TilingPrepareForFusedGdnDecode);
} // namespace optiling
