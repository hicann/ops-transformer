/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "err/ops_err.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling.h"

namespace optiling {

namespace {
constexpr int64_t MAX_N = 8;
constexpr int64_t MIN_C = 128;
constexpr int64_t MAX_C_VALUE = 100000;
constexpr int64_t C_ALIGNMENT = 128;
constexpr int64_t ITERATIONS = 20;
constexpr uint64_t TILING_KEY_ARCH35 = 3000;

constexpr uint32_t INPUT_X = 3;
constexpr uint32_t SUM_OUT = 10;
constexpr uint32_t NORM_OUT = 11;
constexpr uint32_t OUTPUT_GRAD_X = 0;
constexpr uint32_t OUTPUT_GRAD_PHI = 1;
constexpr uint32_t OUTPUT_GRAD_ALPHA = 2;
constexpr uint32_t OUTPUT_GRAD_BIAS = 3;

constexpr float DEFAULT_HC_EPS = 1e-6f;
constexpr uint32_t IMPL_MODE_HF32 = 1U;

constexpr uint32_t ALPHA_GRAD_CORE_FACTOR = 24;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t EXTRA_BUFFER_SIZE = 2 * 1024 * 1024;
constexpr uint32_t WORKSPACE_ALIGN_SIZE = 32;
constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 40 * 1024 * 1024;
constexpr int32_t SCHEDULE_MODE = 1;

constexpr uint32_t ND_BLOCK_SIZE = 128U;
constexpr uint64_t SPLIT_HALF_TILE_MIN_ROWS = 8192U;
constexpr uint64_t VEC_SPLIT_ALIGN_N4 = 8U;
constexpr uint64_t VEC_SPLIT_ALIGN_DEFAULT = 16U;

constexpr uint32_t C0_SET_SHAPE_M = 512U;
constexpr uint32_t C0_SET_SHAPE_N = 128U;
constexpr uint32_t C1_SET_SHAPE_K = 512U;
constexpr uint32_t C0_TO_V2_ROWS = 128U;
constexpr uint32_t V2_TO_C1_ROWS = 512U;

constexpr uint32_t C0_BASE_M = 128U;
constexpr uint32_t C0_BASE_N = 128U;
constexpr uint32_t C0_BASE_K = 32U;
constexpr uint32_t C0_L1_K = 64U;
constexpr uint32_t C0_DEPTH_A1 = 2U;
constexpr uint32_t C0_DEPTH_B1 = 2U;
constexpr uint32_t C0_DB_L0A = 2U;
constexpr uint32_t C0_DB_L0B = 2U;
constexpr uint32_t C0_DB_L0C = 2U;

constexpr uint32_t C1_BASE_M = 128U;
constexpr uint32_t C1_BASE_N = 128U;
constexpr uint32_t C1_BASE_K = 32U;
constexpr uint32_t C1_L1_K = 64U;
constexpr uint32_t C1_DEPTH_A1 = 2U;
constexpr uint32_t C1_DEPTH_B1 = 2U;
constexpr uint32_t C1_DB_L0A = 2U;
constexpr uint32_t C1_DB_L0B = 2U;
constexpr uint32_t C1_DB_L0C = 2U;

uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    return b == 0U ? 0U : (a + b - 1U) / b;
}

uint64_t CeilAlignU64(uint64_t a, uint64_t b)
{
    return b == 0U ? 0U : (a + b - 1U) / b * b;
}
} // namespace

bool MhcPreSinkhornBackwardArch35Tiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "Ascend950 platform information is null"),
                return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    aicNum_ = platform.GetCoreNumAic();
    aivNum_ = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize_);
    systemWorkspaceBytes_ = SYSTEM_WORKSPACE_SIZE;
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0 || ubSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "Invalid platform resources: aic=%lu, aiv=%lu, ub=%lu", aicNum_,
                        aivNum_, ubSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(l1Size_ == 0 || l0ASize_ == 0 || l0BSize_ == 0 || l0CSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "Invalid platform cache resources: l1=%lu, l0a=%lu, l0b=%lu, l0c=%lu",
                        l1Size_, l0ASize_, l0BSize_, l0CSize_),
                return ge::GRAPH_FAILED);
    blockDim_ = static_cast<uint32_t>(aicNum_);
    vecCoreNum_ = static_cast<uint32_t>(aivNum_);
    return ge::GRAPH_SUCCESS;
}

bool MhcPreSinkhornBackwardArch35Tiling::CheckTensorShape(uint32_t index, const char* name, const int64_t* dims,
                                                          size_t rank, bool logError) const
{
    auto inputShape = context_->GetInputShape(index);
    if (inputShape == nullptr) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s shape is null", name);
        }
        return false;
    }
    const auto& shape = inputShape->GetOriginShape();
    if (shape.GetDimNum() != rank) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s must have %zu dimensions, got %zu", name, rank, shape.GetDimNum());
        }
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) != dims[i]) {
            if (logError) {
                OP_LOGE(context_->GetNodeName(), "%s dim[%zu] must be %ld, got %ld", name, i, dims[i], shape.GetDim(i));
            }
            return false;
        }
    }
    return true;
}

bool MhcPreSinkhornBackwardArch35Tiling::CheckOutputShape(uint32_t index, const char* name, const int64_t* dims,
                                                          size_t rank, bool logError) const
{
    auto outputShape = context_->GetOutputShape(index);
    if (outputShape == nullptr) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s output shape is null", name);
        }
        return false;
    }
    const auto& shape = outputShape->GetOriginShape();
    if (shape.GetDimNum() != rank) {
        if (logError) {
            OP_LOGE(context_->GetNodeName(), "%s must have %zu dimensions, got %zu", name, rank, shape.GetDimNum());
        }
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) != dims[i]) {
            if (logError) {
                OP_LOGE(context_->GetNodeName(), "%s dim[%zu] must be %ld, got %ld", name, i, dims[i], shape.GetDim(i));
            }
            return false;
        }
    }
    return true;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::CheckDataTypes() const
{
    auto xDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "x descriptor is null"), return ge::GRAPH_FAILED);
    ge::DataType xType = xDesc->GetDataType();
    OP_CHECK_IF(xType != ge::DT_BF16 && xType != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "x must be BF16 or FP16, got dtype %d", static_cast<int32_t>(xType)),
                return ge::GRAPH_FAILED);

    for (uint32_t index = 0; index <= NORM_OUT; ++index) {
        auto desc = context_->GetInputDesc(index);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "input %u descriptor is null", index),
                    return ge::GRAPH_FAILED);
        ge::DataType expected = index == 0 || index == INPUT_X ? xType : ge::DT_FLOAT;
        OP_CHECK_IF(desc->GetDataType() != expected,
                    OP_LOGE(context_->GetNodeName(), "input %u dtype must be %d, got %d", index,
                            static_cast<int32_t>(expected), static_cast<int32_t>(desc->GetDataType())),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t index = OUTPUT_GRAD_X; index <= OUTPUT_GRAD_BIAS; ++index) {
        auto desc = context_->GetOutputDesc(index);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "output %u descriptor is null", index),
                    return ge::GRAPH_FAILED);
        ge::DataType expected = index == OUTPUT_GRAD_X ? xType : ge::DT_FLOAT;
        OP_CHECK_IF(desc->GetDataType() != expected,
                    OP_LOGE(context_->GetNodeName(), "output %u dtype must be %d, got %d", index,
                            static_cast<int32_t>(expected), static_cast<int32_t>(desc->GetDataType())),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetShapeAttrsInfo()
{
    tilingData_ = context_->GetTilingData<MhcPreSinkhornBackwardArch35TilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "tiling data is null"),
                return ge::GRAPH_FAILED);
    auto xShapeInfo = context_->GetInputShape(INPUT_X);
    OP_CHECK_IF(xShapeInfo == nullptr, OP_LOGE(context_->GetNodeName(), "x shape is null"), return ge::GRAPH_FAILED);
    const auto& xShape = xShapeInfo->GetOriginShape();
    xRank_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(xRank_ != 3 && xRank_ != 4,
                OP_LOGE(context_->GetNodeName(), "x must be 3D [T,N,C] or 4D [B,S,N,C], got rank %ld", xRank_),
                return ge::GRAPH_FAILED);

    const int64_t batchRank = xRank_ - 2;
    int64_t batchDims[2] = {xShape.GetDim(0), xRank_ == 4 ? xShape.GetDim(1) : 1};
    OP_CHECK_IF(batchDims[0] < 0 || batchDims[1] < 0,
                OP_LOGE(context_->GetNodeName(), "batch dimensions must be non-negative, got %ld and %ld", batchDims[0],
                        batchDims[1]),
                return ge::GRAPH_FAILED);
    N_ = static_cast<uint64_t>(xShape.GetDim(xRank_ - 2));
    D_ = static_cast<uint64_t>(xShape.GetDim(xRank_ - 1));
    totalLength_ = static_cast<uint64_t>(xRank_ == 4 ? batchDims[0] * batchDims[1] : batchDims[0]);
    OP_CHECK_IF(N_ == 0 || N_ > MAX_N,
                OP_LOGE(context_->GetNodeName(), "n must be > 0 and <= %ld, but got %lu", MAX_N, N_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(D_ < MIN_C || D_ >= MAX_C_VALUE || D_ % C_ALIGNMENT != 0,
                OP_LOGE(context_->GetNodeName(), "c must be >= %ld, < %ld and divisible by %ld, but got %lu", MIN_C,
                        MAX_C_VALUE, C_ALIGNMENT, D_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDataTypes() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornBackward dtype validation failed"),
                return ge::GRAPH_FAILED);

    fusionSize_ = (2 * N_) + (N_ * N_);
    const int64_t n = static_cast<int64_t>(N_);
    const int64_t d = static_cast<int64_t>(D_);
    const int64_t fusionSize = static_cast<int64_t>(fusionSize_);
    const int64_t nD = n * d;
    int64_t gradHin[3] = {batchDims[0], batchDims[1], d};
    int64_t gradHPost[3] = {batchDims[0], batchDims[1], n};
    int64_t gradHRes[4] = {batchDims[0], batchDims[1], n, n};
    int64_t gradHResFlat[3] = {batchDims[0], batchDims[1], n * n};
    int64_t xDims[4] = {batchDims[0], batchDims[1], n, d};
    int64_t phiDims[2] = {fusionSize, nD};
    int64_t alphaDims[1] = {3};
    int64_t biasDims[1] = {fusionSize};
    int64_t hPre[3] = {batchDims[0], batchDims[1], n};
    int64_t qDims[3] = {batchDims[0], batchDims[1], fusionSize};
    int64_t invDims[3] = {batchDims[0], batchDims[1], 1};
    int64_t sumDims[4] = {2 * ITERATIONS, batchDims[0], batchDims[1], n};
    int64_t normDims[5] = {2 * ITERATIONS, batchDims[0], batchDims[1], n, n};
    if (xRank_ == 3) {
        gradHin[1] = d;
        gradHPost[1] = n;
        gradHRes[1] = n;
        gradHRes[2] = n;
        gradHResFlat[1] = n * n;
        xDims[1] = n;
        xDims[2] = d;
        hPre[1] = n;
        qDims[1] = fusionSize;
        invDims[1] = 1;
        sumDims[1] = batchDims[0];
        sumDims[2] = n;
        normDims[1] = batchDims[0];
        normDims[2] = n;
        normDims[3] = n;
    }

    const size_t dataRank = static_cast<size_t>(batchRank + 1);
    const size_t xExpectedRank = static_cast<size_t>(xRank_);
    bool gradHResOk = CheckTensorShape(2, "grad_h_res", gradHRes, xExpectedRank, false) ||
                      CheckTensorShape(2, "grad_h_res", gradHResFlat, dataRank, false);
    if (!gradHResOk) {
        auto gradHResShape = context_->GetInputShape(2);
        OP_LOGE(context_->GetNodeName(),
                "grad_h_res must be [.., %ld, %ld] (rank %zu) or [.., %ld] (rank %zu), got rank %zu", n, n,
                xExpectedRank, n * n, dataRank,
                gradHResShape == nullptr ? static_cast<size_t>(0) : gradHResShape->GetOriginShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(!CheckTensorShape(0, "grad_hin", gradHin, dataRank) ||
                    !CheckTensorShape(1, "grad_h_post", gradHPost, dataRank) ||
                    !CheckTensorShape(INPUT_X, "x", xDims, xExpectedRank) || !CheckTensorShape(4, "phi", phiDims, 2) ||
                    !CheckTensorShape(5, "alpha", alphaDims, 1) || !CheckTensorShape(6, "bias", biasDims, 1) ||
                    !CheckTensorShape(7, "h_pre", hPre, dataRank) ||
                    !CheckTensorShape(8, "hc_before_norm", qDims, dataRank) ||
                    !CheckTensorShape(9, "inv_rms", invDims, dataRank) ||
                    !CheckTensorShape(SUM_OUT, "sum_out", sumDims, dataRank + 1) ||
                    !CheckTensorShape(NORM_OUT, "norm_out", normDims, dataRank + 2),
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornBackward input shape validation failed"),
                return ge::GRAPH_FAILED);

    int64_t gradXDims[4] = {batchDims[0], batchDims[1], n, d};
    int64_t gradPhiDims[2] = {fusionSize, nD};
    int64_t gradAlphaDims[1] = {3};
    int64_t gradBiasDims[1] = {fusionSize};
    if (xRank_ == 3) {
        gradXDims[1] = n;
        gradXDims[2] = d;
    }
    OP_CHECK_IF(!CheckOutputShape(OUTPUT_GRAD_X, "grad_x", gradXDims, xExpectedRank) ||
                    !CheckOutputShape(OUTPUT_GRAD_PHI, "grad_phi", gradPhiDims, 2) ||
                    !CheckOutputShape(OUTPUT_GRAD_ALPHA, "grad_alpha", gradAlphaDims, 1) ||
                    !CheckOutputShape(OUTPUT_GRAD_BIAS, "grad_bias", gradBiasDims, 1),
                OP_LOGE(context_->GetNodeName(), "MhcPreSinkhornBackward output shape validation failed"),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attributes are null"), return ge::GRAPH_FAILED);
    auto hcEpsPtr = attrs->GetAttrPointer<float>(0);
    hcEps_ = hcEpsPtr == nullptr ? DEFAULT_HC_EPS : *hcEpsPtr;
    OP_CHECK_IF(!std::isfinite(hcEps_) || hcEps_ <= 0.0f,
                OP_LOGE(context_->GetNodeName(), "hc_eps must be finite and positive"), return ge::GRAPH_FAILED);

    // C0/C1 的 FP32 输入默认按 HF32 计算。
    implMode_ = IMPL_MODE_HF32;
    tilingData_->totalLength = totalLength_;
    tilingData_->nD = N_ * D_;
    tilingData_->N = N_;
    tilingData_->D = D_;
    tilingData_->fusionSize = fusionSize_;
    tilingData_->hcEps = hcEps_;
    tilingData_->implMode = implMode_;
    tilingData_->iterTimes = static_cast<uint64_t>(ITERATIONS);
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35Tiling::SetMmConfig()
{
    tilingData_->mmConfigC0.baseM = C0_BASE_M;
    tilingData_->mmConfigC0.baseN = C0_BASE_N;
    tilingData_->mmConfigC0.baseK = C0_BASE_K;
    tilingData_->mmConfigC0.l1K = C0_L1_K;
    tilingData_->mmConfigC0.depthA1 = C0_DEPTH_A1;
    tilingData_->mmConfigC0.depthB1 = C0_DEPTH_B1;
    tilingData_->mmConfigC0.dbL0A = C0_DB_L0A;
    tilingData_->mmConfigC0.dbL0B = C0_DB_L0B;
    tilingData_->mmConfigC0.dbL0C = C0_DB_L0C;

    tilingData_->mmConfigC1.baseM = C1_BASE_M;
    tilingData_->mmConfigC1.baseN = C1_BASE_N;
    tilingData_->mmConfigC1.baseK = C1_BASE_K;
    tilingData_->mmConfigC1.l1K = C1_L1_K;
    tilingData_->mmConfigC1.depthA1 = C1_DEPTH_A1;
    tilingData_->mmConfigC1.depthB1 = C1_DEPTH_B1;
    tilingData_->mmConfigC1.dbL0A = C1_DB_L0A;
    tilingData_->mmConfigC1.dbL0B = C1_DB_L0B;
    tilingData_->mmConfigC1.dbL0C = C1_DB_L0C;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::ValidateMmConfig()
{
    MhcPreSinkhornBackwardMMConfig& mmConfigC0 = tilingData_->mmConfigC0;
    MhcPreSinkhornBackwardMMConfig& mmConfigC1 = tilingData_->mmConfigC1;
    constexpr uint64_t elementSize = sizeof(float);

    uint64_t c0A1SingleBufferBytes = static_cast<uint64_t>(mmConfigC0.baseM) * mmConfigC0.l1K * elementSize;
    uint64_t c0B1SingleBufferBytes = static_cast<uint64_t>(mmConfigC0.baseN) * mmConfigC0.l1K * elementSize;
    uint64_t c1A1SingleBufferBytes = static_cast<uint64_t>(mmConfigC1.baseM) * mmConfigC1.l1K * elementSize;
    uint64_t c1B1SingleBufferBytes = static_cast<uint64_t>(mmConfigC1.baseN) * mmConfigC1.l1K * elementSize;
    uint64_t l1SingleBufferBytes = std::max(std::max(c0A1SingleBufferBytes, c0B1SingleBufferBytes),
                                            std::max(c1A1SingleBufferBytes, c1B1SingleBufferBytes));

    uint64_t c0A2SingleBufferBytes = static_cast<uint64_t>(mmConfigC0.baseM) * mmConfigC0.baseK * elementSize;
    uint64_t c0B2SingleBufferBytes = static_cast<uint64_t>(mmConfigC0.baseN) * mmConfigC0.baseK * elementSize;
    uint64_t c1A2SingleBufferBytes = static_cast<uint64_t>(mmConfigC1.baseM) * mmConfigC1.baseK * elementSize;
    uint64_t c1B2SingleBufferBytes = static_cast<uint64_t>(mmConfigC1.baseN) * mmConfigC1.baseK * elementSize;
    uint64_t l0ABSingleBufferBytes = std::max(std::max(c0A2SingleBufferBytes, c0B2SingleBufferBytes),
                                              std::max(c1A2SingleBufferBytes, c1B2SingleBufferBytes));

    uint64_t c0L0CSingleBufferBytes = static_cast<uint64_t>(mmConfigC0.baseM) * mmConfigC0.baseN * elementSize;
    uint64_t c1L0CSingleBufferBytes = static_cast<uint64_t>(mmConfigC1.baseM) * mmConfigC1.baseN * elementSize;
    uint64_t l0CSingleBufferBytes = std::max(c0L0CSingleBufferBytes, c1L0CSingleBufferBytes);
    uint64_t depthA1 = std::max(mmConfigC0.depthA1, mmConfigC1.depthA1);
    uint64_t depthB1 = std::max(mmConfigC0.depthB1, mmConfigC1.depthB1);
    uint64_t dbL0 =
        std::max(std::max(mmConfigC0.dbL0A, mmConfigC1.dbL0A), std::max(mmConfigC0.dbL0B, mmConfigC1.dbL0B));
    uint64_t dbL0C = std::max(mmConfigC0.dbL0C, mmConfigC1.dbL0C);
    uint64_t l1UsedBytes = (depthA1 + depthB1) * l1SingleBufferBytes;
    // 先校验 MatMul 必需缓冲区，剩余 L1 用于可选缓存。
    uint64_t l0ABUsedBytes = dbL0 * l0ABSingleBufferBytes;
    uint64_t l0CUsedBytes = dbL0C * l0CSingleBufferBytes;
    OP_CHECK_IF(l1UsedBytes > l1Size_ || l0ABUsedBytes > l0ASize_ || l0ABUsedBytes > l0BSize_ ||
                    l0CUsedBytes > l0CSize_ || l1UsedBytes > UINT32_MAX || l0ABUsedBytes > UINT32_MAX ||
                    l0CUsedBytes > UINT32_MAX,
                OP_LOGE(context_->GetNodeName(),
                        "MM config exceeds platform memory: L1=%lu/%lu, L0A/B=%lu/%lu/%lu, L0C=%lu/%lu", l1UsedBytes,
                        l1Size_, l0ABUsedBytes, l0ASize_, l0BSize_, l0CUsedBytes, l0CSize_),
                return ge::GRAPH_FAILED);
    maxBufferDepth_ = static_cast<uint32_t>(std::max(std::max(depthA1, depthB1), std::max(dbL0, dbL0C)));
    l1UsedBytes_ = static_cast<uint32_t>(l1UsedBytes);
    l1SingleBufferElems_ = static_cast<uint32_t>(l1SingleBufferBytes / elementSize);
    l1BOffsetElems_ = static_cast<uint32_t>(depthA1 * l1SingleBufferElems_);
    l0ABUsedBytes_ = static_cast<uint32_t>(l0ABUsedBytes);
    l0ABSingleBufferElems_ = static_cast<uint32_t>(l0ABSingleBufferBytes / elementSize);
    l0CUsedBytes_ = static_cast<uint32_t>(l0CUsedBytes);
    l0CSingleBufferElems_ = static_cast<uint32_t>(l0CSingleBufferBytes / elementSize);

    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPreSinkhornBackwardArch35Tiling::CalculateWorkspaceSize(uint64_t totalLength, uint64_t fusionSize,
                                                                    uint64_t cubeCoreNum, uint64_t vecCoreNum,
                                                                    uint64_t elementSize)
{
    uint64_t v1Elements = totalLength * fusionSize +            // h_mix_grad
                          ALPHA_GRAD_CORE_FACTOR * vecCoreNum + // alpha_grad
                          vecCoreNum * fusionSize +             // bias_grad
                          totalLength;                          // inv_rms_grad

    uint64_t v2Elements = C0_SET_SHAPE_M * C0_SET_SHAPE_N * cubeCoreNum * BUFFER_NUM + // x_rs_grad_mm
                          C0_SET_SHAPE_M * C0_SET_SHAPE_N * cubeCoreNum * BUFFER_NUM + // x_rs
                          EXTRA_BUFFER_SIZE;
    uint64_t totalElements =
        ((v1Elements + WORKSPACE_ALIGN_SIZE - 1) / WORKSPACE_ALIGN_SIZE * WORKSPACE_ALIGN_SIZE) + v2Elements;
    return totalElements * elementSize;
}

void MhcPreSinkhornBackwardArch35Tiling::SetSinkhornTiling()
{
    using namespace MhcPreSinkhornBackwardSinkhornNs;
    SinkhornTilePlan& plan = tilingData_->sinkhornPlan;
    plan.nSize = static_cast<uint32_t>(N_);
    plan.matrixElements = plan.nSize * plan.nSize;
    plan.path = SinkhornPath::COMPACT;
    if (N_ == 4U) {
        plan.path = tilingData_->smallM != 0U ? SinkhornPath::N4_COMPACT : SinkhornPath::N4_ROW_PACKED;
    } else if (N_ == 6U) {
        plan.path = SinkhornPath::N6_ROW_PACKED;
    } else if (N_ == 8U) {
        plan.path = SinkhornPath::N8_INTERLEAVED;
    }
    plan.matricesPerVector = MatricesPerVector(plan.nSize);
    plan.matricesPerTile = MatricesPerTile(plan.nSize, plan.path == SinkhornPath::N4_ROW_PACKED);
    plan.sumElementsPerIteration = AlignFloatBlock(plan.matricesPerTile * plan.nSize);
    const uint32_t norms = AlignFloatBlock(plan.matricesPerTile * plan.matrixElements);
    plan.iterationDataOffset = norms + VECTOR_ELEMENTS;
    plan.iterationStride = 2U * plan.sumElementsPerIteration + norms;
    plan.powerOfTwo = (plan.nSize & (plan.nSize - 1U)) == 0U;
    plan.alignedVectors = (plan.matricesPerVector * plan.matrixElements) % FLOAT_BLOCK_ELEMENTS == 0U &&
                          (plan.matricesPerTile * plan.matrixElements) % FLOAT_BLOCK_ELEMENTS == 0U;
    plan.logN = plan.nSize == 2U ? 1U : plan.nSize == 4U ? 2U : plan.nSize == 8U ? 3U : 0U;
}

void MhcPreSinkhornBackwardArch35Tiling::SetVecCoreSplit()
{
    uint64_t vecDealBSPeCore = totalLength_ / vecCoreNum_;
    uint32_t usedVecCoreNum = 0;
    if (N_ == 8U) {
        // N=8 按 8 行对齐，提高 AIV 利用率。
        vecDealBSPeCore = CeilAlignU64(CeilDivU64(totalLength_, vecCoreNum_), VEC_SPLIT_ALIGN_N4);
        usedVecCoreNum = static_cast<uint32_t>(CeilDivU64(totalLength_, vecDealBSPeCore));
    } else if (vecDealBSPeCore == 0U) {
        if (N_ == 4U) {
            // 小 M 按 8 行切块，尽量使用多个 AIV。
            vecDealBSPeCore = std::min<uint64_t>(totalLength_, VEC_SPLIT_ALIGN_N4);
            usedVecCoreNum = static_cast<uint32_t>(CeilDivU64(totalLength_, vecDealBSPeCore));
        } else {
            vecDealBSPeCore = totalLength_;
            usedVecCoreNum = 1U;
        }
    } else {
        // N=4 按 8 行对齐以增加参与核数；N=6 保留 16 行粒度，维持 grad_alpha 累加顺序。
        const uint64_t rowAlign = N_ == 4U ? VEC_SPLIT_ALIGN_N4 : VEC_SPLIT_ALIGN_DEFAULT;
        vecDealBSPeCore = CeilAlignU64(vecDealBSPeCore, rowAlign);
        uint64_t used = CeilDivU64(totalLength_, vecDealBSPeCore);
        if (used > vecCoreNum_) {
            // 超出物理核数时重新均分，避免末核长尾。
            vecDealBSPeCore = CeilDivU64(totalLength_, vecCoreNum_);
            used = CeilDivU64(totalLength_, vecDealBSPeCore);
        }
        usedVecCoreNum = static_cast<uint32_t>(used);
    }
    tilingData_->vecDealBSPeCore = vecDealBSPeCore;
    tilingData_->usedVecCoreNum = usedVecCoreNum;
}

void MhcPreSinkhornBackwardArch35Tiling::SetCubeCoreSplit()
{
    const uint64_t nD = tilingData_->nD;
    uint32_t cubeDealnDPeCore = static_cast<uint32_t>(nD / blockDim_);
    uint32_t v1UsedCubeCoreNum = 0;
    const bool splitHalfTile = totalLength_ >= SPLIT_HALF_TILE_MIN_ROWS && D_ % (ND_BLOCK_SIZE / 2U) == 0U;
    if (splitHalfTile) {
        // 大 M 按 64 列切分，增加参与长归约的 AIC 数量。
        cubeDealnDPeCore = static_cast<uint32_t>(CeilAlignU64(CeilDivU64(nD, blockDim_), ND_BLOCK_SIZE / 2U));
        v1UsedCubeCoreNum = static_cast<uint32_t>(CeilDivU64(nD, cubeDealnDPeCore));
    } else if (cubeDealnDPeCore == 0U) {
        cubeDealnDPeCore = static_cast<uint32_t>(nD);
        v1UsedCubeCoreNum = 1U;
    } else {
        cubeDealnDPeCore = static_cast<uint32_t>(CeilAlignU64(cubeDealnDPeCore, ND_BLOCK_SIZE));
        v1UsedCubeCoreNum = static_cast<uint32_t>(CeilDivU64(nD, cubeDealnDPeCore));
    }
    tilingData_->cubeDealnDPeCore = cubeDealnDPeCore;
    tilingData_->v1UsedCubeCoreNum = v1UsedCubeCoreNum;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::TilingProcess()
{
    tilingData_->smallM = totalLength_ > 0U && totalLength_ <= C0_SET_SHAPE_M ? 1U : 0U;
    SetSinkhornTiling();
    SetVecCoreSplit();
    SetCubeCoreSplit();
    // 按剩余 L1 规划缓存；容量不足时回退到普通加载。
    tilingData_->cubeCachePlan = BuildCubeCachePlan(totalLength_, fusionSize_, tilingData_->cubeDealnDPeCore,
                                                    C0_SET_SHAPE_M, C0_SET_SHAPE_N, l1Size_ - l1UsedBytes_);
    uint64_t userWorkspaceSize =
        CalculateWorkspaceSize(totalLength_, fusionSize_, blockDim_, vecCoreNum_, sizeof(float));
    workspaceSize_ = userWorkspaceSize + systemWorkspaceBytes_;
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35Tiling::SetCommonTilingParams()
{
    tilingData_->coreNum = blockDim_;
    tilingData_->vecCoreNum = vecCoreNum_;
    tilingData_->totalLength = totalLength_;
    tilingData_->nD = N_ * D_;
    tilingData_->fusionSize = fusionSize_;
    tilingData_->N = N_;
    tilingData_->D = D_;
    tilingData_->hcEps = hcEps_;
    tilingData_->implMode = implMode_;
    tilingData_->maxBufferDepth = maxBufferDepth_;
    tilingData_->l1UsedBytes = l1UsedBytes_;
    tilingData_->l1BOffsetElems = l1BOffsetElems_;
    tilingData_->l1SingleBufferElems = l1SingleBufferElems_;
    tilingData_->l0ABUsedBytes = l0ABUsedBytes_;
    tilingData_->l0ABSingleBufferElems = l0ABSingleBufferElems_;
    tilingData_->l0CUsedBytes = l0CUsedBytes_;
    tilingData_->l0CSingleBufferElems = l0CSingleBufferElems_;
    tilingData_->c0MBlock = C0_SET_SHAPE_M;
    tilingData_->c0NBlock = C0_SET_SHAPE_N;
    tilingData_->c1KBlock = C1_SET_SHAPE_K;
    tilingData_->c0ToV2Rows = C0_TO_V2_ROWS;
    tilingData_->v2ToC1Rows = V2_TO_C1_ROWS;
    tilingData_->iterTimes = static_cast<uint64_t>(ITERATIONS);
}

void MhcPreSinkhornBackwardArch35Tiling::FillTilingData()
{
    SetCommonTilingParams();
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::DoOpTiling()
{
    SetMmConfig();
    if (ValidateMmConfig() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcPreSinkhornBackwardArch35Tiling::GetTilingKey() const
{
    return TILING_KEY_ARCH35;
}

ge::graphStatus MhcPreSinkhornBackwardArch35Tiling::PostTiling()
{
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "tiling data is null"),
                return ge::GRAPH_FAILED);
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_->coreNum);
    context_->SetScheduleMode(SCHEDULE_MODE);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size output is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35Tiling::DumpTilingInfo()
{
    if (tilingData_ == nullptr) {
        return;
    }
    OP_LOGI(context_->GetNodeName(),
            "MhcPreSinkhornBackward arch35: coreNum=%u, vecCoreNum=%u, totalLength=%lu, N=%lu, D=%lu, nD=%lu, "
            "fusionSize=%lu, hcEps=%f, implMode=%u, smallM=%u, iterTimes=%lu",
            tilingData_->coreNum, tilingData_->vecCoreNum, tilingData_->totalLength, tilingData_->N, tilingData_->D,
            tilingData_->nD, tilingData_->fusionSize, tilingData_->hcEps, tilingData_->implMode, tilingData_->smallM,
            tilingData_->iterTimes);
    OP_LOGI(context_->GetNodeName(),
            "MM config: C0=[%u,%u,%u,L1K=%u], C1=[%u,%u,%u,L1K=%u], depth=[%u,%u,%u,%u,%u], "
            "block=[%u,%u,%u], sync=[%u,%u], maxBufferDepth=%u",
            tilingData_->mmConfigC0.baseM, tilingData_->mmConfigC0.baseN, tilingData_->mmConfigC0.baseK,
            tilingData_->mmConfigC0.l1K, tilingData_->mmConfigC1.baseM, tilingData_->mmConfigC1.baseN,
            tilingData_->mmConfigC1.baseK, tilingData_->mmConfigC1.l1K, tilingData_->mmConfigC0.depthA1,
            tilingData_->mmConfigC0.depthB1, tilingData_->mmConfigC0.dbL0A, tilingData_->mmConfigC0.dbL0B,
            tilingData_->mmConfigC0.dbL0C, tilingData_->c0MBlock, tilingData_->c0NBlock, tilingData_->c1KBlock,
            tilingData_->c0ToV2Rows, tilingData_->v2ToC1Rows, tilingData_->maxBufferDepth);
    const auto& plan = tilingData_->sinkhornPlan;
    OP_LOGI(context_->GetNodeName(), "Sinkhorn: path=%u, matricesPerTile=%u, stateOffset=%u, iterationStride=%u",
            static_cast<uint32_t>(plan.path), plan.matricesPerTile, plan.iterationDataOffset, plan.iterationStride);
    OP_LOGI(context_->GetNodeName(), "L1 used=%u, l1BOffset=%u, l1Slot=%u, L0AB=[%u,%u], L0C=[%u,%u], workspace=%lu",
            tilingData_->l1UsedBytes, tilingData_->l1BOffsetElems, tilingData_->l1SingleBufferElems,
            tilingData_->l0ABUsedBytes, tilingData_->l0ABSingleBufferElems, tilingData_->l0CUsedBytes,
            tilingData_->l0CSingleBufferElems, workspaceSize_);
    const auto& cache = tilingData_->cubeCachePlan;
    OP_LOGI(context_->GetNodeName(),
            "Cube cache: hMixGradMode=%u, cachePhi=%u, rows=%u/%u, slots=%u, slotElements=%u, phiColumns=%u, bytes=%lu",
            static_cast<uint32_t>(cache.hMixGradMode), cache.cachePhi, cache.validRows, cache.rows, cache.slotCount,
            cache.slotElements, cache.phiColumns, static_cast<uint64_t>(cache.bufferElements) * sizeof(float));
}

REGISTER_OPS_TILING_TEMPLATE(MhcPreSinkhornBackward, MhcPreSinkhornBackwardArch35Tiling, 0);
} // namespace optiling
