/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mega_mhc_tiling.h"
#include "mega_mhc_tiling_utils.h"
#include "../../../op_kernel/arch35/mega_mhc_tiling_key.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr float kDefaultEpsilon = 1.0e-6f;
constexpr float kDefaultHcPostScale = 2.0f;
constexpr uint32_t kHeadCount = 4;
constexpr uint32_t kProjectionQ = MegaMhc::kProjectionColumns;
constexpr uint32_t kUbLimitBytes = 248 * 1024;
constexpr uint32_t kWorkspaceAlign = 512;
constexpr uint64_t kRuntimeWorkspaceBytes = 20ULL * 1024ULL * 1024ULL;
constexpr uint32_t kSlotCount = MegaMhc::kProjectionSlots;
constexpr uint32_t kPrefetchInputSlotCount = 3;
constexpr uint32_t kFp32RegisterElements = 64;
constexpr uint32_t kMmadBlockSize = 16U;
constexpr uint32_t kMmadBarrierRepeatLimit = 10U;

// Positions in the public operator prototype; each position is fixed explicitly.
constexpr size_t kPostMixInputIndex = 2U;
constexpr size_t kFnInputIndex = 3U;
constexpr size_t kMixScalesInputIndex = 4U;
constexpr size_t kMixBasesInputIndex = 5U;
constexpr size_t kRmsNorm1WeightInputIndex = 6U;
constexpr size_t kCombResMixInputIndex = 7U;
constexpr size_t kShiftedPrevMixInputIndex = 8U;
constexpr size_t kRmsNorm0WeightInputIndex = 9U;
constexpr size_t kRmsNormMidWeightInputIndex = 10U;
constexpr size_t kFirstOptionalInputIndex = 7U;

constexpr uint32_t kNewCombResMixOutputIndex = 2U;
constexpr uint32_t kNewPrevMixOutputIndex = 3U;
constexpr uint32_t kYOutputIndex = 4U;
constexpr uint32_t kYQuantOutputIndex = 5U;
constexpr uint32_t kYScaleOutputIndex = 6U;
constexpr uint32_t kReservedOutputCount = 3U;
constexpr uint32_t kActiveOutputCount = 4U;

constexpr size_t kHcEpsAttrIndex = 2U;
constexpr size_t kOpImplModeAttrIndex = 3U;
constexpr size_t kHcMultAttrIndex = 4U;
constexpr size_t kPostScaleAttrIndex = 5U;
constexpr size_t kSinkhornEpsAttrIndex = 6U;
constexpr size_t kSinkhornItersAttrIndex = 7U;
constexpr size_t kRmsNormScaleAttrIndex = 8U;
constexpr size_t kOutputModeAttrIndex = 9U;

constexpr uint32_t kUnbatchedXRank = 2U;
constexpr size_t kMatrixRank = 2U;
constexpr size_t kTrailingDimensionCount = 2U;
constexpr uint32_t kBatchedXRank = 3U;
constexpr uint32_t kDAlignment = 16U;
constexpr uint32_t kMaximumD = 16384U;
constexpr uint32_t kMinimumAivCount = 2U;
constexpr uint32_t kAivPerAic = 2U;
constexpr uint32_t kMinimumWaveCorePairs = 4U;
constexpr uint32_t kMinimumWaveCount = 2U;
constexpr uint32_t kMinimumWaveKLoops = 2U;
constexpr uint32_t kWaveCandidateCount = 2U;
constexpr uint32_t kWithMidHf32BsRows = 1024U;
constexpr uint32_t kOtherBsRows = 1536U;
constexpr uint32_t kWithMidPrefetchMinRows = 512U;
constexpr uint32_t kWavePrefixBytesPerAiv = 128U;
constexpr uint32_t kAlphaElements = 3U;
constexpr uint32_t kRequiredHcMultiplier = 4U;
constexpr uint32_t kTilingTemplatePriority = 1000U;
constexpr uint32_t kCachedGammaResidualSlots = 2U;
constexpr uint32_t kL1HalfDivisor = 2U;
constexpr uint32_t kMinimumRowsPerAiv = 2U;
constexpr uint32_t kReductionAlignmentOffset = 7U;
constexpr uint32_t kEpiloguePrefetchInput = 2U;
constexpr uint32_t kMultipleMidResidualSlots = 2U;
constexpr uint32_t kResidentBSlots = 2U;
constexpr uint32_t kRmsScalarScratchBytes = 8U;
constexpr uint32_t kInputStorageBytes = 2U;

constexpr const char* kInputNames[] = {"x",
                                       "residual",
                                       "post_mix",
                                       "fn",
                                       "mix_scales",
                                       "mix_bases",
                                       "rms_norm_1_weight",
                                       "comb_res_mix",
                                       "shifted_prev_mix",
                                       "rms_norm_0_weight",
                                       "rms_norm_mid_weight"};
constexpr const char* kOutputNames[] = {"new_residual", "new_post_mix", "new_comb_res_mix", "new_prev_mix", "y",
                                        "y_quant",      "y_scale"};

/** Inputs absent from the public call may omit their descriptor and shape. */
bool MayOmitInput(size_t index)
{
    return index == kShiftedPrevMixInputIndex || index == kRmsNorm0WeightInputIndex ||
           index == kRmsNormMidWeightInputIndex;
}

/** Check dimensions only after establishing the tensor rank. */
bool HasVectorShape(const gert::Shape& shape, int64_t width)
{
    return shape.GetDimNum() == 1U && shape.GetDim(0) == width;
}

bool HasMatrixShape(const gert::Shape& shape, int64_t rows, int64_t columns)
{
    return shape.GetDimNum() == kMatrixRank && shape.GetDim(0) == rows && shape.GetDim(1) == columns;
}

bool HasTrailingShape(const gert::Shape& shape, size_t expectedRank, int64_t penultimate, int64_t last)
{
    return expectedRank >= kTrailingDimensionCount && shape.GetDimNum() == expectedRank &&
           shape.GetDim(expectedRank - kTrailingDimensionCount) == penultimate &&
           shape.GetDim(expectedRank - 1U) == last;
}

/** Format storage dimensions for shape diagnostics without accessing invalid axes. */
std::string ShapeToString(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
        result += (axis == 0 ? "" : ", ") + std::to_string(shape.GetDim(axis));
    }
    return result + "]";
}

/** Ceiling division with overflow protection. */
uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

/** Align a workspace region while rejecting uint64 overflow. */
bool AlignWorkspace(uint64_t value, uint64_t alignment, uint64_t& aligned)
{
    if (alignment == 0) {
        return false;
    }
    const uint64_t blocks = CeilDiv(value, alignment);
    if (blocks > std::numeric_limits<uint64_t>::max() / alignment) {
        return false;
    }
    aligned = blocks * alignment;
    return true;
}

/** Multiply workspace extents while rejecting uint64 overflow. */
bool CheckedBytes(uint64_t first, uint64_t second, uint64_t third, uint64_t elementBytes, uint64_t& bytes)
{
    if (first != 0 && second > std::numeric_limits<uint64_t>::max() / first) {
        return false;
    }
    const uint64_t firstTwo = first * second;
    if (firstTwo != 0 && third > std::numeric_limits<uint64_t>::max() / firstTwo) {
        return false;
    }
    const uint64_t elements = firstTwo * third;
    if (elements != 0 && elementBytes > std::numeric_limits<uint64_t>::max() / elements) {
        return false;
    }
    bytes = elements * elementBytes;
    return true;
}

/** Keep schedule eligibility separate from the geometry and UB checks for a wave candidate. */
bool CanTryWaveTiling(const mega_mhc::CubeTilingParams& request, bool hasMid, bool tiledRows)
{
    return !tiledRows && !hasMid && request.schedule == mega_mhc::ScheduleMode::MK &&
           request.mode == mega_mhc::ProjectionMode::FP32 &&
           std::min(request.aicCount, request.aivCount / kAivPerAic) >= kMinimumWaveCorePairs;
}

/** Reject wave geometries that cannot overlap two producer slots or exceed baseline work or UB. */
bool CanUseWaveCandidate(const mega_mhc::CubeTilingInfo& baseline, const mega_mhc::CubeTilingInfo& wave, uint32_t d,
                         uint64_t availableUb)
{
    if (wave.kGrid == 0U || wave.mTile % (uint64_t(kAivPerAic) * wave.kGrid) != 0U ||
        wave.mWaveCount < kMinimumWaveCount || wave.lastKCoreL1FullLoops < kMinimumWaveKLoops ||
        wave.mTile < uint64_t(kAivPerAic) * kSlotCount * wave.kGrid) {
        return false;
    }
    const uint64_t waveWork = uint64_t(wave.mTile) * wave.mWaveCount * wave.splitK;
    const uint64_t baselineWork = uint64_t(baseline.mTile) * baseline.mWaveCount * baseline.splitK;
    return waveWork <= baselineWork && mega_mhc::CalcProducerUbSize(d, false, true) +
                                               mega_mhc::CalcProjectionUbSize(wave.l1K, wave.mTile / kAivPerAic) <=
                                           availableUb;
}

/** Prefer two prefetched residual slots when a complete Mid gamma fits. */
uint32_t SelectCachedGammaResidualSlots(uint32_t d, uint64_t availableUb)
{
    if (mega_mhc::CalcCachedGammaProducerUbSize(d, kCachedGammaResidualSlots) <= availableUb) {
        return kCachedGammaResidualSlots;
    }
    return mega_mhc::CalcCachedGammaProducerUbSize(d, 1U) <= availableUb ? 1U : 0U;
}

} // namespace

REGISTER_OPS_TILING_TEMPLATE(MegaMhc, MegaMhcTiling, kTilingTemplatePriority);

/** Select Ascend 950. */
bool MegaMhcTiling::IsCapable()
{
    return context_ != nullptr && context_->GetPlatformInfo() != nullptr;
}

/** Read platform capacities. */
ge::graphStatus MegaMhcTiling::GetPlatformInfo()
{
    auto* platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "platformInfo", "nullptr",
                                              "platform information is required for Ascend 950 tiling");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    aicCount_ = platform.GetCoreNumAic();
    aivCount_ = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ABytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BBytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CBytes_);
    if (aicCount_ == 0 || aivCount_ < kMinimumAivCount || ubBytes_ == 0 || l1Bytes_ == 0 || l0ABytes_ == 0 ||
        l0BBytes_ == 0 || l0CBytes_ == 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "aicCount, aivCount, UB, L1, L0A, L0B, L0C",
            (std::to_string(aicCount_) + ", " + std::to_string(aivCount_) + ", " + std::to_string(ubBytes_) + ", " +
             std::to_string(l1Bytes_) + ", " + std::to_string(l0ABytes_) + ", " + std::to_string(l0BBytes_) + ", " +
             std::to_string(l0CBytes_))
                .c_str(),
            "AIC count and memory capacities must be positive; AIV count must be at least 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/** Parse shapes and attributes. */
ge::graphStatus MegaMhcTiling::GetShapeAttrsInfo()
{
    return ParseInputsAndAttributes();
}

/** Validate the supported input contract. */
ge::graphStatus MegaMhcTiling::ParseInputsAndAttributes()
{
    const bool hasShiftedPrevMix = context_->GetDynamicInputTensor(kShiftedPrevMixInputIndex, 0) != nullptr;
    // Inactive public outputs are absent or represented by empty launcher placeholders.
    const uint32_t reservedIndices[] = {kNewPrevMixOutputIndex, kYQuantOutputIndex, kYScaleOutputIndex};
    const ge::DataType reservedTypes[] = {ge::DT_FLOAT, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0};
    for (uint32_t i = 0; i < kReservedOutputCount; ++i) {
        if (reservedIndices[i] == kNewPrevMixOutputIndex && hasShiftedPrevMix) {
            continue;
        }
        const auto* desc = context_->GetOutputDesc(reservedIndices[i]);
        const auto* shape = context_->GetOutputShape(reservedIndices[i]);
        if (desc != nullptr && (shape == nullptr || desc->GetDataType() != reservedTypes[i] ||
                                shape->GetStorageShape().GetDimNum() != 1 || shape->GetStorageShape().GetDim(0) != 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), kOutputNames[reservedIndices[i]],
                (std::string("shape=") + (shape == nullptr ? "nullptr" : ShapeToString(shape->GetStorageShape())) +
                 ", dtype=" + ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()))
                    .c_str(),
                "reserved output must be absent or shape [0] with its declared dtype");
            return ge::GRAPH_FAILED;
        }
    }
    constexpr size_t kInputCount = 11;
    for (size_t index = 0; index < kInputCount; ++index) {
        // comb_res_mix is declared optional, but must be present in this version.
        const bool mayBeAbsent = MayOmitInput(index);
        // Optional inputs are indexed by their prototype position even when earlier options are absent.
        const bool declaredOptional = index >= kFirstOptionalInputIndex;
        const auto* desc = declaredOptional ? context_->GetOptionalInputDesc(index) : context_->GetInputDesc(index);
        const auto* shapeContext =
            declaredOptional ? context_->GetOptionalInputShape(index) : context_->GetInputShape(index);
        const auto* tensor = context_->GetDynamicInputTensor(index, 0);
        const bool missingRequiredTensor = !mayBeAbsent && tensor == nullptr;
        const bool missingMetadata = tensor != nullptr && (desc == nullptr || shapeContext == nullptr);
        if (missingRequiredTensor || missingMetadata) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), kInputNames[index],
                                                  "missing descriptor, shape or tensor",
                                                  "required inputs need a descriptor, shape and tensor; present "
                                                  "optional inputs need a descriptor and shape");
            return ge::GRAPH_FAILED;
        }
    }

    const auto* xShapeContext = context_->GetInputShape(0U);
    const auto& xShape = xShapeContext->GetStorageShape();
    if ((xShape.GetDimNum() != kUnbatchedXRank && xShape.GetDimNum() != kBatchedXRank) ||
        xShape.GetDim(xShape.GetDimNum() - 1) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ShapeToString(xShape).c_str(),
                                              "shape must be [T, D] or [B, S, D], with positive D");
        return ge::GRAPH_FAILED;
    }
    const int64_t d = xShape.GetDim(xShape.GetDimNum() - 1);
    if (d > kMaximumD || d % kDAlignment != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "D", std::to_string(d).c_str(),
                                              "D must be in [16, 16384] and divisible by 16");
        return ge::GRAPH_FAILED;
    }
    d_ = static_cast<uint32_t>(d);
    totalRows_ = 1;
    for (int64_t axis = 0; axis < xShape.GetDimNum() - 1; ++axis) {
        const int64_t dimension = xShape.GetDim(axis);
        if (dimension <= 0 || totalRows_ > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(dimension)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), ("x dimension " + std::to_string(axis)).c_str(),
                std::to_string(dimension).c_str(),
                "dimension must be positive and the product of leading dimensions must fit uint32");
            return ge::GRAPH_FAILED;
        }
        totalRows_ *= static_cast<uint64_t>(dimension);
    }
    if (totalRows_ == 0 || totalRows_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "total token count",
                                              std::to_string(totalRows_).c_str(),
                                              "total token count must be in [1, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& residual = context_->GetInputShape(1U)->GetStorageShape();
    if (!HasTrailingShape(residual, xShape.GetDimNum() + 1U, kHeadCount, d_)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "residual", ShapeToString(residual).c_str(),
                                              "shape must be [T, 4, D] or [B, S, 4, D], matching x layout and D");
        return ge::GRAPH_FAILED;
    }
    for (int64_t axis = 0; axis < xShape.GetDimNum() - 1; ++axis) {
        if (residual.GetDim(axis) != xShape.GetDim(axis)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "residual", ShapeToString(residual).c_str(),
                                                  ("leading dimensions must match x " + ShapeToString(xShape)).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    const auto& hPost = context_->GetInputShape(kPostMixInputIndex)->GetStorageShape();
    const auto& hRes = context_->GetOptionalInputShape(kCombResMixInputIndex)->GetStorageShape();
    const auto& phi = context_->GetInputShape(kFnInputIndex)->GetStorageShape();
    const auto& alpha = context_->GetInputShape(kMixScalesInputIndex)->GetStorageShape();
    const auto& bias = context_->GetInputShape(kMixBasesInputIndex)->GetStorageShape();
    const auto& gamma1 = context_->GetInputShape(kRmsNorm1WeightInputIndex)->GetStorageShape();
    const uint64_t q = kProjectionQ;
    k_ = kHeadCount * d_;
    const bool stateShapesValid = HasTrailingShape(hPost, xShape.GetDimNum() + 1U, kHeadCount, 1U) &&
                                  HasTrailingShape(hRes, xShape.GetDimNum() + 1U, kHeadCount, kHeadCount);
    const bool parameterShapesValid = HasMatrixShape(phi, static_cast<int64_t>(q), k_) &&
                                      HasVectorShape(alpha, kAlphaElements) &&
                                      HasVectorShape(bias, static_cast<int64_t>(q)) && HasVectorShape(gamma1, d_);
    if (!stateShapesValid || !parameterShapesValid) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "post_mix, comb_res_mix, fn, mix_scales, mix_bases, rms_norm_1_weight",
            (ShapeToString(hPost) + ", " + ShapeToString(hRes) + ", " + ShapeToString(phi) + ", " +
             ShapeToString(alpha) + ", " + ShapeToString(bias) + ", " + ShapeToString(gamma1))
                .c_str(),
            ("expected post_mix [..., 4, 1], comb_res_mix [..., 4, 4], fn [24, " + std::to_string(k_) +
             "], mix_scales [3], mix_bases [24], rms_norm_1_weight [" + std::to_string(d_) + "]")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    for (int64_t axis = 0; axis < xShape.GetDimNum() - 1; ++axis) {
        if (hPost.GetDim(axis) != xShape.GetDim(axis) || hRes.GetDim(axis) != xShape.GetDim(axis)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context_->GetNodeName(), "post_mix, comb_res_mix",
                (ShapeToString(hPost) + ", " + ShapeToString(hRes)).c_str(),
                ("leading dimensions must match x " + ShapeToString(xShape)).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->GetDynamicInputTensor(kShiftedPrevMixInputIndex, 0) != nullptr) {
        const auto* mixShape = context_->GetOptionalInputShape(kShiftedPrevMixInputIndex);
        if (mixShape == nullptr || mixShape->GetStorageShape() != hPost) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "shifted_prev_mix",
                (mixShape == nullptr ? std::string("nullptr") : ShapeToString(mixShape->GetStorageShape())).c_str(),
                ("expected the same shape as post_mix " + ShapeToString(hPost)).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    inputType_ = context_->GetInputDesc(0U)->GetDataType();
    if ((inputType_ != ge::DT_FLOAT16 && inputType_ != ge::DT_BF16) ||
        context_->GetInputDesc(1U)->GetDataType() != inputType_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "x dtype, residual dtype",
            (ge::TypeUtils::DataTypeToSerialString(inputType_) + ", " +
             ge::TypeUtils::DataTypeToSerialString(context_->GetInputDesc(1U)->GetDataType()))
                .c_str(),
            "x and residual must have the same FLOAT16 or BFLOAT16 dtype");
        return ge::GRAPH_FAILED;
    }
    for (size_t index = kPostMixInputIndex; index < kInputCount; ++index) {
        if (MayOmitInput(index) && context_->GetDynamicInputTensor(index, 0) == nullptr) {
            continue;
        }
        const auto* desc =
            (index >= kFirstOptionalInputIndex) ? context_->GetOptionalInputDesc(index) : context_->GetInputDesc(index);
        if (desc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), kInputNames[index],
                                      ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(), "FLOAT");
            return ge::GRAPH_FAILED;
        }
    }

    hasRms0_ = context_->GetDynamicInputTensor(kRmsNorm0WeightInputIndex, 0) != nullptr;
    hasMid_ = context_->GetDynamicInputTensor(kRmsNormMidWeightInputIndex, 0) != nullptr;
    if (hasRms0_) {
        const auto* gamma0Context = context_->GetOptionalInputShape(kRmsNorm0WeightInputIndex);
        if (gamma0Context == nullptr || !HasVectorShape(gamma0Context->GetStorageShape(), d_)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "rms_norm_0_weight",
                (gamma0Context == nullptr ? std::string("nullptr") : ShapeToString(gamma0Context->GetStorageShape()))
                    .c_str(),
                ("expected [" + std::to_string(d_) + "]").c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (hasMid_) {
        const auto& gamma2 = context_->GetOptionalInputShape(kRmsNormMidWeightInputIndex)->GetStorageShape();
        if (!HasVectorShape(gamma2, k_)) {
            OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "rms_norm_mid_weight", ShapeToString(gamma2).c_str(),
                                      ("[" + std::to_string(k_) + "]").c_str());
            return ge::GRAPH_FAILED;
        }
    }

    const uint32_t activeOutputs[] = {0U, 1U, kNewCombResMixOutputIndex, kYOutputIndex, kNewPrevMixOutputIndex};
    const gert::Shape* expectedShapes[] = {&residual, &hPost, &hRes, &xShape, &hPost};
    const ge::DataType outputTypes[] = {inputType_, ge::DT_FLOAT, ge::DT_FLOAT, inputType_, ge::DT_FLOAT};
    const uint32_t activeOutputCount = kActiveOutputCount + (hasShiftedPrevMix ? 1U : 0U);
    for (uint32_t i = 0; i < activeOutputCount; ++i) {
        const auto* desc = context_->GetOutputDesc(activeOutputs[i]);
        const auto* shape = context_->GetOutputShape(activeOutputs[i]);
        if (desc == nullptr || shape == nullptr || desc->GetDataType() != outputTypes[i] ||
            shape->GetStorageShape() != *expectedShapes[i]) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), kOutputNames[activeOutputs[i]],
                (std::string("shape=") + (shape == nullptr ? "nullptr" : ShapeToString(shape->GetStorageShape())) +
                 ", dtype=" +
                 (desc == nullptr ? "nullptr" : ge::TypeUtils::DataTypeToSerialString(desc->GetDataType())))
                    .c_str(),
                ("expected shape " + ShapeToString(*expectedShapes[i]) + " and dtype " +
                 ge::TypeUtils::DataTypeToSerialString(outputTypes[i]))
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }

    const auto* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "attrs", "nullptr",
                                              "operator attributes must be available");
        return ge::GRAPH_FAILED;
    }
    const float* normEps = attrs->GetAttrPointer<float>(0U);
    const float* mhcPreNormEps = attrs->GetAttrPointer<float>(1U);
    const float* hcEps = attrs->GetAttrPointer<float>(kHcEpsAttrIndex);
    const int64_t* opImplMode = attrs->GetAttrPointer<int64_t>(kOpImplModeAttrIndex);
    const auto* hcMult = attrs->GetAttrPointer<int64_t>(kHcMultAttrIndex);
    const auto* postScale = attrs->GetAttrPointer<float>(kPostScaleAttrIndex);
    const auto* sinkhornEps = attrs->GetAttrPointer<float>(kSinkhornEpsAttrIndex);
    const auto* sinkhornIters = attrs->GetAttrPointer<int64_t>(kSinkhornItersAttrIndex);
    const auto* rmsScale = attrs->GetAttrPointer<float>(kRmsNormScaleAttrIndex);
    const auto* outputModeAttr = attrs->GetAttrPointer<int64_t>(kOutputModeAttrIndex);
    const int64_t outputMode = outputModeAttr == nullptr ? 0 : *outputModeAttr;
    if (outputMode < 0 || outputMode > 2) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "output_mode", std::to_string(outputMode).c_str(),
                                  "0, 1 or 2");
        return ge::GRAPH_FAILED;
    }
    if (outputMode != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "output_mode",
                                              std::to_string(outputMode).c_str(),
                                              "MX Quant output modes 1 and 2 are reserved in this version");
        return ge::GRAPH_FAILED;
    }
    hcPostScale_ = postScale == nullptr ? kDefaultHcPostScale : *postScale;
    if ((hcMult != nullptr && *hcMult != kRequiredHcMultiplier) || !std::isfinite(hcPostScale_) ||
        (sinkhornEps != nullptr && *sinkhornEps != kDefaultEpsilon) ||
        (sinkhornIters != nullptr && *sinkhornIters != 0) || (rmsScale != nullptr && *rmsScale != 1.0f)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "hc_mult, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters, rms_norm_scale",
            (std::to_string(hcMult == nullptr ? kRequiredHcMultiplier : *hcMult) + ", " + std::to_string(hcPostScale_) +
             ", " + std::to_string(sinkhornEps == nullptr ? kDefaultEpsilon : *sinkhornEps) + ", " +
             std::to_string(sinkhornIters == nullptr ? 0 : *sinkhornIters) + ", " +
             std::to_string(rmsScale == nullptr ? 1.0f : *rmsScale))
                .c_str(),
            "hc_mult must be 4, hc_post_scale finite, and reserved attributes must retain defaults 1e-6, 0 and 1");
        return ge::GRAPH_FAILED;
    }
    normEps_ = normEps == nullptr ? kDefaultEpsilon : *normEps;
    mhcPreNormEps_ = mhcPreNormEps == nullptr ? kDefaultEpsilon : *mhcPreNormEps;
    hcEps_ = hcEps == nullptr ? kDefaultEpsilon : *hcEps;
    if (!std::isfinite(normEps_) || normEps_ <= 0.0f || !std::isfinite(mhcPreNormEps_) || mhcPreNormEps_ <= 0.0f ||
        !std::isfinite(hcEps_) || hcEps_ <= 0.0f) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "rms_norm_eps, hc_norm_eps, hc_pre_eps",
            (std::to_string(normEps_) + ", " + std::to_string(mhcPreNormEps_) + ", " + std::to_string(hcEps_)).c_str(),
            "all epsilon values must be finite and greater than zero");
        return ge::GRAPH_FAILED;
    }
    const int64_t mode = opImplMode == nullptr ? 0 : *opImplMode;
    if (mode != 0 && mode != 1) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "op_impl_mode", std::to_string(mode).c_str(),
                                  "0 (FP32) or 1 (HF32)");
        return ge::GRAPH_FAILED;
    }
    projectionMode_ = mode == 0 ? mega_mhc::ProjectionMode::FP32 : mega_mhc::ProjectionMode::HF32;
    // MK owns arbitrary M/K grids; only the complete-K BS schedule remains separate.
    const uint64_t bsThreshold = hasMid_ && mode == 1 ? kWithMidHf32BsRows : kOtherBsRows;
    scheduleMode_ =
        (hasMid_ || mode == 1) && totalRows_ >= bsThreshold ? mega_mhc::ScheduleMode::BS : mega_mhc::ScheduleMode::MK;
    return ge::GRAPH_SUCCESS;
}

/** Serialize the host-selected M/K grid and all Cube loop/tail decisions into TilingData. */
void MegaMhcTiling::FillCubeTilingData()
{
    tilingData_.set_activeAiv(cubeTiling_.activeAiv);
    tilingData_.set_kGrid(cubeTiling_.kGrid);
    tilingData_.set_mWaveCount(cubeTiling_.mWaveCount);
    tilingData_.set_rowsPerWave(cubeTiling_.rowsPerWave);
    tilingData_.set_lastWaveMGrid(cubeTiling_.lastWaveMGrid);
    tilingData_.set_mTile(cubeTiling_.mTile);
    tilingData_.set_splitK(cubeTiling_.splitK);
    tilingData_.set_largeKCoreCount(cubeTiling_.largeKCoreCount);
    tilingData_.set_shortSplitK(cubeTiling_.shortSplitK);
    tilingData_.set_shortKCoreL1FullLoops(cubeTiling_.shortKCoreL1FullLoops);
    tilingData_.set_l1K(cubeTiling_.l1K);
    tilingData_.set_l1KFullLoops(cubeTiling_.l1KFullLoops);
    tilingData_.set_l1KTail(cubeTiling_.l1KTail);
    tilingData_.set_lastKCoreL1FullLoops(cubeTiling_.lastKCoreL1FullLoops);
    tilingData_.set_lastKCoreL1Tail(cubeTiling_.lastKCoreL1Tail);
    tilingData_.set_lastMCoreRows(cubeTiling_.lastMCoreRows);
    tilingData_.set_loadMStride(cubeTiling_.loadMStride);
    tilingData_.set_loadNStride(cubeTiling_.loadNStride);
    tilingData_.set_fullL0LoadKStep(cubeTiling_.fullL0LoadKStep);
    tilingData_.set_l0TailLoadKStep(cubeTiling_.l0TailLoadKStep);
    tilingData_.set_tailL1TailLoadKStep(cubeTiling_.tailL1TailLoadKStep);
    tilingData_.set_lastCoreTailLoadKStep(cubeTiling_.lastCoreTailLoadKStep);
    tilingData_.set_baseK(cubeTiling_.baseK);
    // Mmad API requires PIPE_M when (m/16)*(n/16)<10; decide once on Host.
    tilingData_.set_mmadNeedsBarrier(
        (cubeTiling_.nzM / kMmadBlockSize) * (cubeTiling_.nzN / kMmadBlockSize) < kMmadBarrierRepeatLimit ? 1U : 0U);
    tilingData_.set_nzM(cubeTiling_.nzM);
    tilingData_.set_nzN(cubeTiling_.nzN);
    tilingData_.set_l1ASlotElements(cubeTiling_.l1ASlotElements);
    tilingData_.set_l1BOffsetElements(cubeTiling_.l1BOffsetElements);
    tilingData_.set_l1BSlotElements(cubeTiling_.l1BSlotElements);
    tilingData_.set_l0ASlotElements(cubeTiling_.l0ASlotElements);
    tilingData_.set_l0BSlotElements(cubeTiling_.l0BSlotElements);
    tilingData_.set_l0CSlotElements(cubeTiling_.l0CSlotElements);

    tilingData_.set_l0KFullLoops(cubeTiling_.l0KLoops);
    tilingData_.set_l0KTail(cubeTiling_.l0KTail);
    tilingData_.set_tailL1L0FullLoops(cubeTiling_.tailL1L0FullLoops);
    tilingData_.set_tailL1L0Tail(cubeTiling_.tailL1L0Tail);
    tilingData_.set_lastCoreTailL1L0FullLoops(cubeTiling_.lastCoreTailL1L0FullLoops);
    tilingData_.set_lastCoreTailL1L0Tail(cubeTiling_.lastCoreTailL1L0Tail);
    tilingData_.set_partialGroupK(cubeTiling_.partialGroupK);
    tilingData_.set_partialGroupCount(cubeTiling_.partialGroupCount);
}

/** Plan Cube/UB tiles and workspace from platform capacities. */
ge::graphStatus MegaMhcTiling::DoOpTiling()
{
    const uint64_t availableUb = std::min<uint64_t>(ubBytes_, kUbLimitBytes);
    const bool hasShiftedPrevMix = context_->GetDynamicInputTensor(kShiftedPrevMixInputIndex, 0) != nullptr;
    const bool packedNz = (hasMid_ ? MegaMhc::kWithMidProjectionTransport : MegaMhc::kNoMidProjectionTransport) ==
                          MegaMhc::ProjectionTransport::UbToL1;
    mega_mhc::CubeTilingParams request;
    request.rows = totalRows_;
    request.schedule = scheduleMode_;
    request.wavePipeline = false;
    request.k = k_;
    request.q = kProjectionQ;
    request.d = d_;
    request.ubBytes = availableUb;
    request.packedNz = packedNz;
    request.aicCount = aicCount_;
    request.aivCount = aivCount_;
    request.l1ABytes = l1Bytes_ / kL1HalfDivisor;
    request.l1BBytes = l1Bytes_ - request.l1ABytes;
    request.l0ABytes = l0ABytes_;
    request.l0BBytes = l0BBytes_;
    request.l0CBytes = l0CBytes_;
    request.mode = projectionMode_;
    request.maxReuseMStages =
        hasMid_ && scheduleMode_ == mega_mhc::ScheduleMode::MK ? MegaMhc::kMaxProjectionMStages : 1U;
    if (!mega_mhc::CalcCubeTiling(request, cubeTiling_)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "T, K, L1, L0A, L0B, L0C",
            (std::to_string(totalRows_) + ", " + std::to_string(k_) + ", " + std::to_string(l1Bytes_) + ", " +
             std::to_string(l0ABytes_) + ", " + std::to_string(l0BBytes_) + ", " + std::to_string(l0CBytes_))
                .c_str(),
            "no legal M/K Cube tiling fits the reported capacities");
        return ge::GRAPH_FAILED;
    }

    const auto rowTiling =
        mega_mhc::CalcRowTiling(d_, hasMid_, cubeTiling_.partialGroupCount, cubeTiling_.fixpipeRowStride,
                                cubeTiling_.kGrid, availableUb, hasShiftedPrevMix);
    if (rowTiling.count == 0U) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "D, UB bytes",
                                               (std::to_string(d_) + ", " + std::to_string(ubBytes_)).c_str(),
                                               "no legal D tile fits the producer and epilogue UB requirements");
        return ge::GRAPH_FAILED;
    }
    const bool tiledRows = rowTiling.enabled != 0U;
    dispatchMode_ = tiledRows ? mega_mhc::ScheduleMode::MK : scheduleMode_;
    tilingData_.set_rowTileMode(rowTiling.enabled);
    tilingData_.set_dTile(rowTiling.tile);
    tilingData_.set_dTileCount(rowTiling.count);
    tilingData_.set_dTileTail(rowTiling.tail);

    // Try waves only when every producer owns complete rows and all live buffers fit.
    if (CanTryWaveTiling(request, hasMid_, tiledRows)) {
        // Prefer a complete-wave tiling; if alignment invalidates it, retain the larger batch and use its tail.
        for (uint32_t candidate = 0U; candidate < kWaveCandidateCount; ++candidate) {
            auto waveRequest = request;
            waveRequest.wavePipeline = true;
            waveRequest.preferCompleteWaves = candidate == 0U;
            mega_mhc::CubeTilingInfo waveTiling;
            if (!mega_mhc::CalcCubeTiling(waveRequest, waveTiling) ||
                !CanUseWaveCandidate(cubeTiling_, waveTiling, d_, availableUb)) {
                continue;
            }
            request = waveRequest;
            cubeTiling_ = waveTiling;
            break;
        }
    }
    tilingData_.set_totalRows(static_cast<uint32_t>(totalRows_));
    tilingData_.set_d(d_);
    tilingData_.set_k(k_);

    tilingData_.set_normEps(normEps_);
    tilingData_.set_mhcPreNormEps(mhcPreNormEps_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_hcPostScale(hcPostScale_);
    FillCubeTilingData();
    const uint32_t waveProducerRows = request.wavePipeline ? cubeTiling_.mTile / (cubeTiling_.kGrid * kAivPerAic) : 0U;
    const uint32_t waveTileCount = cubeTiling_.lastKCoreL1FullLoops;
    const uint32_t rowsPerTick =
        request.wavePipeline ? static_cast<uint32_t>(CeilDiv(waveProducerRows, waveTileCount - 1U)) : 0U;
    const uint32_t tickCount =
        request.wavePipeline ? static_cast<uint32_t>(CeilDiv(waveProducerRows, rowsPerTick)) : 0U;
    tilingData_.set_waveProducerRowsPerTick(rowsPerTick);
    tilingData_.set_wavePipeline(request.wavePipeline ? 1U : 0U);

    tilingData_.set_waveProducerRows(waveProducerRows);
    tilingData_.set_lastWaveFullProducers(request.wavePipeline ? cubeTiling_.lastWaveRows / waveProducerRows : 0U);
    tilingData_.set_lastWaveProducerTailRows(request.wavePipeline ? cubeTiling_.lastWaveRows % waveProducerRows : 0U);
    tilingData_.set_waveProducerTileStride(request.wavePipeline ? (waveTileCount - 1U) / tickCount : 0U);
    tilingData_.set_waveGroupSize(request.wavePipeline ? cubeTiling_.kGrid * kAivPerAic : 0U);

    // Exact simultaneous allocations, each padded like Kernel::InitUbBuffer.
    // The baseline does not reuse live buffers within a phase. Destroy() separates
    // RMS0, producer/projection and epilogue lifetimes, so their peaks do not add.
    const uint32_t projectionRows = static_cast<uint32_t>(CeilDiv(cubeTiling_.mTile, kAivPerAic));
    // Keep short-T producer prefetch: gamma caching only saves loads when each AIV processes multiple rows.
    const bool singleMidResidual = !tiledRows && hasMid_ && scheduleMode_ != mega_mhc::ScheduleMode::BS &&
                                   totalRows_ >= static_cast<uint64_t>(kMinimumRowsPerAiv) * cubeTiling_.activeAiv &&
                                   mega_mhc::CanUseSingleResidualForMid(d_, availableUb);
    // 1: prefetch the complete next row; 0: prefetch only hOut/coefficients and serialize residual reuse.
    tilingData_.set_producerPrefetch(singleMidResidual ? 0U : 1U);
    tilingData_.set_projectionRows(projectionRows);
    // Layout padding is optional: never disable an otherwise legal wave to obtain it.
    const uint64_t liveProducer = request.wavePipeline ? mega_mhc::CalcProducerUbSize(d_, false, true) : 0U;
    const auto projectionUbTiling =
        mega_mhc::CalcProjectionUbTiling(cubeTiling_.l1K, projectionRows, availableUb, liveProducer, packedNz);
    const uint32_t stageRowStride = projectionUbTiling.rowStride;
    const bool padStageRows = stageRowStride != projectionRows;
    tilingData_.set_stageRowStride(stageRowStride);
    // Direct NZ stores require padding to avoid their strided-store penalty; preserve waves via GM otherwise.
    tilingData_.set_projectionTransport(static_cast<uint32_t>(padStageRows ? MegaMhc::ProjectionTransport::UbToL1 :
                                                                             MegaMhc::ProjectionTransport::GmStage));
    // Keep the epilogue's reduction batch explicit, independently of its one-row compute tile.
    const bool needsGammaCacheLayout = !tiledRows && hasMid_ && scheduleMode_ == mega_mhc::ScheduleMode::BS &&
                                       totalRows_ >= kWithMidPrefetchMinRows &&
                                       mega_mhc::CalcProducerUbSize(d_, true, false, k_) > availableUb;
    const uint32_t cachedResidualSlots = needsGammaCacheLayout ? SelectCachedGammaResidualSlots(d_, availableUb) : 0U;
    const bool cacheFullGamma = cachedResidualSlots != 0U;
    tilingData_.set_withMidProducerResidualSlots(cachedResidualSlots);
    uint32_t gammaTile = k_;
    if (tiledRows) {
        gammaTile = rowTiling.tile;
    } else if (hasMid_ && !cacheFullGamma && !singleMidResidual) {
        gammaTile = mega_mhc::CalcMidGammaTile(d_, availableUb);
    }
    if (gammaTile == 0U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t gammaCount = static_cast<uint32_t>(CeilDiv(k_, gammaTile));
    tilingData_.set_midGammaTile(gammaTile);
    tilingData_.set_midGammaTileCount(gammaCount);
    tilingData_.set_midGammaLastTile(k_ - (gammaCount - 1U) * gammaTile);
    // Wave projection rows do not constrain the independent epilogue lifetime or token ownership.
    const uint32_t reductionRowLimit =
        request.wavePipeline ?
            static_cast<uint32_t>(CeilDiv(totalRows_, cubeTiling_.activeAiv)) :
            std::min(projectionRows, static_cast<uint32_t>(CeilDiv(totalRows_, cubeTiling_.activeAiv)));
    const uint32_t capacityRows =
        tiledRows ?
            1U :
            mega_mhc::FitReductionRows(reductionRowLimit, d_, cubeTiling_.partialGroupCount,
                                       cubeTiling_.fixpipeRowStride, cubeTiling_.kGrid, availableUb, hasShiftedPrevMix);
    const uint32_t reduceRows = tiledRows ? 1U : mega_mhc::BalanceReductionRows(reductionRowLimit, capacityRows);
    if (reduceRows == 0U) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_reduceRows(reduceRows);
    tilingData_.set_reduceRowsAligned((reduceRows + kReductionAlignmentOffset) / mega_mhc::kReductionRowAlignment *
                                      mega_mhc::kReductionRowAlignment);
    // Dead partial storage can own the alternate x2 row. Small BS batches reserve sufficient space
    // explicitly instead of disabling DMA overlap solely because their partial tensor is smaller.
    const bool partialHoldsInput = static_cast<uint64_t>(reduceRows) * cubeTiling_.partialGroupCount *
                                       cubeTiling_.fixpipeRowStride * sizeof(float) >=
                                   static_cast<uint64_t>(k_) * kInputStorageBytes;
    const bool reserveEpilogueInput =
        !partialHoldsInput && hasMid_ && scheduleMode_ == mega_mhc::ScheduleMode::BS &&
        totalRows_ >= kWithMidPrefetchMinRows &&
        mega_mhc::CalcEpilogueUbSize(d_, cubeTiling_.partialGroupCount, cubeTiling_.fixpipeRowStride, reduceRows,
                                     cubeTiling_.kGrid, true, hasShiftedPrevMix) <= availableUb;
    tilingData_.set_epiloguePrefetch(
        tiledRows ? 0U : (partialHoldsInput ? 1U : (reserveEpilogueInput ? kEpiloguePrefetchInput : 0U)));
    const uint64_t producerPeak =
        tiledRows ? mega_mhc::CalcRowTileProducerUbSize(rowTiling.tile) :
                    (cacheFullGamma ? mega_mhc::CalcCachedGammaProducerUbSize(d_, cachedResidualSlots) :
                                      mega_mhc::CalcProducerUbSize(d_, hasMid_, false, gammaTile,
                                                                   singleMidResidual ? 1U : kMultipleMidResidualSlots));
    // Add a second FP32 UB stage only when the simultaneous live buffers fit.
    // Cube geometry and GM staging remain unchanged.
    const uint64_t concurrentProducer = request.wavePipeline ? mega_mhc::CalcProducerUbSize(d_, false, true) : 0U;
    const uint32_t stageUbSlots = projectionUbTiling.stageSlots;
    tilingData_.set_stageUbSlots(stageUbSlots);
    const uint64_t projectionPeak = projectionUbTiling.bytes;
    // Reuse each B panel across M stages; pad each stage's row statistics to a register boundary.
    const uint32_t statsStride =
        (projectionRows + kFp32RegisterElements - 1U) / kFp32RegisterElements * kFp32RegisterElements;
    tilingData_.set_projectionStatsStride(statsStride);
    const uint64_t singleStageStatsBytes = mega_mhc::CalcUbAllocSize(projectionRows * sizeof(float));
    if (projectionPeak > availableUb) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "projection UB bytes", std::to_string(projectionPeak).c_str(),
            ("required bytes must not exceed available UB " + std::to_string(availableUb)).c_str());
        return ge::GRAPH_FAILED;
    }
    const uint64_t ubStatsCapacity =
        (availableUb - projectionPeak + singleStageStatsBytes) / (statsStride * sizeof(float));
    // One stage uses the unpadded statistics already included in projectionPeak.
    const uint32_t mStages =
        hasMid_ && scheduleMode_ == mega_mhc::ScheduleMode::MK ?
            static_cast<uint32_t>(std::max<uint64_t>(
                1U, std::min<uint64_t>({request.maxReuseMStages, cubeTiling_.mWaveCount, ubStatsCapacity,
                                        l0CBytes_ / (cubeTiling_.l0CSlotElements * sizeof(float))}))) :
            1U;
    const uint64_t extraStats =
        mStages > 1U ? mega_mhc::CalcUbAllocSize(static_cast<uint64_t>(mStages) * statsStride * sizeof(float)) -
                           singleStageStatsBytes :
                       0U;
    tilingData_.set_projectionMStages(mStages);
    // Two full B panels must fit L0B; keep all other cube geometry unchanged.
    const bool residentB =
        mStages > 1U &&
        static_cast<uint64_t>(kResidentBSlots) * cubeTiling_.l1BSlotElements * sizeof(float) <= l0BBytes_;
    tilingData_.set_projectionBResident(residentB ? 1U : 0U);
    if (residentB) {
        tilingData_.set_l0BSlotElements(cubeTiling_.l1BSlotElements);
    }
    const uint64_t extraInput =
        mega_mhc::CalcUbAllocSize(static_cast<uint64_t>(projectionRows) * cubeTiling_.l1K * sizeof(uint16_t));
    const uint64_t projectionWithStatsPeak = projectionPeak + (mStages > 1U ? extraStats : 0U);
    const uint32_t inputSlots =
        mStages > 1U && projectionWithStatsPeak + extraInput <= availableUb ? kPrefetchInputSlotCount : kSlotCount;
    tilingData_.set_projectionInputSlots(inputSlots);
    const uint64_t batchedProjectionPeak =
        projectionWithStatsPeak + (inputSlots == kPrefetchInputSlotCount ? extraInput : 0U);
    const uint64_t concurrentUb = projectionPeak + concurrentProducer;
    // FP32 projection leaves Cube compute time for the previous wave's vector epilogue.
    // Keep HF32's lighter projection on the separate epilogue schedule.
    const uint32_t waveEpilogueRows =
        request.mode == mega_mhc::ProjectionMode::FP32 && request.wavePipeline && cubeTiling_.mWaveCount > 1U &&
                concurrentUb < availableUb ?
            mega_mhc::FitReductionRows(waveProducerRows, d_, tilingData_.get_partialGroupCount(),
                                       cubeTiling_.fixpipeRowStride, cubeTiling_.kGrid, availableUb - concurrentUb,
                                       hasShiftedPrevMix) :
            0U;
    tilingData_.set_waveEpilogueRows(waveEpilogueRows);
    if (waveEpilogueRows != 0U) {
        // This schedule owns the epilogue; expose its actual row tiling directly to the kernel.
        tilingData_.set_reduceRows(waveEpilogueRows);
        tilingData_.set_reduceRowsAligned((waveEpilogueRows + kReductionAlignmentOffset) /
                                          mega_mhc::kReductionRowAlignment * mega_mhc::kReductionRowAlignment);
        tilingData_.set_epiloguePrefetch(0U);
    }
    // Compact output retains the existing workspace and stage offsets.
    const bool compactPartials = !tiledRows && !hasMid_ && scheduleMode_ == mega_mhc::ScheduleMode::BS &&
                                 projectionMode_ == mega_mhc::ProjectionMode::HF32 && cubeTiling_.kGrid == 1U;
    tilingData_.set_compactPartials(compactPartials ? 1U : 0U);

    const auto bsEpilogue = compactPartials ?
                                mega_mhc::CalcBsHfEpilogueTiling(
                                    totalRows_, d_, tilingData_.get_reduceRows(), tilingData_.get_reduceRowsAligned(),
                                    cubeTiling_.fixpipeRowStride, cubeTiling_.kGrid, availableUb, hasShiftedPrevMix) :
                                mega_mhc::BsHfEpilogueTiling{};
    tilingData_.set_bsEpilogueBatch(bsEpilogue.batch);
    tilingData_.set_bsEpilogueInputBytes(bsEpilogue.inputBytes);
    const uint64_t epiloguePeak =
        bsEpilogue.batch != 0U ? bsEpilogue.ubBytes :
        tiledRows              ? mega_mhc::CalcRowTileEpilogueUbSize(rowTiling.tile, cubeTiling_.partialGroupCount,
                                                                     cubeTiling_.fixpipeRowStride, cubeTiling_.kGrid,
                                                                     hasShiftedPrevMix) :
                                 mega_mhc::CalcEpilogueUbSize(
                        d_, cubeTiling_.partialGroupCount, cubeTiling_.fixpipeRowStride, reduceRows, cubeTiling_.kGrid,
                        tilingData_.get_epiloguePrefetch() == kEpiloguePrefetchInput, hasShiftedPrevMix);
    const uint64_t rms0Peak =
        mega_mhc::CalcUbAllocSize(d_ * kInputStorageBytes) + mega_mhc::CalcUbAllocSize(kRmsScalarScratchBytes);
    const uint64_t waveEpiloguePeak =
        waveEpilogueRows != 0U ?
            mega_mhc::CalcEpilogueUbSize(d_, cubeTiling_.partialGroupCount, cubeTiling_.fixpipeRowStride,
                                         waveEpilogueRows, cubeTiling_.kGrid, false, hasShiftedPrevMix) :
            0U;
    const uint64_t pipelinePeak =
        request.wavePipeline ? mega_mhc::CalcProducerUbSize(d_, false, true) + projectionPeak + waveEpiloguePeak : 0U;
    const uint64_t ubPeak = std::max({producerPeak, batchedProjectionPeak, epiloguePeak, rms0Peak, pipelinePeak});
    if (ubPeak > availableUb) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "stage peak UB bytes", std::to_string(ubPeak).c_str(),
            ("required bytes must not exceed available UB " + std::to_string(availableUb)).c_str());
        return ge::GRAPH_FAILED;
    }
    const uint64_t l1Elements =
        cubeTiling_.l1BOffsetElements + static_cast<uint64_t>(kResidentBSlots) * cubeTiling_.l1BSlotElements;
    tilingData_.set_l1PeakBytes(l1Elements * sizeof(float));
    return CalcWorkspaceSize();
}

/** Compute disjoint FP32 partial and optional GM-staging workspace regions with 64-bit offsets. */
ge::graphStatus MegaMhcTiling::CalcWorkspaceSize()
{
    // Retain the no_mid prefix so partial and staging offsets remain unchanged.
    uint64_t prefixBytes = 0;
    if (!CheckedBytes(totalRows_, 1, 1, sizeof(float), prefixBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "totalRows", std::to_string(totalRows_).c_str(),
                                              "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    if (hasMid_) {
        prefixBytes = 0; // WithMid partials start at offset zero.
    } else if (tilingData_.get_wavePipeline() != 0U) {
        // Retain the no_mid wave prefix used by the workspace layout.
        prefixBytes =
            std::max<uint64_t>(prefixBytes, static_cast<uint64_t>(cubeTiling_.activeAiv) * kWavePrefixBytesPerAiv);
    }
    uint64_t partialMmBytes = 0;
    uint64_t partialRmsBytes = 0;
    if (!CheckedBytes(cubeTiling_.partialGroupCount, totalRows_, cubeTiling_.fixpipeRowStride, sizeof(float),
                      partialMmBytes) ||
        !CheckedBytes(cubeTiling_.kGrid, totalRows_, 1, sizeof(float), partialRmsBytes)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "partialGroupCount, totalRows, fixpipeRowStride, kGrid",
            (std::to_string(cubeTiling_.partialGroupCount) + ", " + std::to_string(totalRows_) + ", " +
             std::to_string(cubeTiling_.fixpipeRowStride) + ", " + std::to_string(cubeTiling_.kGrid))
                .c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t partialMmOffset = 0;
    if (!AlignWorkspace(prefixBytes, kWorkspaceAlign, partialMmOffset)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "RMS0 workspace bytes",
                                              std::to_string(prefixBytes).c_str(),
                                              "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    if (partialMmBytes > std::numeric_limits<uint64_t>::max() - partialMmOffset) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "partialMmOffset, partialMmBytes",
            (std::to_string(partialMmOffset) + ", " + std::to_string(partialMmBytes)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t partialRmsOffset = 0;
    if (!AlignWorkspace(partialMmOffset + partialMmBytes, kWorkspaceAlign, partialRmsOffset)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "partialMmOffset, partialMmBytes",
            (std::to_string(partialMmOffset) + ", " + std::to_string(partialMmBytes)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    if (partialRmsBytes > std::numeric_limits<uint64_t>::max() - partialRmsOffset) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "partialRmsOffset, partialRmsBytes",
            (std::to_string(partialRmsOffset) + ", " + std::to_string(partialRmsBytes)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t stageOffset = 0;
    if (!AlignWorkspace(partialRmsOffset + partialRmsBytes, kWorkspaceAlign, stageOffset)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "partialRmsOffset, partialRmsBytes",
            (std::to_string(partialRmsOffset) + ", " + std::to_string(partialRmsBytes)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t stageElementsBase = 0;
    if (!CheckedBytes(cubeTiling_.activeAic, kSlotCount, cubeTiling_.mTile, 1, stageElementsBase)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "activeAic, slotCount, mTile",
                                               (std::to_string(cubeTiling_.activeAic) + ", " +
                                                std::to_string(kSlotCount) + ", " + std::to_string(cubeTiling_.mTile))
                                                   .c_str(),
                                               "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    // `CheckedBytes` receives three extents; multiply by l1K and FP32 separately
    // so every size conversion remains checked before TilingData serialization.
    if (cubeTiling_.l1K != 0 && stageElementsBase > std::numeric_limits<uint64_t>::max() / cubeTiling_.l1K) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "stageElementsBase, l1K",
            (std::to_string(stageElementsBase) + ", " + std::to_string(cubeTiling_.l1K)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t stageElements = stageElementsBase * cubeTiling_.l1K;
    if (stageElements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "stageElements",
                                              std::to_string(stageElements).c_str(),
                                              "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t stageBytes = stageElements * sizeof(float);
    if (stageBytes > std::numeric_limits<uint64_t>::max() - stageOffset) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "stageOffset, stageBytes",
            (std::to_string(stageOffset) + ", " + std::to_string(stageBytes)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t userBytes = stageOffset + stageBytes;
    if (userBytes > std::numeric_limits<uint64_t>::max() - kRuntimeWorkspaceBytes) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "userBytes, runtimeWorkspaceBytes",
            (std::to_string(userBytes) + ", " + std::to_string(kRuntimeWorkspaceBytes)).c_str(),
            "workspace size arithmetic and alignment must not overflow uint64");
        return ge::GRAPH_FAILED;
    }
    workspaceBytes_ = userBytes + kRuntimeWorkspaceBytes;
    tilingData_.set_partialMmOffset(partialMmOffset);
    tilingData_.set_partialRmsOffset(partialRmsOffset);
    tilingData_.set_stageOffset(stageOffset);
    return ge::GRAPH_SUCCESS;
}

/** Basic/RegBase kernels require no library tiling. */
ge::graphStatus MegaMhcTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

/** Encode schedule, Mid, Cube precision and the optional RMS0 bypass in the tiling key. */
uint64_t MegaMhcTiling::GetTilingKey() const
{
    static_assert(static_cast<uint32_t>(mega_mhc::ScheduleMode::MK) == MEGA_MHC_SPLIT_M_K);
    static_assert(static_cast<uint32_t>(mega_mhc::ScheduleMode::BS) == MEGA_MHC_SPLIT_BS);
    // D-tiled rows use global producer/epilogue ownership. Keep the planned Cube grid,
    // but select its actual staged MK executor here instead of rerouting inside Kernel.
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(dispatchMode_), hasMid_,
                              projectionMode_ == mega_mhc::ProjectionMode::HF32, !hasRms0_);
}

/** Workspace is sized in DoOpTiling. */
ge::graphStatus MegaMhcTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

/** Serialize TilingData, launch the M/K block count, and publish its workspace requirement. */
ge::graphStatus MegaMhcTiling::PostTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity() ||
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "tilingData bytes, buffer capacity",
            (std::to_string(tilingData_.GetDataSize()) + ", " +
             std::to_string(context_->GetRawTilingData()->GetCapacity()))
                .c_str(),
            "TilingData must fit the buffer capacity and be aligned to sizeof(uint64_t)");
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(cubeTiling_.activeAic);
    context_->SetScheduleMode(1U);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "workspace sizes", "nullptr",
                                                      "GetWorkspaceSizes(1) must return a valid array"),
                return ge::GRAPH_FAILED);
    if (workspaceBytes_ > std::numeric_limits<size_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "workspace bytes",
                                              std::to_string(workspaceBytes_).c_str(),
                                              "workspace bytes must fit size_t");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(workspaceBytes_);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
