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
 * \file mhc_pre_tiling.cpp
 * \brief
 */

#include "mhc_pre_tiling.h"
#include <initializer_list>
#include <string>
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "../../../op_kernel/arch35/mhc_pre_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

// Supported input ranks and input tensor indices.
const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr uint32_t X_INDEX = 0;
const constexpr uint32_t PHI_INDEX = 1;
const constexpr uint32_t ALPHA_INDEX = 2;
const constexpr uint32_t BIAS_INDEX = 3;
const constexpr uint32_t GAMMA_INDEX = 4;

// Output tensor indices. These are shared by validation and optional-output handling.
constexpr size_t OUT_H_IN_INDEX = 0;
constexpr size_t OUT_H_POST_INDEX = 1;
constexpr size_t OUT_H_RES_INDEX = 2;
constexpr size_t OUT_INV_RMS_INDEX = 3;
constexpr size_t OUT_H_MIX_INDEX = 4;
constexpr size_t OUT_H_PRE_INDEX = 5;

// Input layout dimension indices.
const constexpr int64_t INDEX_B_BSND = 0;
const constexpr int64_t INDEX_S_BSND = 1;
const constexpr int64_t INDEX_N_BSND = 2;
const constexpr int64_t INDEX_D_BSND = 3;

const constexpr int64_t INDEX_T_TND = 0;
const constexpr int64_t INDEX_N_TND = 1;
const constexpr int64_t INDEX_D_TND = 2;

// Shape constraints and default vector tile sizes.
const constexpr uint32_t N_VALID_VALUES[] = {4, 6, 8};
const constexpr uint32_t D_ALIGNMENT = 16;
const constexpr uint32_t CHUNK_T_MAX = 128;
const constexpr uint32_t V1_CHUNK_D_SIZE = 5120;
const constexpr uint32_t CHUNK_T_CALC_FACTOR = 32;
const constexpr uint64_t D_MIN = 16;
const constexpr uint64_t D_MAX = 16384;
const constexpr size_t SYSTEM_WORKSPACE = 20 * 1024 * 1024;

// Runtime scheduling configuration.
const constexpr uint32_t SCHEDULE_MODE = 1;
const constexpr uint32_t AIV_AIC_RATIO = 2U;

// Attributes and their accepted values.
const constexpr float DEFAULT_NORM_EPS = 1e-6f;
const constexpr float DEFAULT_HC_EPS = 1e-6f;
const constexpr uint32_t IMPL_MODE_ATTR_INDEX = 3;
const constexpr uint32_t OUT_FLAG_ATTR_INDEX = 0;
const constexpr int64_t IMPL_MODE_FP32 = 0;
const constexpr int64_t IMPL_MODE_HF32 = 1;

// Decode and Basic API BS/ND tiling parameters.
const constexpr uint64_t DECODE_BS_THRESHOLD = 256;
const constexpr uint32_t DECODE_CHUNK_T_SIZE = 2;
const constexpr size_t DECODE_WORKSPACE_ALIGN = 32;
const constexpr size_t SPLIT_BS_GAMMA_BUFFER_LENGTH = 256;
const constexpr size_t SPLIT_ND_GAMMA_BUFFER_LENGTH = 2048;
const constexpr uint32_t BASIC_API_ALIGN_BYTES = 32;
const constexpr uint32_t BASIC_API_FLOAT_ALIGN = BASIC_API_ALIGN_BYTES / sizeof(float);
const constexpr uint32_t BASIC_API_M_L1_SIZE = 32;
const constexpr uint32_t BASIC_API_K_UB_SIZE = 256;
const constexpr uint32_t BASIC_API_K_L1_SIZE = 256;
const constexpr uint32_t BASIC_API_X_STAGE_BUFFER_COUNT = 2;
const constexpr uint32_t BASIC_API_BUFFER_POOL0_SIZE = 64 * 1024;
const constexpr uint32_t BASIC_API_BUFFER_POOL1_SIZE = 96 * 1024;

// M-K routing, tiling and workspace parameters.
// Keep automatic M-K routing inside the range covered by the generalized
// precision and performance sweeps. ND and BS remain available as fallbacks
// and for future shape-specific routing outside this validated domain.
const constexpr uint64_t M_K_MAX_VALIDATED_TOTAL_LENGTH = 10240;
const constexpr uint64_t M_K_MIN_MAT_K = 512;
const constexpr int32_t BATCH_CONSISTENCY_LEVEL = 3;
const constexpr uint64_t M_K_FP32_L1_MIN_TOTAL_LENGTH = 512;
const constexpr uint64_t M_K_N8_L1_MAX_TOTAL_LENGTH = 1536;
const constexpr uint64_t M_K_MIN_STAGE2_ROWS_PER_CORE = 2;
const constexpr uint32_t M_K_M_L1_MAX_SIZE = 256;
const constexpr uint32_t M_K_SPLIT_ALIGN = 256;
const constexpr uint32_t M_K_A_L1_ELEMENT_COUNT = 128 * 256;
const constexpr uint32_t M_K_B_L1_ELEMENT_COUNT = 128 * 256;
const constexpr uint32_t M_K_K_L1_MAX_SIZE = 1024;
const constexpr uint32_t M_K_SEQUENTIAL_PARTIAL_K = 1024;
const constexpr uint32_t M_K_SEQUENTIAL_PARTIAL_THRESHOLD = 2048;
const constexpr uint32_t M_K_K_L1_ALIGN = 128;
const constexpr size_t M_K_WORKSPACE_ALIGN = 512;
const constexpr uint32_t M_K_USE_L1_STAGE = 0;
const constexpr uint32_t M_K_USE_GM_STAGE = 1;

// Generic rank and dimension indices used by output-shape validation.
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_NUM_4 = 4;
static constexpr uint32_t HAS_GAMMA_TRUE = 1;

// Diagnostic tensor names.
static constexpr const char *INPUT_NAMES[] = {"x", "phi", "alpha", "bias", "gamma"};
static constexpr const char *OUTPUT_NAMES[] = {"hIn", "hPost", "hRes", "invRms", "hMix", "hPre"};

static std::string ShapeToString(std::initializer_list<uint64_t> dims)
{
    std::string result = "[";
    size_t index = 0;
    for (const auto dim : dims) {
        result += (index++ == 0 ? "" : ", ") + std::to_string(dim);
    }
    return result + "]";
}

REGISTER_OPS_TILING_TEMPLATE(MhcPre, MhcPreBaseTiling, 1000);

inline size_t MhcPreBaseTiling::GetLastRequiredInputIndex() const
{
    return (hasGamma_ == HAS_GAMMA_TRUE) ? GAMMA_INDEX : BIAS_INDEX;
}

ge::graphStatus MhcPreBaseTiling::GetInputShape()
{
    auto xTensor = context_->GetDynamicInputTensor(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto phiTensor = context_->GetDynamicInputTensor(PHI_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, phiTensor);
    auto alphaTensor = context_->GetDynamicInputTensor(ALPHA_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaTensor);
    auto biasTensor = context_->GetDynamicInputTensor(BIAS_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasTensor);

    auto alphaShape = alphaTensor->GetStorageShape();
    OP_CHECK_IF(alphaShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "alpha",
                                             std::to_string(alphaShape.GetDimNum()).c_str(), "1"),
                return ge::GRAPH_FAILED);
    int64_t alphaSize = alphaShape.GetDim(0);
    OP_CHECK_IF(
        alphaSize != 2 && alphaSize != 3,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "alpha", Ops::Base::ToString(alphaShape).c_str(),
                                              "shape must be [2] or [3]"),
        return ge::GRAPH_FAILED);
    hasResi_ = (alphaSize == 3) ? 1 : 0;

    auto gammaTensor = context_->GetDynamicInputTensor(GAMMA_INDEX, 0);
    hasGamma_ = (gammaTensor == nullptr) ? 0 : 1;

    auto xDims = xTensor->GetStorageShape().GetDimNum();
    if (xDims == BSND_DIM_NUM) {
        return ParseBsndFormat(xTensor);
    } else if (xDims == TND_DIM_NUM) {
        return ParseTndFormat(xTensor);
    }

    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xDims).c_str(), "3 or 4");
    return ge::GRAPH_FAILED;
}
ge::graphStatus MhcPreBaseTiling::CheckDescAndShape()
{
    // gamma is optional; skip checks according to hasGamma_.
    size_t maxInputIdx = GetLastRequiredInputIndex();
    for (size_t i = 0; i <= maxInputIdx; i++) {
        auto desc = context_->GetInputDesc(i);
        const std::string descName = std::string(INPUT_NAMES[i]) + " descriptor";
        OP_CHECK_IF(desc == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), descName.c_str(), "nullptr",
                                                          "input descriptor must not be nullptr"),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        const std::string shapeName = std::string(INPUT_NAMES[i]) + " shape";
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), shapeName.c_str(), "nullptr",
                                                          "input shape must not be nullptr"),
                    return ge::GRAPH_FAILED);
    }

    for (size_t i = 0; i <= OUT_H_PRE_INDEX; i++) {
        if (i == OUT_H_RES_INDEX && !hasResi_) {
            continue;
        }
        auto desc = context_->GetOutputDesc(i);
        const std::string descName = std::string(OUTPUT_NAMES[i]) + " descriptor";
        OP_CHECK_IF(desc == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), descName.c_str(), "nullptr",
                                                          "output descriptor must not be nullptr"),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetOutputShape(i);
        const std::string shapeName = std::string(OUTPUT_NAMES[i]) + " shape";
        OP_CHECK_IF(shape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), shapeName.c_str(), "nullptr",
                                                          "output shape must not be nullptr"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MhcPreBaseTiling::CheckShapePositive()
{
    // gamma is optional; skip checks according to hasGamma_.
    size_t maxInputIdx = GetLastRequiredInputIndex();
    for (size_t i = 0; i <= maxInputIdx; i++) {
        auto shape = context_->GetInputShape(i)->GetStorageShape();
        for (size_t j = 0; j < shape.GetDimNum(); j++) {
            const std::string dimName = std::string(INPUT_NAMES[i]) + " shape[" + std::to_string(j) + "]";
            OP_CHECK_IF(shape.GetDim(j) <= 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), dimName.c_str(),
                                                              std::to_string(shape.GetDim(j)).c_str(),
                                                              "dimension must be greater than 0"),
                        return ge::GRAPH_FAILED);
        }
    }

    for (size_t i = 0; i <= OUT_H_PRE_INDEX; i++) {
        if (i == OUT_H_RES_INDEX && !hasResi_) {
            continue;
        }
        auto shape = context_->GetOutputShape(i)->GetStorageShape();
        for (size_t j = 0; j < shape.GetDimNum(); j++) {
            const std::string dimName = std::string(OUTPUT_NAMES[i]) + " shape[" + std::to_string(j) + "]";
            OP_CHECK_IF(shape.GetDim(j) <= 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), dimName.c_str(),
                                                              std::to_string(shape.GetDim(j)).c_str(),
                                                              "dimension must be greater than 0"),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MhcPreBaseTiling::CheckDataType()
{
    auto xDtype = context_->GetInputDesc(X_INDEX)->GetDataType();
    OP_CHECK_IF(
        xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                  "DT_BF16 or DT_FLOAT16"),
        return ge::GRAPH_FAILED);

    auto phiDtype = context_->GetInputDesc(PHI_INDEX)->GetDataType();
    OP_CHECK_IF(phiDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "phi",
                                          ge::TypeUtils::DataTypeToSerialString(phiDtype).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    auto alphaDtype = context_->GetInputDesc(ALPHA_INDEX)->GetDataType();
    OP_CHECK_IF(alphaDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "alpha",
                                          ge::TypeUtils::DataTypeToSerialString(alphaDtype).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    auto biasDtype = context_->GetInputDesc(BIAS_INDEX)->GetDataType();
    OP_CHECK_IF(biasDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "bias",
                                          ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    if (hasGamma_ == HAS_GAMMA_TRUE) {
        auto gammaDtype = context_->GetInputDesc(GAMMA_INDEX)->GetDataType();
        OP_CHECK_IF(gammaDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gamma",
                                              ge::TypeUtils::DataTypeToSerialString(gammaDtype).c_str(), "DT_FLOAT"),
                    return ge::GRAPH_FAILED);
    }

    auto outHinDtype = context_->GetOutputDesc(OUT_H_IN_INDEX)->GetDataType();
    OP_CHECK_IF(outHinDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "hIn",
                                          ge::TypeUtils::DataTypeToSerialString(outHinDtype).c_str(),
                                          ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MhcPreBaseTiling::CheckOutputShapeConsistency()
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    auto xShape = &xShapePtr->GetStorageShape();
    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum == BSND_DIM_NUM) {
        uint64_t b = xShape->GetDim(DIM_0);
        uint64_t s = xShape->GetDim(DIM_1);
        uint64_t n = xShape->GetDim(DIM_2);
        uint64_t d = xShape->GetDim(DIM_3);
        if (CheckBsndOutputShape(b, s, n, d) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        uint64_t t = xShape->GetDim(DIM_0);
        uint64_t n = xShape->GetDim(DIM_1);
        uint64_t d = xShape->GetDim(DIM_2);
        if (CheckTndOutputShape(t, n, d) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MhcPreBaseTiling::CheckBsndOutputShape(uint64_t b, uint64_t s, uint64_t n, uint64_t d)
{
    auto outHinShapePtr = context_->GetOutputShape(OUT_H_IN_INDEX);
    auto outHinShape = &outHinShapePtr->GetStorageShape();
    OP_CHECK_IF(outHinShape->GetDimNum() != DIM_NUM_3 || outHinShape->GetDim(DIM_0) != b ||
                    outHinShape->GetDim(DIM_1) != s || outHinShape->GetDim(DIM_2) != d,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hIn", Ops::Base::ToString(*outHinShape).c_str(),
                                          ShapeToString({b, s, d}).c_str()),
                return ge::GRAPH_FAILED);

    auto outHpostShapePtr = context_->GetOutputShape(OUT_H_POST_INDEX);
    auto outHpostShape = &outHpostShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpostShape->GetDimNum() != DIM_NUM_3 || outHpostShape->GetDim(DIM_0) != b ||
                    outHpostShape->GetDim(DIM_1) != s || outHpostShape->GetDim(DIM_2) != n,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hPost", Ops::Base::ToString(*outHpostShape).c_str(),
                                          ShapeToString({b, s, n}).c_str()),
                return ge::GRAPH_FAILED);

    if (hasResi_) {
        auto outHresShapePtr = context_->GetOutputShape(OUT_H_RES_INDEX);
        auto outHresShape = &outHresShapePtr->GetStorageShape();
        OP_CHECK_IF(
            outHresShape->GetDimNum() != DIM_NUM_4 || outHresShape->GetDim(DIM_0) != b ||
                outHresShape->GetDim(DIM_1) != s || outHresShape->GetDim(DIM_2) != n ||
                outHresShape->GetDim(DIM_3) != n,
            OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hRes", Ops::Base::ToString(*outHresShape).c_str(),
                                      ShapeToString({b, s, n, n}).c_str()),
            return ge::GRAPH_FAILED);
    }

    auto outInvRmsShapePtr = context_->GetOutputShape(OUT_INV_RMS_INDEX);
    auto outInvRmsShape = &outInvRmsShapePtr->GetStorageShape();
    OP_CHECK_IF(outInvRmsShape->GetDimNum() != DIM_NUM_2 || outInvRmsShape->GetDim(DIM_0) != b ||
                    outInvRmsShape->GetDim(DIM_1) != s,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "invRms",
                                          Ops::Base::ToString(*outInvRmsShape).c_str(), ShapeToString({b, s}).c_str()),
                return ge::GRAPH_FAILED);

    auto outMmresShapePtr = context_->GetOutputShape(OUT_H_MIX_INDEX);
    auto outMmresShape = &outMmresShapePtr->GetStorageShape();
    OP_CHECK_IF(outMmresShape->GetDimNum() != DIM_NUM_3 || outMmresShape->GetDim(DIM_0) != b ||
                    outMmresShape->GetDim(DIM_1) != s || outMmresShape->GetDim(DIM_2) != matN_,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hMix", Ops::Base::ToString(*outMmresShape).c_str(),
                                          ShapeToString({b, s, matN_}).c_str()),
                return ge::GRAPH_FAILED);

    auto outHpreShapePtr = context_->GetOutputShape(OUT_H_PRE_INDEX);
    auto outHpreShape = &outHpreShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpreShape->GetDimNum() != DIM_NUM_3 || outHpreShape->GetDim(DIM_0) != b ||
                    outHpreShape->GetDim(DIM_1) != s || outHpreShape->GetDim(DIM_2) != n,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hPre", Ops::Base::ToString(*outHpreShape).c_str(),
                                          ShapeToString({b, s, n}).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MhcPreBaseTiling::CheckTndOutputShape(uint64_t t, uint64_t n, uint64_t d)
{
    auto outHinShapePtr = context_->GetOutputShape(OUT_H_IN_INDEX);
    auto outHinShape = &outHinShapePtr->GetStorageShape();
    OP_CHECK_IF(
        outHinShape->GetDimNum() != DIM_NUM_2 || outHinShape->GetDim(DIM_0) != t || outHinShape->GetDim(DIM_1) != d,
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hIn", Ops::Base::ToString(*outHinShape).c_str(),
                                  ShapeToString({t, d}).c_str()),
        return ge::GRAPH_FAILED);

    auto outHpostShapePtr = context_->GetOutputShape(OUT_H_POST_INDEX);
    auto outHpostShape = &outHpostShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpostShape->GetDimNum() != DIM_NUM_2 || outHpostShape->GetDim(DIM_0) != t ||
                    outHpostShape->GetDim(DIM_1) != n,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hPost", Ops::Base::ToString(*outHpostShape).c_str(),
                                          ShapeToString({t, n}).c_str()),
                return ge::GRAPH_FAILED);

    if (hasResi_) {
        auto outHresShapePtr = context_->GetOutputShape(OUT_H_RES_INDEX);
        auto outHresShape = &outHresShapePtr->GetStorageShape();
        OP_CHECK_IF(
            outHresShape->GetDimNum() != DIM_NUM_3 || outHresShape->GetDim(DIM_0) != t ||
                outHresShape->GetDim(DIM_1) != n || outHresShape->GetDim(DIM_2) != n,
            OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hRes", Ops::Base::ToString(*outHresShape).c_str(),
                                      ShapeToString({t, n, n}).c_str()),
            return ge::GRAPH_FAILED);
    }

    auto outInvRmsShapePtr = context_->GetOutputShape(OUT_INV_RMS_INDEX);
    auto outInvRmsShape = &outInvRmsShapePtr->GetStorageShape();
    OP_CHECK_IF(outInvRmsShape->GetDimNum() != 1 || outInvRmsShape->GetDim(DIM_0) != t,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "invRms",
                                          Ops::Base::ToString(*outInvRmsShape).c_str(), ShapeToString({t}).c_str()),
                return ge::GRAPH_FAILED);

    auto outMmresShapePtr = context_->GetOutputShape(OUT_H_MIX_INDEX);
    auto outMmresShape = &outMmresShapePtr->GetStorageShape();
    OP_CHECK_IF(outMmresShape->GetDimNum() != DIM_NUM_2 || outMmresShape->GetDim(DIM_0) != t ||
                    outMmresShape->GetDim(DIM_1) != matN_,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hMix", Ops::Base::ToString(*outMmresShape).c_str(),
                                          ShapeToString({t, matN_}).c_str()),
                return ge::GRAPH_FAILED);

    auto outHpreShapePtr = context_->GetOutputShape(OUT_H_PRE_INDEX);
    auto outHpreShape = &outHpreShapePtr->GetStorageShape();
    OP_CHECK_IF(
        outHpreShape->GetDimNum() != DIM_NUM_2 || outHpreShape->GetDim(DIM_0) != t || outHpreShape->GetDim(DIM_1) != n,
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "hPre", Ops::Base::ToString(*outHpreShape).c_str(),
                                  ShapeToString({t, n}).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus MhcPreBaseTiling::CheckDataRange()
{
    OP_CHECK_IF(D_ < D_MIN || D_ > D_MAX,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "D", std::to_string(D_).c_str(),
                                          ("[" + std::to_string(D_MIN) + ", " + std::to_string(D_MAX) + "]").c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseBsndFormat(const gert::Tensor *xTensor)
{
    uint64_t batch = xTensor->GetStorageShape().GetDim(INDEX_B_BSND);
    uint64_t sequence = xTensor->GetStorageShape().GetDim(INDEX_S_BSND);
    uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_BSND);
    uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_BSND);

    totalLength_ = batch * sequence;
    matK_ = numsResidual * dimens;
    N_ = numsResidual;
    D_ = dimens;

    return ValidateAndSetTilingParams(xTensor);
}

ge::graphStatus MhcPreBaseTiling::ParseTndFormat(const gert::Tensor *xTensor)
{
    totalLength_ = xTensor->GetStorageShape().GetDim(INDEX_T_TND);
    uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_TND);
    uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_TND);

    matK_ = numsResidual * dimens;
    N_ = numsResidual;
    D_ = dimens;

    return ValidateAndSetTilingParams(xTensor);
}

ge::graphStatus MhcPreBaseTiling::ValidateAndSetTilingParams(const gert::Tensor *xTensor)
{
    auto phiTensor = context_->GetDynamicInputTensor(PHI_INDEX, 0);
    auto phiDims = phiTensor->GetStorageShape().GetDimNum();
    if (phiDims != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "phi", std::to_string(phiDims).c_str(), "2");
        return ge::GRAPH_FAILED;
    }

    bool isValidN = false;
    for (auto validN : N_VALID_VALUES) {
        if (N_ == validN) {
            isValidN = true;
            break;
        }
    }
    if (!isValidN) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "N", std::to_string(N_).c_str(), "4, 6 or 8");
        return ge::GRAPH_FAILED;
    }

    if (D_ % D_ALIGNMENT != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "D", std::to_string(D_).c_str(),
                                              "D must be aligned to 16 elements");
        return ge::GRAPH_FAILED;
    }

    matM_ = totalLength_;

    if (hasResi_) {
        matN_ = N_ * N_ + 2 * N_;
    } else {
        matN_ = 2 * N_;
    }

    uint64_t phiFirstDim = phiTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(phiFirstDim != matN_,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "phi",
                                          Ops::Base::ToString(phiTensor->GetStorageShape()).c_str(),
                                          ShapeToString({matN_, matK_}).c_str()),
                return ge::GRAPH_FAILED);

    auto biasTensor = context_->GetDynamicInputTensor(BIAS_INDEX, 0);
    uint64_t biasFirstDim = biasTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(biasFirstDim != matN_,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "bias",
                                          Ops::Base::ToString(biasTensor->GetStorageShape()).c_str(),
                                          ShapeToString({matN_}).c_str()),
                return ge::GRAPH_FAILED);

    uint64_t phiSecondDim = phiTensor->GetStorageShape().GetDim(1);
    if (phiSecondDim != matK_) {
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "phi",
                                  Ops::Base::ToString(phiTensor->GetStorageShape()).c_str(),
                                  ShapeToString({matN_, matK_}).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseInputAndAttr()
{
    if (InitPlatformMemory() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // GetInputShape fills hasGamma_ and dimension parameters before checks.
    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDescAndShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapePositive() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDataType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDataRange() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseOutputFlags() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ParseEpsAttributes();
}
ge::graphStatus MhcPreBaseTiling::InitPlatformMemory()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "platformInfo", "nullptr",
                                                      "platform info must not be nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim_ = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    aivBlockDim_ = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());

    OP_CHECK_IF(
        aivBlockDim_ != blockDim_ * AIV_AIC_RATIO,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "aivNum", std::to_string(aivBlockDim_).c_str(),
                                              "MhcPre only support aivNum == aicNum * 2"),
        return ge::GRAPH_FAILED);

    ubSize_ = ubSize; // Save UB size for later validation

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseOutputFlags()
{
    auto attrs = context_->GetAttrs();
    auto outFlagPtr = attrs->GetAttrPointer<int64_t>(OUT_FLAG_ATTR_INDEX);
    int64_t outFlag = outFlagPtr != nullptr ? *outFlagPtr : 0;
    if (outFlag != 0 && outFlag != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "out_flag", std::to_string(outFlag).c_str(),
                                              "out_flag must be 0 or 1");
        return ge::GRAPH_FAILED;
    }
    outFlag_ = outFlag == 1;

    if (outFlag_) {
        auto invRmsDesc = context_->GetOutputDesc(OUT_INV_RMS_INDEX);
        auto hMixDesc = context_->GetOutputDesc(OUT_H_MIX_INDEX);
        auto hPreDesc = context_->GetOutputDesc(OUT_H_PRE_INDEX);
        OP_CHECK_IF(invRmsDesc == nullptr || hMixDesc == nullptr || hPreDesc == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "optional outputs", "nullptr",
                                                          "invRms, hMix and hPre are required when out_flag is 1"),
                    return ge::GRAPH_FAILED);
    }

    if (hasResi_) {
        auto hResDesc = context_->GetOutputDesc(OUT_H_RES_INDEX);
        OP_CHECK_IF(hResDesc == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "hRes", "nullptr",
                                                          "output is required when alpha shape is [3]"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseEpsAttributes()
{
    auto attrs = context_->GetAttrs();

    auto normEpsPtr = attrs->GetAttrPointer<float>(1);
    normEps_ = (normEpsPtr != nullptr) ? *normEpsPtr : DEFAULT_NORM_EPS;

    auto hcEpsPtr = attrs->GetAttrPointer<float>(2);
    hcEps_ = (hcEpsPtr != nullptr) ? *hcEpsPtr : DEFAULT_HC_EPS;

    auto implModePtr = attrs->GetAttrPointer<int64_t>(IMPL_MODE_ATTR_INDEX);
    int64_t implMode = (implModePtr != nullptr) ? *implModePtr : IMPL_MODE_FP32;
    if (implMode != IMPL_MODE_FP32 && implMode != IMPL_MODE_HF32) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "op_impl_mode", std::to_string(implMode).c_str(),
                                              "op_impl_mode must be 0 (FP32) or 1 (HF32)");
        return ge::GRAPH_FAILED;
    }
    implMode_ = static_cast<uint32_t>(implMode);

    return ge::GRAPH_SUCCESS;
}

bool MhcPreBaseTiling::CanUseMk() const
{
    bool supportedN = N_ == N_VALID_VALUES[0] || N_ == N_VALID_VALUES[1] || N_ == N_VALID_VALUES[2];
    uint64_t activeCoreNum = std::min<uint64_t>(blockDim_, aivBlockDim_ / 2U);
    // Each M-K AIC owns one M tile; reject shapes that cannot be covered by all active AICs.
    uint64_t maxCoveredM = activeCoreNum * M_K_M_L1_MAX_SIZE;
    return supportedN && totalLength_ <= M_K_MAX_VALIDATED_TOTAL_LENGTH && totalLength_ <= maxCoveredM &&
           matK_ >= M_K_MIN_MAT_K;
}

uint32_t MhcPreBaseTiling::SelectMkStage() const
{
    // Stable msprof measurements favor direct UB-to-L1 staging for large-M
    // FP32 and small-M N=8. GM staging is the robust default elsewhere.
    bool useL1Stage = (implMode_ == IMPL_MODE_FP32 && totalLength_ >= M_K_FP32_L1_MIN_TOTAL_LENGTH) ||
                      (N_ == N_VALID_VALUES[2] && totalLength_ < M_K_N8_L1_MAX_TOTAL_LENGTH);
    return useL1Stage ? M_K_USE_L1_STAGE : M_K_USE_GM_STAGE;
}

ge::graphStatus MhcPreBaseTiling::CalculateBsTiling()
{
    usedCoreNum_ = blockDim_;
    scaleMean_ = 1.0f / static_cast<float>(matK_);
    auto roundUp = [](uint64_t value, uint64_t align) -> uint32_t {
        return static_cast<uint32_t>(((value + align - 1U) / align) * align);
    };
    uint32_t fusionAlign = roundUp(matN_, BASIC_API_FLOAT_ALIGN);
    tilingData_.set_mL1Size(BASIC_API_M_L1_SIZE);
    tilingData_.set_kUbSize(BASIC_API_K_UB_SIZE);
    tilingData_.set_kL1Size(BASIC_API_K_L1_SIZE);
    tilingData_.set_fusionAlign(fusionAlign);

    chunkTSize_ = (((totalLength_ + blockDim_ - 1U) / blockDim_) + CHUNK_T_CALC_FACTOR - 1U) / CHUNK_T_CALC_FACTOR *
                  CHUNK_T_CALC_FACTOR;
    chunkTSize_ = std::min(chunkTSize_, CHUNK_T_MAX);
    v1ChunkDSize_ = V1_CHUNK_D_SIZE;

    // BS user workspace layout:
    //   [0, xStageWorkspaceSize): two FP32 X slots per AIC, [2, blockDim, chunkTSize, 256].
    //   [xStageWorkspaceSize, ...): compact FP32 hMix [M, fusionSize], only when outFlag is false.
    // SYSTEM_WORKSPACE follows the user workspace and is reserved for the runtime.
    size_t xStageWorkspaceSize = static_cast<size_t>(chunkTSize_) * BASIC_API_K_UB_SIZE *
                                 BASIC_API_X_STAGE_BUFFER_COUNT * sizeof(float) * blockDim_;
    size_t hMixWorkspaceSize = outFlag_ ? 0U : static_cast<size_t>(matM_) * static_cast<size_t>(matN_) * sizeof(float);
    workspaceSize_ = xStageWorkspaceSize + hMixWorkspaceSize + SYSTEM_WORKSPACE;
    return CheckUbBufferSize();
}

ge::graphStatus MhcPreBaseTiling::CalculateMkTiling()
{
    auto ceilDiv = [](uint64_t value, uint64_t divisor) -> uint64_t { return (value + divisor - 1U) / divisor; };
    auto roundUp = [&ceilDiv](uint64_t value, uint64_t align) -> uint64_t { return ceilDiv(value, align) * align; };

    uint64_t activeCoreNum = std::min<uint64_t>(blockDim_, aivBlockDim_ / 2U);
    usedCoreNum_ = static_cast<uint32_t>(activeCoreNum);
    scaleMean_ = 1.0f / static_cast<float>(matK_);
    uint64_t minMDim = std::min<uint64_t>(activeCoreNum, ceilDiv(totalLength_, M_K_M_L1_MAX_SIZE));
    uint64_t kDim = activeCoreNum / minMDim;
    // Fill otherwise idle AIC slots without reducing K parallelism. For example,
    // M=3072 uses 16x2 instead of 12x2 on a 32-AIC device.
    uint64_t mDim = activeCoreNum / kDim;
    uint64_t singleCoreM = roundUp(ceilDiv(totalLength_, mDim), AscendC::BLOCK_CUBE);
    // Alignment can make the last planned M block empty. Remove such blocks so
    // every launched M index starts inside the tensor (M=2561: 16 -> 15 blocks).
    mDim = ceilDiv(totalLength_, singleCoreM);
    uint64_t singleCoreK = roundUp(ceilDiv(matK_, kDim), M_K_SPLIT_ALIGN);
    // Keep the Cube accumulation order independent of N and the optional residual
    // segment. The kernel derives the same decision from multCoreSplitKSize.
    bool useSequentialKPartials = singleCoreK >= M_K_SEQUENTIAL_PARTIAL_THRESHOLD;
    uint64_t workspaceGroupK = useSequentialKPartials ? ceilDiv(matK_, M_K_SEQUENTIAL_PARTIAL_K) : 0U;
    uint64_t splitK = useSequentialKPartials ? ceilDiv(workspaceGroupK, kDim) * M_K_SEQUENTIAL_PARTIAL_K : singleCoreK;
    uint64_t actualKBlockNum = ceilDiv(matK_, splitK);
    uint64_t mL1Size = std::min<uint64_t>(M_K_M_L1_MAX_SIZE, singleCoreM);
    uint64_t fusionAlign = roundUp(matN_, BASIC_API_FLOAT_ALIGN);
    uint64_t kL1Size = std::min<uint64_t>(M_K_A_L1_ELEMENT_COUNT / mL1Size, M_K_B_L1_ELEMENT_COUNT / fusionAlign);
    kL1Size = std::min<uint64_t>(kL1Size, M_K_K_L1_MAX_SIZE);
    kL1Size = std::max<uint64_t>(M_K_K_L1_ALIGN, kL1Size / M_K_K_L1_ALIGN * M_K_K_L1_ALIGN);
    // Keep every L1 tile inside one sequential partial; only the final partial may be shorter.
    if (useSequentialKPartials) {
        while (M_K_SEQUENTIAL_PARTIAL_K % kL1Size != 0U) {
            kL1Size -= M_K_K_L1_ALIGN;
        }
    }

    tilingData_.set_cubeBlockDimM(static_cast<uint32_t>(mDim));
    tilingData_.set_cubeBlockDimK(static_cast<uint32_t>(actualKBlockNum));
    tilingData_.set_multCoreSplitKSize(static_cast<uint32_t>(splitK));
    tilingData_.set_mL1Size(static_cast<uint32_t>(mL1Size));
    tilingData_.set_kL1Size(static_cast<uint32_t>(kL1Size));
    tilingData_.set_kUbSize(static_cast<uint32_t>(kL1Size));
    tilingData_.set_fusionAlign(static_cast<uint32_t>(fusionAlign));
    // Keep enough rows per AIV to amortize setup while distributing all rows across the available AIVs.
    uint64_t stage2RowsPerCore = std::min<uint64_t>(
        totalLength_, std::max<uint64_t>(M_K_MIN_STAGE2_ROWS_PER_CORE, ceilDiv(totalLength_, aivBlockDim_)));
    uint64_t stage2UsedAivNum = std::min<uint64_t>(aivBlockDim_, ceilDiv(totalLength_, stage2RowsPerCore));
    tilingData_.set_stage2UsedAivNum(static_cast<uint32_t>(stage2UsedAivNum));
    tilingData_.set_stage2RowsPerCore(static_cast<uint32_t>(stage2RowsPerCore));

    uint64_t mmWorkspaceGroupK = useSequentialKPartials ? workspaceGroupK : actualKBlockNum;
    // M-K user workspace layout, with every region boundary aligned to M_K_WORKSPACE_ALIGN:
    //   [mmOffset, rmsOffset): FP32 matmul partials [mmWorkspaceGroupK, M, fusionSize].
    //   [rmsOffset, finalOffset): FP32 RMS partials [actualKBlockNum, M].
    //   [finalOffset, ...): optional ping-pong X staging
    //       [mDim * actualKBlockNum, 2, mL1Size, kL1Size], used only by the GM staging path.
    // Part2 reads the first two regions into UB; its reduced hMix stays in UB or goes to the optional output.
    // SYSTEM_WORKSPACE follows the user workspace and is reserved for the runtime.
    size_t partialMmBytes = static_cast<size_t>(mmWorkspaceGroupK) * totalLength_ * matN_ * sizeof(float);
    size_t partialRmsBytes = static_cast<size_t>(actualKBlockNum) * totalLength_ * sizeof(float);
    size_t mmOffset = 0;
    size_t rmsOffset = roundUp(partialMmBytes, M_K_WORKSPACE_ALIGN);
    size_t finalOffset = roundUp(rmsOffset + partialRmsBytes, M_K_WORKSPACE_ALIGN);
    tilingData_.set_mkWorkspaceMmOffset(static_cast<uint32_t>(mmOffset));
    tilingData_.set_mkWorkspaceRmsOffset(static_cast<uint32_t>(rmsOffset));
    tilingData_.set_mkWorkspaceFinalOffset(static_cast<uint32_t>(finalOffset));
    uint32_t mkStage = SelectMkStage();
    tilingData_.set_mkUseGmStage(mkStage);

    chunkTSize_ = static_cast<uint32_t>(mL1Size);
    v1ChunkDSize_ = V1_CHUNK_D_SIZE;
    size_t xStageBytes = mkStage == M_K_USE_L1_STAGE ?
                             0U :
                             static_cast<size_t>(mDim) * actualKBlockNum * BASIC_API_X_STAGE_BUFFER_COUNT * mL1Size *
                                 kL1Size * sizeof(float);
    workspaceSize_ = finalOffset + xStageBytes + SYSTEM_WORKSPACE;
    return CheckUbBufferSize();
}

ge::graphStatus MhcPreBaseTiling::CalculateNdTiling()
{
    usedCoreNum_ = blockDim_;
    scaleMean_ = 1.0f / static_cast<float>(matK_);
    chunkTSize_ = DECODE_CHUNK_T_SIZE;
    v1ChunkDSize_ = V1_CHUNK_D_SIZE;

    // ND user workspace layout:
    //   [0, xFloatWorkspaceSize): complete FP32 X [M, K], padded to DECODE_WORKSPACE_ALIGN.
    //   [xFloatWorkspaceSize, ...): FP32 matmul partials [mmResBlockNum, M, fusionSize].
    // AIV reduces the second region in UB, so no separate final hMix workspace is required.
    // SYSTEM_WORKSPACE follows the user workspace and is reserved for the runtime.
    size_t xFloatWorkspaceSizeRaw = static_cast<size_t>(matM_) * static_cast<size_t>(matK_) * sizeof(float);
    size_t xFloatWorkspaceSize =
        (xFloatWorkspaceSizeRaw + DECODE_WORKSPACE_ALIGN - 1U) / DECODE_WORKSPACE_ALIGN * DECODE_WORKSPACE_ALIGN;
    uint64_t chunkNd = (matK_ + blockDim_ - 1U) / blockDim_;
    uint64_t mmResBlockNum = (matK_ + chunkNd - 1U) / chunkNd;
    size_t mmResWorkspaceSize =
        static_cast<size_t>(mmResBlockNum) * static_cast<size_t>(matM_) * static_cast<size_t>(matN_) * sizeof(float);
    workspaceSize_ = xFloatWorkspaceSize + mmResWorkspaceSize + SYSTEM_WORKSPACE;
    return CheckUbBufferSize();
}

void MhcPreBaseTiling::FillTilingData()
{
    tilingData_.set_coreNum(usedCoreNum_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_nD(matK_);
    tilingData_.set_fusionSize(matN_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_chunkTSize(chunkTSize_);
    tilingData_.set_v1ChunkDSize(v1ChunkDSize_);
    tilingData_.set_outFlag(outFlag_);
    tilingData_.set_hasGamma(hasGamma_);
    tilingData_.set_hasResi(hasResi_);
    tilingData_.set_implMode(implMode_);

    tilingData_.set_scaleMean(scaleMean_);
}

ge::graphStatus MhcPreBaseTiling::TilingProcess()
{
    // Batch-consistency level 3 requires the BS accumulation order. Other modes prefer M-K when the shape is
    // supported, then fall back to the ND/BS threshold policy.
    if (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL) {
        tilingMode_ = TilingMode::SPLIT_BS;
    } else if (CanUseMk()) {
        tilingMode_ = TilingMode::SPLIT_M_K;
    } else {
        tilingMode_ = totalLength_ <= DECODE_BS_THRESHOLD ? TilingMode::SPLIT_ND : TilingMode::SPLIT_BS;
    }

    switch (tilingMode_) {
        case TilingMode::SPLIT_BS:
            return CalculateBsTiling();
        case TilingMode::SPLIT_ND:
            return CalculateNdTiling();
        case TilingMode::SPLIT_M_K:
            return CalculateMkTiling();
        default:
            return ge::GRAPH_FAILED;
    }
}

ge::graphStatus MhcPreBaseTiling::CheckUbBufferSize()
{
    size_t fixedBufferSize = 0;
    size_t dynamicBufferSize = 0;
    const size_t floatSize = sizeof(float);
    const size_t doubleBufferCount = 2;

    if (tilingMode_ == TilingMode::SPLIT_BS || tilingMode_ == TilingMode::SPLIT_M_K) {
        fixedBufferSize = BASIC_API_BUFFER_POOL0_SIZE + BASIC_API_BUFFER_POOL1_SIZE;
    } else {
        fixedBufferSize = 80 * 1024 * doubleBufferCount + 16 * 1024 * doubleBufferCount + 20 * 1024;
    }

    dynamicBufferSize = static_cast<size_t>(matN_) * floatSize * 2;

    if (outFlag_) {
        size_t invRmsSize =
            (tilingMode_ == TilingMode::SPLIT_ND) ? ((chunkTSize_ + 1) / 2) * floatSize : (chunkTSize_ / 2) * floatSize;
        dynamicBufferSize += invRmsSize;
    }

    if (hasGamma_) {
        size_t gammaBufferLength =
            (tilingMode_ == TilingMode::SPLIT_ND) ? SPLIT_ND_GAMMA_BUFFER_LENGTH : SPLIT_BS_GAMMA_BUFFER_LENGTH;
        dynamicBufferSize += gammaBufferLength * floatSize;
    }

    size_t totalUbRequired = fixedBufferSize + dynamicBufferSize;
    if (totalUbRequired > ubSize_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "required UB size", std::to_string(totalUbRequired).c_str(),
            ("must not exceed available UB size " + std::to_string(ubSize_) + " bytes").c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x descriptor", "nullptr",
                                              "input descriptor must not be nullptr");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcPreBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "BlockDim: [%u]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "TotalLength: [%lu]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "ND: [%lu]", tilingData_.get_nD());
    OP_LOGD(context_->GetNodeName(), "FusionSize: [%lu]", tilingData_.get_fusionSize());
    OP_LOGD(context_->GetNodeName(), "N: [%lu]", tilingData_.get_N());
    OP_LOGD(context_->GetNodeName(), "D: [%lu]", tilingData_.get_D());
    OP_LOGD(context_->GetNodeName(), "NormEps: [%e]", tilingData_.get_normEps());
    OP_LOGD(context_->GetNodeName(), "HcEps: [%e]", tilingData_.get_hcEps());
    OP_LOGD(context_->GetNodeName(), "OutFlag: [%d]", tilingData_.get_outFlag());
    OP_LOGD(context_->GetNodeName(), "HasGamma: [%u]", tilingData_.get_hasGamma());
    OP_LOGD(context_->GetNodeName(), "HasResi: [%u]", tilingData_.get_hasResi());
    OP_LOGD(context_->GetNodeName(), "ChunkTSize: [%u]", tilingData_.get_chunkTSize());
    OP_LOGD(context_->GetNodeName(), "V1ChunkDSize: [%u]", tilingData_.get_v1ChunkDSize());
}

uint64_t MhcPreBaseTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(tilingMode_), static_cast<uint64_t>(hasResi_ ? 0 : 1));
}

ge::graphStatus MhcPreBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tilingData size",
                                                      std::to_string(tilingData_.GetDataSize()).c_str(),
                                                      "size must be aligned to 8 bytes"),
                return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(SCHEDULE_MODE);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "Workspaces is null"),
                return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
