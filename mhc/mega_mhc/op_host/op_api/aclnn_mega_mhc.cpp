/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_mega_mhc.h"
#include "mega_mhc.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "log/log.h"

namespace {
constexpr const char* ACLNN_OP_NAME = "aclnnMegaMhcGetWorkspaceSize";
constexpr size_t INPUT_COUNT = 11;
constexpr size_t OUTPUT_COUNT = 7;
constexpr size_t ACTIVE_OUTPUTS[] = {0, 1, 2, 4};
constexpr size_t RESERVED_OUTPUTS[] = {3, 5, 6};
constexpr size_t MID_INDEX = 10;
constexpr size_t PRE_MIX_INDEX = 8;
constexpr size_t RMS0_INDEX = 9;
constexpr int64_t HEADS = 4;
constexpr int64_t PROJECTION = 24;
using Inputs = std::array<const aclTensor*, INPUT_COUNT>;
using Outputs = std::array<const aclTensor*, OUTPUT_COUNT>;

/** Format expected dimensions for diagnostics. */
std::string ShapeToString(const std::vector<int64_t>& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return result + "]";
}

/** Validate tensor shape, dtype and ND format. */
bool CheckTensor(const aclTensor* tensor, const char* name, const std::vector<int64_t>& shape, op::DataType dtype)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, name, "nullptr", "tensor must not be null");
        return false;
    }
    if (tensor->GetDataType() != dtype) {
        OP_LOGE_FOR_INVALID_DTYPE(ACLNN_OP_NAME, name, op::ToString(tensor->GetDataType()).GetString(),
                                  op::ToString(dtype).GetString());
        return false;
    }
    if (op::IsPrivateFormat(tensor->GetViewFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ACLNN_OP_NAME, name, op::ToString(tensor->GetViewFormat()).GetString(),
                                               "tensor must use a public format");
        return false;
    }
    const auto& actual = tensor->GetViewShape();
    if (actual.GetDimNum() != shape.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(ACLNN_OP_NAME, name, std::to_string(actual.GetDimNum()).c_str(),
                                     std::to_string(shape.size()).c_str());
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (actual.GetDim(axis) != shape[axis]) {
            std::vector<int64_t> actualShape;
            for (size_t i = 0; i < actual.GetDimNum(); ++i) {
                actualShape.push_back(actual.GetDim(i));
            }
            OP_LOGE_FOR_INVALID_SHAPE(ACLNN_OP_NAME, name, ShapeToString(actualShape).c_str(),
                                      ShapeToString(shape).c_str());
            return false;
        }
    }
    return true;
}

/** Validate supported inputs, attributes and active/reserved outputs. */
aclnnStatus CheckContract(const Inputs& inputs, const Outputs& outputs, double normEps, double mhcPreNormEps,
                          double hcEps, int64_t mode, int64_t hcMult, double hcPostScale, double sinkhornEps,
                          int64_t numSinkhornIters, double rmsNormScale, int64_t outputMode)
{
    if (outputMode < 0 || outputMode > 2) {
        OP_LOGE_FOR_INVALID_VALUE(ACLNN_OP_NAME, "outputMode", std::to_string(outputMode).c_str(), "0, 1 or 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outputMode != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "outputMode", std::to_string(outputMode).c_str(),
                                              "MX Quant output modes 1 and 2 are reserved in this version");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < MID_INDEX; ++i) {
        if (i != PRE_MIX_INDEX && i != RMS0_INDEX) {
            CHECK_RET(inputs[i] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        }
    }
    if (inputs[PRE_MIX_INDEX] != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "shiftedPrevMixOptional", "non-null",
                                              "reserved input must be null in this version");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto index : RESERVED_OUTPUTS) {
        CHECK_RET(outputs[index] == nullptr, ACLNN_ERR_PARAM_INVALID);
    }
    for (const auto index : ACTIVE_OUTPUTS) {
        CHECK_RET(outputs[index] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    CHECK_RET(hcMult == HEADS && numSinkhornIters == 0 && static_cast<float>(sinkhornEps) == 1.0e-6f &&
                  rmsNormScale == 1.0 && std::isfinite(static_cast<float>(hcPostScale)),
              ACLNN_ERR_PARAM_INVALID);
    const float norm = static_cast<float>(normEps);
    const float preNorm = static_cast<float>(mhcPreNormEps);
    const float hc = static_cast<float>(hcEps);
    CHECK_RET(std::isfinite(norm) && norm > 0 && std::isfinite(preNorm) && preNorm > 0 && std::isfinite(hc) && hc > 0 &&
                  (mode == 0 || mode == 1),
              ACLNN_ERR_PARAM_INVALID);
    const auto& shape = inputs[0]->GetViewShape();
    const auto rank = shape.GetDimNum();
    CHECK_RET(rank == 2 || rank == 3, ACLNN_ERR_PARAM_INVALID);
    const int64_t d = shape.GetDim(rank - 1);
    CHECK_RET(d >= 16 && d <= 16384 && d % 16 == 0, ACLNN_ERR_PARAM_INVALID);
    const auto dtype = inputs[0]->GetDataType();
    CHECK_RET(dtype == op::DataType::DT_FLOAT16 || dtype == op::DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID);
    std::vector<int64_t> leading;
    uint64_t rows = 1;
    for (size_t axis = 0; axis + 1 < rank; ++axis) {
        const int64_t dim = shape.GetDim(axis);
        CHECK_RET(dim > 0 && rows <= UINT32_MAX / static_cast<uint64_t>(dim), ACLNN_ERR_PARAM_INVALID);
        rows *= dim;
        leading.push_back(dim);
    }
    auto xShape = leading;
    xShape.push_back(d);
    auto postShape = leading;
    postShape.push_back(HEADS);
    auto resShape = postShape;
    resShape.push_back(HEADS);
    auto x2Shape = postShape;
    x2Shape.push_back(d);
    postShape.push_back(1);
    const std::array<std::vector<int64_t>, INPUT_COUNT> shapes = {xShape,
                                                                  x2Shape,
                                                                  postShape,
                                                                  std::vector<int64_t>{PROJECTION, HEADS * d},
                                                                  std::vector<int64_t>{3},
                                                                  std::vector<int64_t>{PROJECTION},
                                                                  std::vector<int64_t>{d},
                                                                  resShape,
                                                                  postShape,
                                                                  std::vector<int64_t>{d},
                                                                  std::vector<int64_t>{HEADS * d}};
    const char* names[] = {"x",
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
    for (size_t i = 0; i < INPUT_COUNT; ++i) {
        if (i == PRE_MIX_INDEX || ((i == RMS0_INDEX || i == MID_INDEX) && inputs[i] == nullptr)) {
            continue;
        }
        CHECK_RET(CheckTensor(inputs[i], names[i], shapes[i], i < 2 ? dtype : op::DataType::DT_FLOAT),
                  ACLNN_ERR_PARAM_INVALID);
    }
    const size_t outputSources[] = {1, 2, 7, 2, 0, 0, 0};
    const char* outputNames[] = {"new_residual", "new_post_mix", "new_comb_res_mix", "new_prev_mix", "y",
                                 "y_quant",      "y_scale"};
    for (const auto i : ACTIVE_OUTPUTS) {
        CHECK_RET(CheckTensor(outputs[i], outputNames[i], shapes[outputSources[i]],
                              i == 0 || i == 4 ? dtype : op::DataType::DT_FLOAT),
                  ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

/** Materialize strided inputs, launch to contiguous outputs, then ViewCopy back to the caller's output views. */
aclnnStatus BuildExecution(Inputs inputs, const Outputs& outputs, double normEps, double mhcPreNormEps, double hcEps,
                           int64_t mode, int64_t hcMult, double hcPostScale, double sinkhornEps,
                           int64_t numSinkhornIters, double rmsNormScale, int64_t outputMode, uint64_t* workspaceSize,
                           aclOpExecutor** executor)
{
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const auto status = CheckContract(inputs, outputs, normEps, mhcPreNormEps, hcEps, mode, hcMult, hcPostScale,
                                      sinkhornEps, numSinkhornIters, rmsNormScale, outputMode);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    auto owner = CREATE_EXECUTOR();
    CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    for (auto& input : inputs) {
        if (input != nullptr) {
            input = l0op::Contiguous(input, owner.get());
            CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    Outputs contiguousOutputs{};
    for (const auto i : ACTIVE_OUTPUTS) {
        contiguousOutputs[i] =
            owner->AllocTensor(outputs[i]->GetViewShape(), outputs[i]->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(contiguousOutputs[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // Keep all seven argument slots even though the public reserved outputs are null.
    const op::DataType reservedTypes[] = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT8_E4M3FN,
                                          op::DataType::DT_FLOAT8_E8M0};
    for (size_t i = 0; i < 3; ++i) {
        contiguousOutputs[RESERVED_OUTPUTS[i]] =
            owner->AllocTensor(op::Shape{0}, reservedTypes[i], op::Format::FORMAT_ND);
        CHECK_RET(contiguousOutputs[RESERVED_OUTPUTS[i]] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const bool launched = l0op::MegaMhc(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9],
        inputs[10], normEps, mhcPreNormEps, hcEps, mode, hcMult, hcPostScale, sinkhornEps, numSinkhornIters,
        rmsNormScale, outputMode, contiguousOutputs[0], contiguousOutputs[1], contiguousOutputs[2],
        contiguousOutputs[3], contiguousOutputs[4], contiguousOutputs[5], contiguousOutputs[6], owner.get());
    CHECK_RET(launched, ACLNN_ERR_INNER_NULLPTR);
    for (const auto i : ACTIVE_OUTPUTS) {
        CHECK_RET(l0op::ViewCopy(contiguousOutputs[i], outputs[i], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

/** Build the execution graph and report its workspace; no kernel executes in this phase. */
aclnnStatus aclnnMegaMhcGetWorkspaceSize(
    const aclTensor* x, const aclTensor* residual, const aclTensor* postMix, const aclTensor* fn,
    const aclTensor* mixScales, const aclTensor* mixBases, const aclTensor* rmsNorm1Weight,
    const aclTensor* combResMixOptional, const aclTensor* shiftedPrevMixOptional,
    const aclTensor* rmsNorm0WeightOptional, const aclTensor* rmsNormMidWeightOptional, double rmsNormEps,
    double hcNormEps, double hcPreEps, int64_t opImplMode, int64_t hcMult, double hcPostScale, double sinkhornEps,
    int64_t numSinkhornIters, double rmsNormScale, int64_t outputMode, aclTensor* newResidualOut,
    aclTensor* newPostMixOut, aclTensor* newCombResMixOut, aclTensor* newPrevMixOut, aclTensor* yOut,
    aclTensor* yQuantOut, aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnMegaMhc,
        DFX_IN(x, residual, postMix, fn, mixScales, mixBases, rmsNorm1Weight, combResMixOptional,
               shiftedPrevMixOptional, rmsNorm0WeightOptional, rmsNormMidWeightOptional, rmsNormEps, hcNormEps,
               hcPreEps, opImplMode, hcMult, hcPostScale, sinkhornEps, numSinkhornIters, rmsNormScale, outputMode),
        DFX_OUT(newResidualOut, newPostMixOut, newCombResMixOut, newPrevMixOut, yOut, yQuantOut, yScaleOut));
    return BuildExecution({x, residual, postMix, fn, mixScales, mixBases, rmsNorm1Weight, combResMixOptional,
                           shiftedPrevMixOptional, rmsNorm0WeightOptional, rmsNormMidWeightOptional},
                          {newResidualOut, newPostMixOut, newCombResMixOut, newPrevMixOut, yOut, yQuantOut, yScaleOut},
                          rmsNormEps, hcNormEps, hcPreEps, opImplMode, hcMult, hcPostScale, sinkhornEps,
                          numSinkhornIters, rmsNormScale, outputMode, workspaceSize, executor);
}

/** Execute the prepared graph on the caller's stream using its allocated workspace. */
aclnnStatus aclnnMegaMhc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMegaMhc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
