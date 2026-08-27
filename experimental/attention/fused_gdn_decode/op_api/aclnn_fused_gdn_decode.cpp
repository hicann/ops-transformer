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
 * \file aclnn_fused_gdn_decode.cpp
 * \brief
 */

#include "aclnn_fused_gdn_decode.h"
#include "fused_gdn_decode_l0.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t MIXED_RANK = 2;
constexpr size_t GATE_RANK = 2;
constexpr size_t PARAM_RANK = 1;
constexpr size_t STATE_RANK = 4;
constexpr size_t OUT_RANK = 4;
constexpr uint32_t MIN_SUPPORTED_K = 64;
constexpr uint32_t K_ALIGN_ELEMS = 16;
constexpr uint32_t FP32_ELEMS_PER_BLOCK = 8;
constexpr uint32_t MAX_DATA_COPY_PAD_BYTES = 32;
constexpr uint32_t MAX_SUPPORTED_K =
    (std::numeric_limits<uint8_t>::max() * FP32_ELEMS_PER_BLOCK / K_ALIGN_ELEMS) * K_ALIGN_ELEMS;

struct Params {
    const aclTensor *mixedQkv{nullptr};
    const aclTensor *a{nullptr};
    const aclTensor *b{nullptr};
    const aclTensor *aLog{nullptr};
    const aclTensor *dtBias{nullptr};
    aclTensor *stateRef{nullptr};
    const aclTensor *ssmStateIndices{nullptr};
    float scale{1.0f};
    float softplusThreshold{20.0f};
    aclTensor *out{nullptr};
};

bool IsActivationType(DataType dtype)
{
    return dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16;
}

bool IsPositiveU32(int64_t value)
{
    return value > 0 && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool CheckNotNull(const Params &params)
{
    OP_CHECK_NULL(params.mixedQkv, return false);
    OP_CHECK_NULL(params.a, return false);
    OP_CHECK_NULL(params.b, return false);
    OP_CHECK_NULL(params.aLog, return false);
    OP_CHECK_NULL(params.dtBias, return false);
    OP_CHECK_NULL(params.stateRef, return false);
    OP_CHECK_NULL(params.ssmStateIndices, return false);
    OP_CHECK_NULL(params.out, return false);
    return true;
}

bool CheckDtype(const Params &params)
{
    const DataType inputDtype = params.mixedQkv->GetDataType();
    const DataType stateDtype = params.stateRef->GetDataType();
    OP_CHECK(IsActivationType(inputDtype), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mixedQkv must be BF16 or FP16."),
             return false);
    OP_CHECK(params.a->GetDataType() == inputDtype && params.b->GetDataType() == inputDtype &&
                 params.dtBias->GetDataType() == inputDtype && params.out->GetDataType() == inputDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mixedQkv, a, b, dtBias and out must have the same dtype."),
             return false);
    OP_CHECK(params.aLog->GetDataType() == DataType::DT_FLOAT, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aLog must be FP32."),
             return false);
    OP_CHECK(stateDtype == DataType::DT_FLOAT || stateDtype == inputDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stateRef must be FP32 or have the same dtype as mixedQkv."),
             return false);
    OP_CHECK(params.ssmStateIndices->GetDataType() == DataType::DT_INT32,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ssmStateIndices must be INT32."), return false);
    return true;
}

bool CheckShape(const Params &params)
{
    const auto &mixedShape = params.mixedQkv->GetViewShape();
    const auto &aShape = params.a->GetViewShape();
    const auto &bShape = params.b->GetViewShape();
    const auto &aLogShape = params.aLog->GetViewShape();
    const auto &dtBiasShape = params.dtBias->GetViewShape();
    const auto &stateShape = params.stateRef->GetViewShape();
    const auto &stateIndicesShape = params.ssmStateIndices->GetViewShape();
    const auto &outShape = params.out->GetViewShape();

    OP_CHECK(mixedShape.GetDimNum() == MIXED_RANK && aShape.GetDimNum() == GATE_RANK &&
                 bShape.GetDimNum() == GATE_RANK && aLogShape.GetDimNum() == PARAM_RANK &&
                 dtBiasShape.GetDimNum() == PARAM_RANK && stateShape.GetDimNum() == STATE_RANK &&
                 stateIndicesShape.GetDimNum() == PARAM_RANK && outShape.GetDimNum() == OUT_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input or output rank is invalid."), return false);

    const int64_t batch = mixedShape.GetDim(0);
    const int64_t mixedDim = mixedShape.GetDim(1);
    const int64_t stateSlots = stateShape.GetDim(0);
    const int64_t hv = stateShape.GetDim(1);
    const int64_t v = stateShape.GetDim(2);
    const int64_t k = stateShape.GetDim(3);
    OP_CHECK(IsPositiveU32(batch) && IsPositiveU32(mixedDim) && IsPositiveU32(stateSlots) && IsPositiveU32(hv) &&
                 IsPositiveU32(v) && IsPositiveU32(k),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "core tensor dimensions must be in (0, UINT32_MAX]."), return false);
    OP_CHECK(
        k >= MIN_SUPPORTED_K && k <= MAX_SUPPORTED_K,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "K must be in [%u, %u], but got %ld.", MIN_SUPPORTED_K, MAX_SUPPORTED_K, k),
        return false);

    const uint64_t alignedK =
        (static_cast<uint64_t>(k) / K_ALIGN_ELEMS + static_cast<uint64_t>(k % K_ALIGN_ELEMS != 0)) * K_ALIGN_ELEMS;
    const uint64_t stateElementBytes =
        params.stateRef->GetDataType() == DataType::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    OP_CHECK((alignedK - static_cast<uint64_t>(k)) * stateElementBytes <= MAX_DATA_COPY_PAD_BYTES,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "state row padding exceeds the DataCopyPad limit."), return false);

    const uint64_t valueDim = static_cast<uint64_t>(hv) * static_cast<uint64_t>(v);
    OP_CHECK(valueDim < static_cast<uint64_t>(mixedDim),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mixedQkv last dimension is too small for V."), return false);
    const uint64_t qkDim = static_cast<uint64_t>(mixedDim) - valueDim;
    const uint64_t qkRowWidth = 2ULL * static_cast<uint64_t>(k);
    OP_CHECK(qkDim % qkRowWidth == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mixedQkv last dimension is inconsistent with K, Hv and V."),
             return false);
    const uint64_t h = qkDim / qkRowWidth;
    OP_CHECK(h > 0 && static_cast<uint64_t>(hv) % h == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Hv must be divisible by the derived H."), return false);
    OP_CHECK(static_cast<uint64_t>(batch) * static_cast<uint64_t>(hv) <= std::numeric_limits<uint32_t>::max(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "B * Hv must not exceed UINT32_MAX."), return false);

    OP_CHECK(aShape.GetDim(0) == batch && aShape.GetDim(1) == hv && bShape.GetDim(0) == batch &&
                 bShape.GetDim(1) == hv && aLogShape.GetDim(0) == hv && dtBiasShape.GetDim(0) == hv &&
                 stateIndicesShape.GetDim(0) == batch,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input shapes are inconsistent."), return false);
    OP_CHECK(
        outShape.GetDim(0) == batch && outShape.GetDim(1) == 1 && outShape.GetDim(2) == hv && outShape.GetDim(3) == v,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out must have shape [B, 1, HV, V]."), return false);
    return true;
}

aclnnStatus CheckParams(const Params &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(IsContiguous(params.stateRef), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stateRef must be contiguous."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(std::isfinite(params.scale) && std::isfinite(params.softplusThreshold),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale and softplusThreshold must be finite."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnFusedGdnDecodeGetWorkspaceSize(const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
                                                const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
                                                const aclTensor *ssmStateIndices, float scale, float softplusThreshold,
                                                aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedGdnDecode,
                   DFX_IN(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scale, softplusThreshold),
                   DFX_OUT(out, stateRef));
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const Params params{mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scale, softplusThreshold, out};
    const aclnnStatus checkStatus = CheckParams(params);
    CHECK_RET(checkStatus == ACLNN_SUCCESS, checkStatus);

    auto mixedQkvContiguous = l0op::Contiguous(mixedQkv, uniqueExecutor.get());
    auto aContiguous = l0op::Contiguous(a, uniqueExecutor.get());
    auto bContiguous = l0op::Contiguous(b, uniqueExecutor.get());
    auto aLogContiguous = l0op::Contiguous(aLog, uniqueExecutor.get());
    auto dtBiasContiguous = l0op::Contiguous(dtBias, uniqueExecutor.get());
    auto stateIndicesContiguous = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    CHECK_RET(mixedQkvContiguous != nullptr && aContiguous != nullptr && bContiguous != nullptr &&
                  aLogContiguous != nullptr && dtBiasContiguous != nullptr && stateIndicesContiguous != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto outRet =
        l0op::FusedGdnDecode(mixedQkvContiguous, aContiguous, bContiguous, aLogContiguous, dtBiasContiguous, stateRef,
                             stateIndicesContiguous, scale, softplusThreshold, uniqueExecutor.get());
    CHECK_RET(outRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto copyRet = l0op::ViewCopy(outRet, out, uniqueExecutor.get());
    CHECK_RET(copyRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedGdnDecode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedGdnDecode);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
