/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <initializer_list>
#include <tuple>

#include "aclnn_moe_token_permute_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_permute.h" // 该文件为自动生成，在build/autogen/inner路径下
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/slice.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_init_routing_v2/op_host/op_api/moe_init_routing_v2.h"
#include "moe/moe_init_routing_v3/op_host/op_api/moe_init_routing_v3.h"

using namespace op;

namespace MoeTokenPermuteV2Check {

static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                             DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X_V3 = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                                DataType::DT_FLOAT, DataType::DT_INT8};
static const std::initializer_list<op::DataType> MOE_QUANT_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16,
                                                                                   DataType::DT_BF16};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE = {DataType::DT_INT32,
                                                                                              DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};
static constexpr int64_t QUANT_MODE_NONE = -1;
static constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
static constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
static constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
static constexpr int64_t EXPERT_NUM = 10240;
static constexpr int64_t EXPERT_TOKENS_NUM_TYPE_COUNT = 1;
static constexpr int64_t ROW_IDX_TYPE_GATHER = 0;
static constexpr int64_t TOKENS_REQUIRED_DIM = 2;
static constexpr int64_t INDICES_MIN_DIM = 1;
static constexpr int64_t INDICES_MAX_DIM = 2;

static inline bool CheckNotNull(const aclTensor *tokens, const aclTensor *indices, const aclTensor *permuteTokensOut,
                                const aclTensor *sortedIndicesOut, const aclTensor *expandedScaleOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(sortedIndicesOut, return false);
    OP_CHECK_NULL(expandedScaleOut, return false);
    return true;
}

static inline bool CheckDtypeValidRegbase(const aclTensor *tokens, const aclTensor *indices,
                                          const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut)
{
    if (tokens != nullptr && tokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (indices != nullptr && indices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (permuteTokensOut != nullptr && permuteTokensOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, MOE_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(tokens, permuteTokensOut, return false);
    }
    if (sortedIndicesOut != nullptr && sortedIndicesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static inline bool CheckShapeValid(const aclTensor *tokens, const aclTensor *indices)
{
    auto tokensDimNum = tokens->GetViewShape().GetDimNum();
    OP_CHECK(tokensDimNum == TOKENS_REQUIRED_DIM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of tokens should be 2, but got %ld.",
                     static_cast<int64_t>(tokensDimNum)),
             return false);

    auto indicesDimNum = indices->GetViewShape().GetDimNum();
    OP_CHECK(indicesDimNum == INDICES_MIN_DIM || indicesDimNum == INDICES_MAX_DIM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of indices should be 1 or 2, but got %ld.",
                     static_cast<int64_t>(indicesDimNum)),
             return false);
    return true;
}

static inline bool CheckDtypeValidV2Regbase(const aclTensor *tokens, const aclTensor *indices, int64_t quantMode,
                                            const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                            const aclTensor *expandedScaleOut)
{
    if (indices != nullptr && indices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (sortedIndicesOut != nullptr && sortedIndicesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }

    if (quantMode == QUANT_MODE_NONE) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_DTYPE_SUPPORT_LIST_X_V3, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, MOE_DTYPE_SUPPORT_LIST_X_V3, return false);
        OP_CHECK_DTYPE_NOT_SAME(tokens, permuteTokensOut, return false);
        OP_CHECK_DTYPE_NOT_MATCH(expandedScaleOut, DataType::DT_FLOAT, return false);
        return true;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_QUANT_DTYPE_SUPPORT_LIST_X, return false);
    if (quantMode == QUANT_MODE_MXFP8_E5M2) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_FLOAT8_E5M2, return false);
    } else if (quantMode == QUANT_MODE_MXFP8_E4M3FN) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_FLOAT8_E4M3FN, return false);
    } else if (quantMode == QUANT_MODE_MXFP4_E2M1) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_FLOAT4_E2M1, return false);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode only supports -1, 2, 3 or 9 on Ascend 950.");
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(expandedScaleOut, DataType::DT_FLOAT8_E8M0, return false);
    return true;
}

static aclnnStatus CheckParamsV2Regbase(const aclTensor *tokens, const aclTensor *indices, int64_t quantMode,
                                        const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                        const aclTensor *expandedScaleOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut, expandedScaleOut),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(
        CheckDtypeValidV2Regbase(tokens, indices, quantMode, permuteTokensOut, sortedIndicesOut, expandedScaleOut),
        ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsV2Fallback(const aclTensor *tokens, const aclTensor *indices,
                                         const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                         const aclTensor *expandedScaleOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut, expandedScaleOut),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidRegbase(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(expandedScaleOut, DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenPermuteV2Check

static aclnnStatus BuildMoeInitRoutingV2Executor(const aclTensor *tokens, const aclTensor *indices,
                                                 int64_t numOutTokens, const aclTensor *permuteTokensOut,
                                                 const aclTensor *sortedIndicesOut, uint64_t *workspaceSize,
                                                 aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::MoeInitRoutingV2(tokensContiguous, indicesContiguous, numOutTokens, 0, 0, 0, 0, false,
                                         permuteTokensOut, sortedIndicesOut, nullptr, nullptr, uniqueExecutor.get());
    auto expandedXOut = std::get<0>(result);
    auto expandedRowIdxOut = std::get<1>(result);
    CHECK_RET(expandedXOut != nullptr && expandedRowIdxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedXOut, permuteTokensOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedRowIdxOut, sortedIndicesOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus SliceAndViewCopy(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    FVector<int64_t> offsets(dst->GetViewShape().GetDimNum(), 0);
    auto size = ToShapeVector(dst->GetViewShape());
    auto slice = l0op::Slice(src, executor->AllocIntArray(offsets.data(), offsets.size()),
                             executor->AllocIntArray(size.data(), size.size()), executor);
    CHECK_RET(slice != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(slice, dst, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PrepareIndicesForV3(const aclTensor *tokens, const aclTensor *indices,
                                       const aclTensor **outTokensContiguous, const aclTensor **outIndicesContiguous,
                                       int64_t *outFullTokenNum, aclOpExecutor *executor)
{
    auto tokensContiguous = l0op::Contiguous(tokens, executor);
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, executor);
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (indicesContiguous->GetDataType() == DataType::DT_INT64) {
        indicesContiguous = l0op::Cast(indicesContiguous, DataType::DT_INT32, executor);
        CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (indicesContiguous->GetViewShape().GetDimNum() == 1) {
        op::Shape indicesShape;
        indicesShape.AppendDim(indicesContiguous->GetViewShape().GetDim(0));
        indicesShape.AppendDim(1);
        indicesContiguous = executor->CreateView(indicesContiguous, indicesShape, 0);
        CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *outTokensContiguous = tokensContiguous;
    *outIndicesContiguous = indicesContiguous;
    *outFullTokenNum = indicesContiguous->GetViewShape().GetShapeSize();
    return ACLNN_SUCCESS;
}

static aclnnStatus AllocIntermediateTensorsForV3(int64_t fullTokenNum, int64_t quantMode,
                                                 const aclTensor *permuteTokensOut, const aclTensor *expandedScaleOut,
                                                 aclTensor **outFullExpandedX, aclTensor **outFullExpandedRowIdx,
                                                 aclTensor **outExpertTokensCount, aclTensor **outFullExpandedScale,
                                                 aclIntArray **outActiveExpertRange, aclOpExecutor *executor)
{
    op::Shape fullExpandedXShape = permuteTokensOut->GetViewShape();
    fullExpandedXShape.SetDim(0, fullTokenNum);
    auto fullExpandedX = executor->AllocTensor(fullExpandedXShape, permuteTokensOut->GetDataType(), Format::FORMAT_ND);
    op::Shape fullRowIdxShape;
    fullRowIdxShape.AppendDim(fullTokenNum);
    auto fullExpandedRowIdx = executor->AllocTensor(fullRowIdxShape, DataType::DT_INT32, Format::FORMAT_ND);
    op::Shape expertTokensShape;
    expertTokensShape.AppendDim(MoeTokenPermuteV2Check::EXPERT_NUM);
    auto expertTokensCount = executor->AllocTensor(expertTokensShape, DataType::DT_INT64, Format::FORMAT_ND);
    op::Shape fullExpandedScaleShape;
    // quantMode == -1 (QUANT_MODE_NONE): 非量化，scale shape 为一维 [fullTokenNum]
    // quantMode == 2/3/9 (MXFP8_E5M2/E4M3FN/MXFP4_E2M1): 量化模式，scale shape 从用户传入的
    //   expandedScaleOut 多维 shape 拷贝，并修正首维为 fullTokenNum
    if (quantMode == MoeTokenPermuteV2Check::QUANT_MODE_NONE) {
        fullExpandedScaleShape.AppendDim(fullTokenNum);
    } else {
        fullExpandedScaleShape = expandedScaleOut->GetViewShape();
        fullExpandedScaleShape.SetDim(0, fullTokenNum);
    }
    auto fullExpandedScale = executor->AllocTensor(
        fullExpandedScaleShape,
        quantMode == MoeTokenPermuteV2Check::QUANT_MODE_NONE ? DataType::DT_FLOAT : DataType::DT_FLOAT8_E8M0,
        Format::FORMAT_ND);
    CHECK_RET(fullExpandedX != nullptr && fullExpandedRowIdx != nullptr && expertTokensCount != nullptr &&
                  fullExpandedScale != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const int64_t activeExpertRangeData[] = {0, MoeTokenPermuteV2Check::EXPERT_NUM};
    auto activeExpertRange = executor->AllocIntArray(activeExpertRangeData, 2);
    CHECK_RET(activeExpertRange != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *outFullExpandedX = fullExpandedX;
    *outFullExpandedRowIdx = fullExpandedRowIdx;
    *outExpertTokensCount = expertTokensCount;
    *outFullExpandedScale = fullExpandedScale;
    *outActiveExpertRange = activeExpertRange;
    return ACLNN_SUCCESS;
}

static aclnnStatus CallMoeInitRoutingV3AndUnpack(
    const aclTensor *tokensContiguous, const aclTensor *indicesContiguous, int64_t fullTokenNum, int64_t quantMode,
    const aclTensor *fullExpandedX, const aclTensor *fullExpandedRowIdx, const aclTensor *expertTokensCount,
    const aclTensor *fullExpandedScale, const aclIntArray *activeExpertRange, const aclTensor **outExpandedXOut,
    const aclTensor **outExpandedRowIdxOut, const aclTensor **outExpandedScale, aclOpExecutor *executor)
{
    auto result = l0op::MoeInitRoutingV3(tokensContiguous, indicesContiguous, nullptr, nullptr, fullTokenNum, 0,
                                         MoeTokenPermuteV2Check::EXPERT_NUM, 0,
                                         MoeTokenPermuteV2Check::EXPERT_TOKENS_NUM_TYPE_COUNT, true, quantMode,
                                         activeExpertRange, MoeTokenPermuteV2Check::ROW_IDX_TYPE_GATHER, fullExpandedX,
                                         fullExpandedRowIdx, expertTokensCount, fullExpandedScale, executor);
    auto expandedXOut = std::get<0>(result);
    auto expandedRowIdxOut = std::get<1>(result);
    auto expandedScale = std::get<3>(result);
    bool hasNullptr = (expandedXOut == nullptr) || (expandedRowIdxOut == nullptr) || (expandedScale == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);
    *outExpandedXOut = expandedXOut;
    *outExpandedRowIdxOut = expandedRowIdxOut;
    *outExpandedScale = expandedScale;
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenPermuteV2GetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices,
                                                   int64_t numOutTokens, bool paddedMode, int64_t quantMode,
                                                   const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                                   const aclTensor *expandedScaleOut, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermuteV2, DFX_IN(tokens, indices, numOutTokens, paddedMode, quantMode),
                   DFX_OUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    CHECK_RET(
        MoeTokenPermuteV2Check::CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut, expandedScaleOut),
        ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(MoeTokenPermuteV2Check::CheckShapeValid(tokens, indices), ACLNN_ERR_PARAM_INVALID);

    if (!Ops::Transformer::AclnnUtil::IsRegbase()) {
        return aclnnInnerMoeTokenPermuteGetWorkspaceSize(tokens, indices, numOutTokens, paddedMode, permuteTokensOut,
                                                         sortedIndicesOut, workspaceSize, executor);
    }
    if (paddedMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paddedMode only supports false currently, but got true.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950) {
        CHECK_RET(MoeTokenPermuteV2Check::CheckParamsV2Fallback(tokens, indices, permuteTokensOut, sortedIndicesOut,
                                                                expandedScaleOut) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        return BuildMoeInitRoutingV2Executor(tokens, indices, numOutTokens, permuteTokensOut, sortedIndicesOut,
                                             workspaceSize, executor);
    }

    CHECK_RET(MoeTokenPermuteV2Check::CheckParamsV2Regbase(tokens, indices, quantMode, permuteTokensOut,
                                                           sortedIndicesOut, expandedScaleOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor *tokensContiguous = nullptr;
    const aclTensor *indicesContiguous = nullptr;
    int64_t fullTokenNum = 0;
    CHECK_RET(PrepareIndicesForV3(tokens, indices, &tokensContiguous, &indicesContiguous, &fullTokenNum,
                                  uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    aclTensor *fullExpandedX = nullptr;
    aclTensor *fullExpandedRowIdx = nullptr;
    aclTensor *expertTokensCount = nullptr;
    aclTensor *fullExpandedScale = nullptr;
    aclIntArray *activeExpertRange = nullptr;
    CHECK_RET(AllocIntermediateTensorsForV3(fullTokenNum, quantMode, permuteTokensOut, expandedScaleOut, &fullExpandedX,
                                            &fullExpandedRowIdx, &expertTokensCount, &fullExpandedScale,
                                            &activeExpertRange, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *expandedXOut = nullptr;
    const aclTensor *expandedRowIdxOut = nullptr;
    const aclTensor *expandedScale = nullptr;
    CHECK_RET(CallMoeInitRoutingV3AndUnpack(tokensContiguous, indicesContiguous, fullTokenNum, quantMode, fullExpandedX,
                                            fullExpandedRowIdx, expertTokensCount, fullExpandedScale, activeExpertRange,
                                            &expandedXOut, &expandedRowIdxOut, &expandedScale,
                                            uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(SliceAndViewCopy(expandedXOut, permuteTokensOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(SliceAndViewCopy(expandedRowIdxOut, sortedIndicesOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    if (quantMode != MoeTokenPermuteV2Check::QUANT_MODE_NONE) {
        CHECK_RET(SliceAndViewCopy(expandedScale, expandedScaleOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermuteV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    static bool isRegbase = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!isRegbase) {
        return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenPermuteV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
