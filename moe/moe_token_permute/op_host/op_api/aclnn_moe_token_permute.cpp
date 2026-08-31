/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_permute.h"
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

namespace MoeTokenPermuteCheck {

static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                             DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE = {DataType::DT_INT32,
                                                                                              DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};
static constexpr int64_t QUANT_MODE_NONE = -1;
static constexpr int64_t EXPERT_NUM = 10240;
static constexpr int64_t EXPERT_TOKENS_NUM_TYPE_COUNT = 1;
static constexpr int64_t ROW_IDX_TYPE_GATHER = 0;

static inline bool CheckNotNull(const aclTensor *tokens, const aclTensor *indices, const aclTensor *permuteTokensOut,
                                const aclTensor *sortedIndicesOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(sortedIndicesOut, return false);
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

static inline bool CheckDtypeValidInt8Regbase(const aclTensor *tokens, const aclTensor *indices,
                                              const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut)
{
    if (tokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_MATCH(tokens, DataType::DT_INT8, return false);
    }
    if (indices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (permuteTokensOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_INT8, return false);
        OP_CHECK_DTYPE_NOT_SAME(tokens, permuteTokensOut, return false);
    }
    if (sortedIndicesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static aclnnStatus CheckParamsRegbase(const aclTensor *tokens, const aclTensor *indices,
                                      const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidRegbase(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenPermuteCheck

static aclnnStatus BuildMoeInitRoutingV3Executor(const aclTensor *tokens, const aclTensor *indices,
                                                 const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (indicesContiguous->GetDataType() == DataType::DT_INT64) {
        indicesContiguous = l0op::Cast(indicesContiguous, DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (indicesContiguous->GetViewShape().GetDimNum() == 1) {
        op::Shape indicesShape;
        indicesShape.AppendDim(indicesContiguous->GetViewShape().GetDim(0));
        indicesShape.AppendDim(1);
        indicesContiguous = uniqueExecutor->CreateView(indicesContiguous, indicesShape, 0);
        CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    int64_t fullTokenNum = indicesContiguous->GetViewShape().GetShapeSize();
    op::Shape fullExpandedXShape = permuteTokensOut->GetViewShape();
    fullExpandedXShape.SetDim(0, fullTokenNum);
    auto fullExpandedX =
        uniqueExecutor->AllocTensor(fullExpandedXShape, permuteTokensOut->GetDataType(), Format::FORMAT_ND);

    op::Shape fullRowIdxShape;
    fullRowIdxShape.AppendDim(fullTokenNum);
    auto fullExpandedRowIdx = uniqueExecutor->AllocTensor(fullRowIdxShape, DataType::DT_INT32, Format::FORMAT_ND);

    op::Shape expertTokensShape;
    expertTokensShape.AppendDim(MoeTokenPermuteCheck::EXPERT_NUM);
    auto expertTokensCount = uniqueExecutor->AllocTensor(expertTokensShape, DataType::DT_INT64, Format::FORMAT_ND);

    // MoeInitRoutingV3 requires an expandedScale output descriptor even though MoeTokenPermute does not expose it.
    op::Shape fullExpandedScaleShape;
    fullExpandedScaleShape.AppendDim(fullTokenNum);
    auto fullExpandedScale = uniqueExecutor->AllocTensor(fullExpandedScaleShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    CHECK_RET(fullExpandedX != nullptr && fullExpandedRowIdx != nullptr && expertTokensCount != nullptr &&
                  fullExpandedScale != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const int64_t activeExpertRangeData[] = {0, MoeTokenPermuteCheck::EXPERT_NUM};
    auto activeExpertRange = uniqueExecutor->AllocIntArray(activeExpertRangeData, 2);
    CHECK_RET(activeExpertRange != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::MoeInitRoutingV3(
        tokensContiguous, indicesContiguous, nullptr, nullptr, fullTokenNum, 0, MoeTokenPermuteCheck::EXPERT_NUM, 0,
        MoeTokenPermuteCheck::EXPERT_TOKENS_NUM_TYPE_COUNT, true, MoeTokenPermuteCheck::QUANT_MODE_NONE,
        activeExpertRange, MoeTokenPermuteCheck::ROW_IDX_TYPE_GATHER, fullExpandedX, fullExpandedRowIdx,
        expertTokensCount, fullExpandedScale, uniqueExecutor.get());
    auto expandedXOut = std::get<0>(result);
    auto expandedRowIdxOut = std::get<1>(result);
    CHECK_RET(expandedXOut != nullptr && expandedRowIdxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t> expandedXOffsets(permuteTokensOut->GetViewShape().GetDimNum(), 0);
    auto expandedXSize = ToShapeVector(permuteTokensOut->GetViewShape());
    auto expandedXSlice =
        l0op::Slice(expandedXOut, uniqueExecutor->AllocIntArray(expandedXOffsets.data(), expandedXOffsets.size()),
                    uniqueExecutor->AllocIntArray(expandedXSize.data(), expandedXSize.size()), uniqueExecutor.get());
    CHECK_RET(expandedXSlice != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedXSlice, permuteTokensOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t> expandedRowIdxOffsets(sortedIndicesOut->GetViewShape().GetDimNum(), 0);
    auto expandedRowIdxSize = ToShapeVector(sortedIndicesOut->GetViewShape());
    auto expandedRowIdxSlice = l0op::Slice(
        expandedRowIdxOut, uniqueExecutor->AllocIntArray(expandedRowIdxOffsets.data(), expandedRowIdxOffsets.size()),
        uniqueExecutor->AllocIntArray(expandedRowIdxSize.data(), expandedRowIdxSize.size()), uniqueExecutor.get());
    CHECK_RET(expandedRowIdxSlice != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedRowIdxSlice, sortedIndicesOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *tokens, const aclTensor *indices,
                                                 int64_t numOutTokens, bool paddedMode,
                                                 const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermute, DFX_IN(tokens, indices, numOutTokens, paddedMode),
                   DFX_OUT(permuteTokensOut, sortedIndicesOut));

    CHECK_RET(MoeTokenPermuteCheck::CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut),
              ACLNN_ERR_PARAM_NULLPTR);

    bool useMoeInitRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2) {
        return aclnnInnerMoeTokenPermuteGetWorkspaceSize(tokens, indices, numOutTokens, paddedMode, permuteTokensOut,
                                                         sortedIndicesOut, workspaceSize, executor);
    }
    if (paddedMode) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "paddedMode only supports false currently, but got true.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool isRegbaseCalled =
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950 && tokens->GetDataType() == DataType::DT_INT8;
    if (isRegbaseCalled) {
        CHECK_RET(MoeTokenPermuteCheck::CheckDtypeValidInt8Regbase(tokens, indices, permuteTokensOut, sortedIndicesOut),
                  ACLNN_ERR_PARAM_INVALID);
        return BuildMoeInitRoutingV3Executor(tokens, indices, permuteTokensOut, sortedIndicesOut, workspaceSize,
                                             executor);
    }

    aclnnStatus ret = MoeTokenPermuteCheck::CheckParamsRegbase(tokens, indices, permuteTokensOut, sortedIndicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口进行计算
    auto result = l0op::MoeInitRoutingV2(tokensContiguous, indicesContiguous, numOutTokens, 0, 0, 0, 0, false,
                                         permuteTokensOut, sortedIndicesOut, nullptr, nullptr, uniqueExecutor.get());

    auto expandedXOut_ = std::get<0>(result);
    auto expandedRowIdxOut_ = std::get<1>(result);
    bool hasNullptr = (expandedXOut_ == nullptr) || (expandedRowIdxOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyExpandedXOutResult = l0op::ViewCopy(expandedXOut_, permuteTokensOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyExpandedRowIdxOutResult = l0op::ViewCopy(expandedRowIdxOut_, sortedIndicesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedRowIdxOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    static bool useMoeInitRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2) {
        return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenPermute);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
