/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include <algorithm>
#include <string>
#include "aclnn_moe_token_permute.h"
#include "op_host/util/op_const_def.h"
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
                                                                             DataType::DT_FLOAT, DataType::DT_INT8};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE = {DataType::DT_INT32,
                                                                                              DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};
static constexpr int64_t QUANT_MODE_NONE = -1;
static constexpr int64_t EXPERT_NUM = 10240;
static constexpr int64_t EXPERT_TOKENS_NUM_TYPE_COUNT = 1;
static constexpr int64_t ROW_IDX_TYPE_GATHER = 0;

static inline bool CheckNotNull(const aclTensor* tokens, const aclTensor* indices, const aclTensor* permuteTokensOut,
                                const aclTensor* sortedIndicesOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(sortedIndicesOut, return false);
    return true;
}

static inline bool CheckDtypeValidRegbase(const aclTensor* tokens, const aclTensor* indices,
                                          const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut)
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
static inline bool CheckDtypeValidInt8Regbase(const aclTensor* tokens, const aclTensor* indices,
                                              const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut)
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
static aclnnStatus CheckParamsRegbase(const aclTensor* tokens, const aclTensor* indices,
                                      const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidRegbase(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;

// 计算indices的元素个数，即num_tokens * topK，与MoeInitRoutingV2 tiling中的totalLength推导保持一致
static int64_t CalcIndicesTotalLength(const aclTensor* indices)
{
    const op::Shape& indicesShape = indices->GetViewShape();
    int64_t topK = (indicesShape.GetDimNum() == DIM_TWO) ? indicesShape.GetDim(1) : 1;
    return indicesShape.GetDim(0) * topK;
}

// 计算permuteTokensOut的首维大小，推导逻辑与MoeInitRoutingV2 tiling(regbase分支)中activateNum保持一致
static int64_t CalcPermutedTokenNum(const aclTensor* indices, int64_t numOutTokens)
{
    int64_t totalLength = CalcIndicesTotalLength(indices);
    int64_t permutedTokenNum = (numOutTokens <= 0) ? numOutTokens + totalLength : numOutTokens;
    permutedTokenNum = std::min(permutedTokenNum, totalLength);
    return std::max(permutedTokenNum, static_cast<int64_t>(0));
}

// 计算除首维外其余维度的乘积
static int64_t CalcTrailingDimsProduct(const aclTensor* tensor)
{
    const op::Shape& shape = tensor->GetViewShape();
    int64_t product = 1;
    for (size_t i = 1; i < shape.GetDimNum(); i++) {
        product *= shape.GetDim(i);
    }
    return product;
}

// 输入shape校验：tokens维度大于等于2，indices为1D或2D，且tokens与indices首维一致
static bool CheckInputShapeValid(const aclTensor* tokens, const aclTensor* indices)
{
    size_t tokensDimNum = tokens->GetViewShape().GetDimNum();
    OP_CHECK(tokensDimNum >= DIM_TWO,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 "MoeTokenPermute", "tokens", std::to_string(tokensDimNum),
                 "The dim number of tokens should be greater than or equal to 2, but got [" +
                     std::to_string(tokensDimNum) + "]."),
             return false);

    size_t indicesDimNum = indices->GetViewShape().GetDimNum();
    OP_CHECK(indicesDimNum == DIM_ONE || indicesDimNum == DIM_TWO,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 "MoeTokenPermute", "indices", std::to_string(indicesDimNum),
                 "The dim number of indices should be 1 or 2, but got [" + std::to_string(indicesDimNum) + "]."),
             return false);

    OP_CHECK(
        tokens->GetViewShape().GetDim(0) == indices->GetViewShape().GetDim(0),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "MoeTokenPermute", "tokens, indices",
            "The dim 0 of tokens [" + std::to_string(tokens->GetViewShape().GetDim(0)) +
                "] should be same with indices' dim 0 [" + std::to_string(indices->GetViewShape().GetDim(0)) + "]."),
        return false);
    return true;
}

// 输出shape校验：permuteTokensOut维度大于等于2，除首维外维度乘积与tokens一致，
// 首维与numOutTokens推导结果一致；sortedIndicesOut为1D且shape为(num_tokens * topK)
static bool CheckOutputShapeValid(const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens,
                                  const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut)
{
    size_t permuteTokensOutDimNum = permuteTokensOut->GetViewShape().GetDimNum();
    OP_CHECK(permuteTokensOutDimNum >= DIM_TWO,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 "MoeTokenPermute", "permute_tokens", std::to_string(permuteTokensOutDimNum),
                 "The dim number of permuteTokensOut should be greater than or equal to 2, but got [" +
                     std::to_string(permuteTokensOutDimNum) + "]."),
             return false);

    int64_t permuteTokensOutCols = CalcTrailingDimsProduct(permuteTokensOut);
    OP_CHECK(permuteTokensOutCols == CalcTrailingDimsProduct(tokens),
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                 "MoeTokenPermute", "permute_tokens",
                 "The product of permuteTokensOut's trailing dims [" + std::to_string(permuteTokensOutCols) +
                     "] should be equal to the product of tokens' trailing dims [" +
                     std::to_string(CalcTrailingDimsProduct(tokens)) + "]."),
             return false);

    int64_t permutedTokenNum = CalcPermutedTokenNum(indices, numOutTokens);
    OP_CHECK(permuteTokensOut->GetViewShape().GetDim(0) == permutedTokenNum,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                 "MoeTokenPermute", "permute_tokens",
                 "The dim 0 of permuteTokensOut should be [" + std::to_string(permutedTokenNum) + "], but got [" +
                     std::to_string(permuteTokensOut->GetViewShape().GetDim(0)) + "]."),
             return false);

    size_t sortedIndicesOutDimNum = sortedIndicesOut->GetViewShape().GetDimNum();
    OP_CHECK(sortedIndicesOutDimNum == DIM_ONE,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("MoeTokenPermute", "sorted_indices",
                                                      std::to_string(sortedIndicesOutDimNum),
                                                      "The dim number of sortedIndicesOut should be 1, but got [" +
                                                          std::to_string(sortedIndicesOutDimNum) + "]."),
             return false);

    int64_t indicesTotalLength = CalcIndicesTotalLength(indices);
    OP_CHECK(sortedIndicesOut->GetViewShape().GetDim(0) == indicesTotalLength,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                 "MoeTokenPermute", "sorted_indices",
                 "The dim 0 of sortedIndicesOut should be [" + std::to_string(indicesTotalLength) + "], but got [" +
                     std::to_string(sortedIndicesOut->GetViewShape().GetDim(0)) + "]."),
             return false);
    return true;
}

// 非INT8路径的shape前置校验，不满足时接口直接返回参数错误
static bool CheckShapeValid(const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens,
                            const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut)
{
    return CheckInputShapeValid(tokens, indices) &&
           CheckOutputShapeValid(tokens, indices, numOutTokens, permuteTokensOut, sortedIndicesOut);
}
} // namespace MoeTokenPermuteCheck

static aclnnStatus BuildMoeInitRoutingV3Executor(const aclTensor* tokens, const aclTensor* indices,
                                                 const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
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

// MoeInitRoutingV2仅支持2D输入输出：tokens维度大于2时，先将首维以外的维度乘积展平为cols，
// 以2D [num_tokens, cols]调用MoeInitRoutingV2，计算完成后将2D结果恢复为用户报的输出shape再copyout，
// 与文档声明的tokens/permuteTokensOut维度大于等于2、以及非regbase平台的语义保持一致
static aclnnStatus BuildMoeInitRoutingV2Executor(const aclTensor* tokens, const aclTensor* indices,
                                                 int64_t numOutTokens, const aclTensor* permuteTokensOut,
                                                 const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // tokens维度大于2时展平为2D，首维num_tokens不变，其余维度乘积作为cols
    const aclTensor* tokens2D = tokensContiguous;
    if (tokensContiguous->GetViewShape().GetDimNum() > MoeTokenPermuteCheck::DIM_TWO) {
        op::Shape tokens2DShape;
        tokens2DShape.AppendDim(tokensContiguous->GetViewShape().GetDim(0));
        tokens2DShape.AppendDim(MoeTokenPermuteCheck::CalcTrailingDimsProduct(tokensContiguous));
        tokens2D = uniqueExecutor->CreateView(tokensContiguous, tokens2DShape, 0);
        CHECK_RET(tokens2D != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 构造2D输出描述符，MoeInitRoutingV2内部按该描述符的shape/dtype分配实际输出
    op::Shape permuteTokens2DShape;
    permuteTokens2DShape.AppendDim(MoeTokenPermuteCheck::CalcPermutedTokenNum(indicesContiguous, numOutTokens));
    permuteTokens2DShape.AppendDim(MoeTokenPermuteCheck::CalcTrailingDimsProduct(tokensContiguous));
    auto permuteTokens2D =
        uniqueExecutor->AllocTensor(permuteTokens2DShape, permuteTokensOut->GetDataType(), Format::FORMAT_ND);
    CHECK_RET(permuteTokens2D != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口进行计算
    auto result = l0op::MoeInitRoutingV2(tokens2D, indicesContiguous, numOutTokens, 0, 0, 0, 0, false, permuteTokens2D,
                                         sortedIndicesOut, nullptr, nullptr, uniqueExecutor.get());

    auto expandedXOut = std::get<0>(result);
    auto expandedRowIdxOut = std::get<1>(result);
    CHECK_RET(expandedXOut != nullptr && expandedRowIdxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将2D计算结果恢复为用户输出的shape，copyout结果，如果出参是非连续Tensor，
    // 需要把计算完的连续Tensor转非连续
    auto expandedXOutRestored = uniqueExecutor->CreateView(expandedXOut, permuteTokensOut->GetViewShape(), 0);
    CHECK_RET(expandedXOutRestored != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedXOutRestored, permuteTokensOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedRowIdxOut, sortedIndicesOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor* tokens, const aclTensor* indices,
                                                 int64_t numOutTokens, bool paddedMode,
                                                 const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermute, DFX_IN(tokens, indices, numOutTokens, paddedMode),
                   DFX_OUT(permuteTokensOut, sortedIndicesOut));

    CHECK_RET(MoeTokenPermuteCheck::CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut),
              ACLNN_ERR_PARAM_NULLPTR);

    bool useMoeInitRouting = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRouting) {
        return aclnnInnerMoeTokenPermuteGetWorkspaceSize(tokens, indices, numOutTokens, paddedMode, permuteTokensOut,
                                                         sortedIndicesOut, workspaceSize, executor);
    }
    if (paddedMode) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("MoeTokenPermute", "padded_mode",
                                                 "paddedMode only supports false currently, but got true.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool useMoeInitRoutingV3 = tokens->GetDataType() == DataType::DT_INT8;
    if (useMoeInitRoutingV3) {
        CHECK_RET(MoeTokenPermuteCheck::CheckDtypeValidInt8Regbase(tokens, indices, permuteTokensOut, sortedIndicesOut),
                  ACLNN_ERR_PARAM_INVALID);
        return BuildMoeInitRoutingV3Executor(tokens, indices, permuteTokensOut, sortedIndicesOut, workspaceSize,
                                             executor);
    }
    aclnnStatus ret = MoeTokenPermuteCheck::CheckParamsRegbase(tokens, indices, permuteTokensOut, sortedIndicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(MoeTokenPermuteCheck::CheckShapeValid(tokens, indices, numOutTokens, permuteTokensOut, sortedIndicesOut),
              ACLNN_ERR_PARAM_INVALID);

    return BuildMoeInitRoutingV2Executor(tokens, indices, numOutTokens, permuteTokensOut, sortedIndicesOut,
                                         workspaceSize, executor);
}

aclnnStatus aclnnMoeTokenPermute(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeInitRouting = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRouting) {
        return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
    }
    L2_DFX_PHASE_2(aclnnMoeTokenPermute);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
