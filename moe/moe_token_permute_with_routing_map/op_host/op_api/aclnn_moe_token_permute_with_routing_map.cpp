/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_permute_with_routing_map.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_moe_token_permute_with_routing_map.h"
#ifdef BUILD_OPEN_PROJECT_API
#include "../../../moe/3rd/moe_gather_v2/op_host/op_api/moe_gather_v2.h"
#else
#include "level0/gather_v2.h"
#endif
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static constexpr int64_t TOKENS_SHAPE_SIZE = 2;
static constexpr int64_t TRANSPOSE_SHAPE_SIZE = 2;
static constexpr int64_t INPUT_MAX_GROUP = 2048;
static constexpr int64_t SORT_LIMIT_LENGTH = 16777215;

static const std::initializer_list<DataType> dtype_list = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                           op::DataType::DT_BF16};

static const std::initializer_list<DataType> routing_map_dtype_list = {op::DataType::DT_INT8, op::DataType::DT_BOOL};

static const std::initializer_list<DataType> indice_dtype_list = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *tokens, const aclTensor *routingMap, const aclTensor *probsOptional,
                                const aclTensor *permuteTokensOut, const aclTensor *permuteProbsOutOptional,
                                const aclTensor *sortedIndicesOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(routingMap, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(sortedIndicesOut, return false);
    if (probsOptional != nullptr) {
        // 检查groupIdxOptional的数据类型是否在支持列表内
        OP_CHECK_NULL(permuteProbsOutOptional, return false);
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *tokens, const aclTensor *routingMap, const aclTensor *probsOptional,
                                   const aclTensor *permuteTokensOut, const aclTensor *permuteProbsOutOptional,
                                   const aclTensor *sortedIndicesOut)
{
    // 检查gradY的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(tokens, dtype_list, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, dtype_list, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(routingMap, routing_map_dtype_list, return false);
    OP_CHECK_DTYPE_NOT_MATCH(tokens, permuteTokensOut->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, indice_dtype_list, return false);

    // 检查输入和输出的数据类型是否一致

    if (probsOptional != nullptr) {
        // 检查groupIdxOptional的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(probsOptional, dtype_list, return false);
        if (probsOptional->GetDataType() != op::DataType::DT_FLOAT || tokens->GetDataType() != op::DataType::DT_BF16) {
            OP_CHECK_DTYPE_NOT_MATCH(tokens, probsOptional->GetDataType(), return false);
        }
    }
    if (permuteProbsOutOptional != nullptr) {
        // 检查groupIdxOptional的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteProbsOutOptional, dtype_list, return false);
        OP_CHECK_DTYPE_NOT_MATCH(probsOptional, permuteProbsOutOptional->GetDataType(), return false);
    }
    return true;
}

static bool CheckShapeValid(const aclTensor *routingMap, const aclTensor *probsOptional)
{
    auto routingMapDimNum = routingMap->GetViewShape().GetDimNum();
    OP_CHECK(routingMapDimNum == TRANSPOSE_SHAPE_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of routingMap should be two, but got %ld.",
                     static_cast<int64_t>(routingMapDimNum)),
             return false);
    if (probsOptional != nullptr) {
        auto probsDimNum = probsOptional->GetViewShape().GetDimNum();
        OP_CHECK(probsDimNum == TRANSPOSE_SHAPE_SIZE,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of probs should be two, but got %ld.",
                         static_cast<int64_t>(probsDimNum)),
                 return false);
        OP_CHECK(probsOptional->GetViewShape().GetDim(1) == routingMap->GetViewShape().GetDim(1),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim 1 of probs %ld should be same with routingMap's dim 0 %ld.",
                         static_cast<int64_t>(probsOptional->GetViewShape().GetDim(1)),
                         static_cast<int64_t>(routingMap->GetViewShape().GetDim(1))),
                 return false);
        OP_CHECK(probsOptional->GetViewShape().GetDim(0) == routingMap->GetViewShape().GetDim(0),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim 0 of probs %ld should be same with routingMap's dim 0 %ld.",
                         static_cast<int64_t>(probsOptional->GetViewShape().GetDim(0)),
                         static_cast<int64_t>(routingMap->GetViewShape().GetDim(0))),
                 return false);
    }

    return true;
}

static bool CheckTokensValid(const aclTensor *tokens)
{
    auto tokensDimNum = tokens->GetViewShape().GetDimNum();
    OP_CHECK(tokensDimNum == TOKENS_SHAPE_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of tokens should be two, but got %ld.",
                     static_cast<int64_t>(tokensDimNum)),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *tokens, const aclTensor *routingMap, const aclTensor *probsOptional,
                               const aclTensor *permuteTokensOut, const aclTensor *permuteProbsOutOptional,
                               const aclTensor *sortedIndicesOut, int64_t &numOutTokens, bool dropAndPad)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(
        CheckNotNull(tokens, routingMap, probsOptional, permuteTokensOut, permuteProbsOutOptional, sortedIndicesOut),
        ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(
        CheckDtypeValid(tokens, routingMap, probsOptional, permuteTokensOut, permuteProbsOutOptional, sortedIndicesOut),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(routingMap, probsOptional), ACLNN_ERR_PARAM_INVALID);
    int64_t tokenNum = routingMap->GetViewShape().GetDim(0);
    int64_t expertNum = routingMap->GetViewShape().GetDim(1);

    int64_t alignNum = (dropAndPad == true) ? expertNum : tokenNum;
    alignNum = (alignNum == 0) ? 1 : alignNum;
    CHECK_RET(CheckTokensValid(tokens), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(numOutTokens >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numOutTokens should be greater than %ld, but got %ld.", int64_t(0),
                     numOutTokens),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(numOutTokens <= tokenNum * expertNum,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numOutTokens should not be greater than %ld, but got %ld.",
                     tokenNum * expertNum, numOutTokens),
             return ACLNN_ERR_PARAM_INVALID);
    int64_t numOutTokensAlign = numOutTokens / alignNum * alignNum;
    int64_t permuteOutD = permuteTokensOut->GetViewShape().GetDim(0);
    OP_CHECK(
        tokenNum < SORT_LIMIT_LENGTH,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tokenNum should be less than %ld, but got %ld.", SORT_LIMIT_LENGTH, tokenNum),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(expertNum < SORT_LIMIT_LENGTH,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expertNum should be less than %ld, but got %ld.", SORT_LIMIT_LENGTH,
                     expertNum),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(numOutTokensAlign == permuteOutD,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out token's dim 0' should be %ld, but got %ld.", numOutTokensAlign,
                     permuteOutD),
             return ACLNN_ERR_PARAM_INVALID);
    if (permuteProbsOutOptional != nullptr) {
        int64_t permuteProbOutD = permuteProbsOutOptional->GetViewShape().GetDim(0);
        OP_CHECK(numOutTokensAlign == permuteProbOutD,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out permuteTokensOut's dim 0' should be %ld, but got %ld.", alignNum,
                         permuteProbOutD),
                 return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ProbsOptionalHandler(const aclTensor *probsOptional, const aclTensor *permuteProbsOpOut,
                                        aclTensor *permuteProbsOutOptional, aclOpExecutor *executor)
{
    if (probsOptional != nullptr) {
        CHECK_RET(permuteProbsOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto permuteProbsResult = l0op::ViewCopy(permuteProbsOpOut, permuteProbsOutOptional, executor);
        CHECK_RET(permuteProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(const aclTensor *tokens, const aclTensor *routingMap,
                                                               const aclTensor *probsOptional, int64_t numOutTokens,
                                                               bool dropAndPad, aclTensor *permuteTokensOut,
                                                               aclTensor *permuteProbsOutOptional,
                                                               aclTensor *sortedIndicesOut, uint64_t *workspaceSize,
                                                               aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMoeTokenPermuteWithRoutingMap,
                   DFX_IN(tokens, routingMap, probsOptional, numOutTokens, dropAndPad),
                   DFX_OUT(permuteTokensOut, permuteProbsOutOptional, sortedIndicesOut));

    auto ret = CheckParams(tokens, routingMap, probsOptional, permuteTokensOut, permuteProbsOutOptional,
                           sortedIndicesOut, numOutTokens, dropAndPad);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    if (routingMap->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // gradY如果非连续，需要转连续
    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto routingMapContiguous = l0op::Contiguous(routingMap, uniqueExecutor.get());
    CHECK_RET(routingMapContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    FVector<int64_t> transposeDim = {1, 0};

    auto perm = uniqueExecutor->AllocIntArray(transposeDim.data(), transposeDim.size());

    auto routingMapTranspose = l0op::Transpose(routingMapContiguous, perm, uniqueExecutor.get());
    CHECK_RET(routingMapTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (probsOptional != nullptr) {
        probsOptional = l0op::Contiguous(probsOptional, uniqueExecutor.get());
        CHECK_RET(probsOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);

        probsOptional = l0op::Transpose(probsOptional, perm, uniqueExecutor.get());
        CHECK_RET(probsOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 调用l0算子GroupedBiasAddGrad行计算
    auto MoeTokenPermuteWithRoutingMapOut = l0op::MoeTokenPermuteWithRoutingMap(
        tokensContiguous, routingMapTranspose, probsOptional, numOutTokens, dropAndPad, uniqueExecutor.get());

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续

    auto sortedIndicesOpOut = MoeTokenPermuteWithRoutingMapOut[2];
    CHECK_RET(sortedIndicesOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto sortedIndicesResult = l0op::ViewCopy(sortedIndicesOpOut, sortedIndicesOut, uniqueExecutor.get());
    CHECK_RET(sortedIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (probsOptional != nullptr) {
        auto permuteProbsOpOut = MoeTokenPermuteWithRoutingMapOut[1];
        CHECK_RET(permuteProbsOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto permuteProbsResult = l0op::ViewCopy(permuteProbsOpOut, permuteProbsOutOptional, uniqueExecutor.get());
        CHECK_RET(permuteProbsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *permuteTokensOpOut;
    bool isRegbase = Ops::Transformer::AclnnUtil::IsRegbase();
    if (isRegbase) { // regbase 场景在算子内部进行gather，其余场景由gather算子计算
        permuteTokensOpOut = MoeTokenPermuteWithRoutingMapOut[0];
    } else {
#ifdef BUILD_OPEN_PROJECT_API
        if (dropAndPad) {
            permuteTokensOpOut = l0op::MoeGatherV2(tokensContiguous, 0, sortedIndicesOut, uniqueExecutor.get());
        } else {
            permuteTokensOpOut = MoeTokenPermuteWithRoutingMapOut[0];
        }
#else
        if (dropAndPad) {
            permuteTokensOpOut = l0op::GatherV2(tokensContiguous, 0, sortedIndicesOut, uniqueExecutor.get());
        } else {
            permuteTokensOpOut = MoeTokenPermuteWithRoutingMapOut[0];
        }
#endif
    }

    CHECK_RET(permuteTokensOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto permuteTokensResult = l0op::ViewCopy(permuteTokensOpOut, permuteTokensOut, uniqueExecutor.get());
    CHECK_RET(permuteTokensResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnMoeTokenPermuteWithRoutingMap);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
