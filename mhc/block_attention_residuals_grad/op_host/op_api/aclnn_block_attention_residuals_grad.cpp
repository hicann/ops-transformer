/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_attention_residuals_grad.h"
#include "block_attention_residuals_grad.h"
#include "log/log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "level0/zero_op.h"

#include <initializer_list>
#include <string>
#include <vector>

using namespace op;

namespace {
constexpr const char *API_NAME = "aclnnBlockAttentionResidualsGradGetWorkspaceSize";
constexpr size_t DIM_NUM_1D = 1;
constexpr size_t DIM_NUM_2D = 2;
constexpr size_t DIM_NUM_3D = 3;
constexpr size_t DIM_INDEX_0 = 0;
constexpr size_t DIM_INDEX_1 = 1;
constexpr size_t DIM_INDEX_2 = 2;
constexpr int64_t MIN_TOKEN_NUM = 0;
constexpr int64_t MIN_BLOCK_NUM = 0;
constexpr int64_t MAX_BLOCK_NUM = 128;
constexpr int64_t MIN_HIDDEN_SIZE = 0;
constexpr int64_t PROJ_WEIGHT_ROW_NUM = 1;

aclnnStatus CheckRequiredParameter(const void *parameter, const char *parameterName)
{
    if (parameter == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, parameterName, "nullptr",
                                              "required parameter must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNotNull(const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                         const aclTensor *normWeight, const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                         const aclTensor *probs, const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes,
                         const aclTensor *gradProjWeight, const aclTensor *gradNormWeight, uint64_t *workspaceSize,
                         aclOpExecutor **executor)
{
    if (CheckRequiredParameter(partialBlock, "partialBlock") != ACLNN_SUCCESS ||
        CheckRequiredParameter(blockRes, "blockRes") != ACLNN_SUCCESS ||
        CheckRequiredParameter(projWeight, "projWeight") != ACLNN_SUCCESS ||
        CheckRequiredParameter(normWeight, "normWeight") != ACLNN_SUCCESS ||
        CheckRequiredParameter(gradHiddenStates, "gradHiddenStates") != ACLNN_SUCCESS ||
        CheckRequiredParameter(invNorm, "invNorm") != ACLNN_SUCCESS ||
        CheckRequiredParameter(probs, "probs") != ACLNN_SUCCESS ||
        CheckRequiredParameter(gradPartialBlock, "gradPartialBlock") != ACLNN_SUCCESS ||
        CheckRequiredParameter(gradBlockRes, "gradBlockRes") != ACLNN_SUCCESS ||
        CheckRequiredParameter(gradProjWeight, "gradProjWeight") != ACLNN_SUCCESS ||
        CheckRequiredParameter(gradNormWeight, "gradNormWeight") != ACLNN_SUCCESS ||
        CheckRequiredParameter(workspaceSize, "workspaceSize") != ACLNN_SUCCESS ||
        CheckRequiredParameter(executor, "executor") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSupportedMainDtype(const aclTensor *tensor, const char *tensorName)
{
    const DataType actualDtype = tensor->GetDataType();
    if (actualDtype != DataType::DT_FLOAT16 && actualDtype != DataType::DT_BF16 && actualDtype != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(API_NAME, tensorName, op::ToString(actualDtype).GetString(),
                                  "FLOAT16, BFLOAT16 or FLOAT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypeMatches(const aclTensor *tensor, const char *tensorName, DataType expectedDtype)
{
    const DataType actualDtype = tensor->GetDataType();
    if (actualDtype != expectedDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(API_NAME, tensorName, op::ToString(actualDtype).GetString(),
                                  op::ToString(expectedDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                       const aclTensor *normWeight, const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                       const aclTensor *probs, const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes,
                       const aclTensor *gradProjWeight, const aclTensor *gradNormWeight)
{
    if (CheckSupportedMainDtype(partialBlock, "partialBlock") != ACLNN_SUCCESS ||
        CheckSupportedMainDtype(blockRes, "blockRes") != ACLNN_SUCCESS ||
        CheckSupportedMainDtype(projWeight, "projWeight") != ACLNN_SUCCESS ||
        CheckSupportedMainDtype(normWeight, "normWeight") != ACLNN_SUCCESS ||
        CheckSupportedMainDtype(gradHiddenStates, "gradHiddenStates") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const DataType mainDtype = partialBlock->GetDataType();
    if (CheckDtypeMatches(blockRes, "blockRes", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(projWeight, "projWeight", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(normWeight, "normWeight", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(gradHiddenStates, "gradHiddenStates", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(invNorm, "invNorm", DataType::DT_FLOAT) != ACLNN_SUCCESS ||
        CheckDtypeMatches(probs, "probs", DataType::DT_FLOAT) != ACLNN_SUCCESS ||
        CheckDtypeMatches(gradPartialBlock, "gradPartialBlock", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(gradBlockRes, "gradBlockRes", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(gradProjWeight, "gradProjWeight", mainDtype) != ACLNN_SUCCESS ||
        CheckDtypeMatches(gradNormWeight, "gradNormWeight", mainDtype) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorDimension(const aclTensor *tensor, const char *tensorName, size_t expectedDimension)
{
    const size_t actualDimension = tensor->GetViewShape().GetDimNum();
    if (actualDimension != expectedDimension) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(API_NAME, tensorName, std::to_string(actualDimension).c_str(),
                                     std::to_string(expectedDimension).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDimensions(const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                            const aclTensor *normWeight, const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                            const aclTensor *probs, const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes,
                            const aclTensor *gradProjWeight, const aclTensor *gradNormWeight)
{
    if (CheckTensorDimension(partialBlock, "partialBlock", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(blockRes, "blockRes", DIM_NUM_3D) != ACLNN_SUCCESS ||
        CheckTensorDimension(projWeight, "projWeight", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(normWeight, "normWeight", DIM_NUM_1D) != ACLNN_SUCCESS ||
        CheckTensorDimension(gradHiddenStates, "gradHiddenStates", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(invNorm, "invNorm", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(probs, "probs", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(gradPartialBlock, "gradPartialBlock", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(gradBlockRes, "gradBlockRes", DIM_NUM_3D) != ACLNN_SUCCESS ||
        CheckTensorDimension(gradProjWeight, "gradProjWeight", DIM_NUM_2D) != ACLNN_SUCCESS ||
        CheckTensorDimension(gradNormWeight, "gradNormWeight", DIM_NUM_1D) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckInputShapes(const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                             const aclTensor *normWeight, const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                             const aclTensor *probs)
{
    const auto &partialBlockShape = partialBlock->GetViewShape();
    const auto &blockResShape = blockRes->GetViewShape();
    const auto &projWeightShape = projWeight->GetViewShape();
    const auto &normWeightShape = normWeight->GetViewShape();
    const auto &gradHiddenStatesShape = gradHiddenStates->GetViewShape();
    const auto &invNormShape = invNorm->GetViewShape();
    const auto &probsShape = probs->GetViewShape();
    const int64_t tokenNum = partialBlockShape.GetDim(DIM_INDEX_0);
    const int64_t hiddenSize = partialBlockShape.GetDim(DIM_INDEX_1);
    const int64_t blockNum = blockResShape.GetDim(DIM_INDEX_1);
    if (tokenNum < MIN_TOKEN_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "partialBlock.shape[0]", std::to_string(tokenNum).c_str(),
                                              "partialBlock.shape[0] must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockNum < MIN_BLOCK_NUM || blockNum > MAX_BLOCK_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "blockRes.shape[1]", std::to_string(blockNum).c_str(),
                                              "blockRes.shape[1] must be in [0, 128]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hiddenSize < MIN_HIDDEN_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "partialBlock.shape[1]", std::to_string(hiddenSize).c_str(),
                                              "partialBlock.shape[1] must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockResShape.GetDim(DIM_INDEX_0) != tokenNum || blockResShape.GetDim(DIM_INDEX_2) != hiddenSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s blockRes shape must be [%ld, %ld, %ld], got [%ld, %ld, %ld]", API_NAME,
                tokenNum, blockNum, hiddenSize, blockResShape.GetDim(DIM_INDEX_0), blockResShape.GetDim(DIM_INDEX_1),
                blockResShape.GetDim(DIM_INDEX_2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (projWeightShape.GetDim(DIM_INDEX_0) != PROJ_WEIGHT_ROW_NUM ||
        projWeightShape.GetDim(DIM_INDEX_1) != hiddenSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s projWeight shape must be [1, %ld], got [%ld, %ld]", API_NAME, hiddenSize,
                projWeightShape.GetDim(DIM_INDEX_0), projWeightShape.GetDim(DIM_INDEX_1));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (normWeightShape.GetDim(DIM_INDEX_0) != hiddenSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s normWeight shape must be [%ld], got [%ld]", API_NAME, hiddenSize,
                normWeightShape.GetDim(DIM_INDEX_0));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gradHiddenStatesShape.GetDim(DIM_INDEX_0) != tokenNum ||
        gradHiddenStatesShape.GetDim(DIM_INDEX_1) != hiddenSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s gradHiddenStates shape must be [%ld, %ld], got [%ld, %ld]", API_NAME,
                tokenNum, hiddenSize, gradHiddenStatesShape.GetDim(DIM_INDEX_0),
                gradHiddenStatesShape.GetDim(DIM_INDEX_1));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (invNormShape.GetDim(DIM_INDEX_0) != tokenNum || invNormShape.GetDim(DIM_INDEX_1) != blockNum + 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s invNorm shape must be [%ld, %ld], got [%ld, %ld]", API_NAME, tokenNum,
                blockNum + 1, invNormShape.GetDim(DIM_INDEX_0), invNormShape.GetDim(DIM_INDEX_1));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (probsShape.GetDim(DIM_INDEX_0) != tokenNum || probsShape.GetDim(DIM_INDEX_1) != blockNum + 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s probs shape must be [%ld, %ld], got [%ld, %ld]", API_NAME, tokenNum,
                blockNum + 1, probsShape.GetDim(DIM_INDEX_0), probsShape.GetDim(DIM_INDEX_1));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes,
                              const aclTensor *gradProjWeight, const aclTensor *gradNormWeight,
                              const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                              const aclTensor *normWeight)
{
    if (gradPartialBlock->GetViewShape() != partialBlock->GetViewShape()) {
        OP_LOGE_FOR_INVALID_SHAPE(API_NAME, "gradPartialBlock",
                                  op::ToString(gradPartialBlock->GetViewShape()).GetString(),
                                  op::ToString(partialBlock->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gradBlockRes->GetViewShape() != blockRes->GetViewShape()) {
        OP_LOGE_FOR_INVALID_SHAPE(API_NAME, "gradBlockRes", op::ToString(gradBlockRes->GetViewShape()).GetString(),
                                  op::ToString(blockRes->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gradProjWeight->GetViewShape() != projWeight->GetViewShape()) {
        OP_LOGE_FOR_INVALID_SHAPE(API_NAME, "gradProjWeight", op::ToString(gradProjWeight->GetViewShape()).GetString(),
                                  op::ToString(projWeight->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gradNormWeight->GetViewShape() != normWeight->GetViewShape()) {
        OP_LOGE_FOR_INVALID_SHAPE(API_NAME, "gradNormWeight", op::ToString(gradNormWeight->GetViewShape()).GetString(),
                                  op::ToString(normWeight->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckValidBlockNum(const aclTensor *blockRes, int64_t validBlockNum)
{
    const int64_t blockNum = blockRes->GetViewShape().GetDim(DIM_INDEX_1);
    if (validBlockNum != -1 && validBlockNum != blockNum) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "validBlockNum", std::to_string(validBlockNum).c_str(),
                                              "validBlockNum must be -1 or blockRes.shape[1]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

const aclTensor *CreateZeroLikeOutput(const aclTensor *grad, const aclTensor *reshapeRef,
                                      UniqueExecutor &uniqueExecutor)
{
    const auto *zero = l0op::ZerosLike(grad, uniqueExecutor.get());
    if (zero == nullptr) {
        return nullptr;
    }
    if (reshapeRef == nullptr) {
        return zero;
    }

    const auto &shape = reshapeRef->GetViewShape();
    const size_t dimNum = shape.GetDimNum();
    std::vector<int64_t> dims(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        dims[i] = shape.GetDim(i);
    }
    auto *shapeArray = uniqueExecutor->AllocIntArray(dims.data(), dimNum);
    if (shapeArray == nullptr) {
        return nullptr;
    }
    return l0op::Reshape(zero, shapeArray, uniqueExecutor.get());
}

aclnnStatus HandleEmptyTensor(const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                              const aclTensor *normWeight, const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                              const aclTensor *probs, const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes,
                              const aclTensor *gradProjWeight, const aclTensor *gradNormWeight,
                              UniqueExecutor &uniqueExecutor, uint64_t *workspaceSize, aclOpExecutor **executor,
                              bool &handled)
{
    handled = false;
    const bool hasEmptyInput = partialBlock->IsEmpty() || blockRes->IsEmpty() || projWeight->IsEmpty() ||
                               normWeight->IsEmpty() || gradHiddenStates->IsEmpty() || invNorm->IsEmpty() ||
                               probs->IsEmpty();
    if (!hasEmptyInput) {
        return ACLNN_SUCCESS;
    }
    // CheckParams has validated all shape relationships. Only blockRes can be
    // empty while partialBlock is nonempty (N=0); that case still needs computation.
    if (blockRes->IsEmpty() && !partialBlock->IsEmpty()) {
        return ACLNN_SUCCESS;
    }
    // T=0 or H=0: skip empty outputs and zero any nonempty gradients.
    for (const aclTensor *grad : {gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight}) {
        if (grad->IsEmpty()) {
            continue;
        }

        // ZerosLike may drop the leading singleton dimension of [1, H].
        // Restore gradProjWeight's view shape before copying the zero result.
        const aclTensor *zero =
            CreateZeroLikeOutput(grad, grad == gradProjWeight ? gradProjWeight : nullptr, uniqueExecutor);
        CHECK_RET(zero != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const auto *output = l0op::ViewCopy(zero, grad, uniqueExecutor.get());
        CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    handled = true;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight,
                        const aclTensor *normWeight, const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                        const aclTensor *probs, int64_t validBlockNum, const aclTensor *gradPartialBlock,
                        const aclTensor *gradBlockRes, const aclTensor *gradProjWeight, const aclTensor *gradNormWeight,
                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (CheckNotNull(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, gradPartialBlock,
                     gradBlockRes, gradProjWeight, gradNormWeight, workspaceSize, executor) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (CheckDtype(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, gradPartialBlock,
                   gradBlockRes, gradProjWeight, gradNormWeight) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckDimensions(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs,
                        gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckValidBlockNum(blockRes, validBlockNum) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckInputShapes(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs) !=
        ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckOutputShapes(gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight, partialBlock, blockRes,
                             projWeight, normWeight);
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnBlockAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight, const aclTensor *normWeight,
    const aclTensor *gradHiddenStates, const aclTensor *invNorm, const aclTensor *probs, int64_t validBlockNum,
    const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes, const aclTensor *gradProjWeight,
    const aclTensor *gradNormWeight, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnBlockAttentionResidualsGrad,
        DFX_IN(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, validBlockNum),
        DFX_OUT(gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclnnStatus checkRet =
        CheckParams(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, validBlockNum,
                    gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight, workspaceSize, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    bool emptyHandled = false;
    const aclnnStatus emptyRet = HandleEmptyTensor(
        partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, gradPartialBlock,
        gradBlockRes, gradProjWeight, gradNormWeight, uniqueExecutor, workspaceSize, executor, emptyHandled);
    CHECK_RET(emptyRet == ACLNN_SUCCESS, emptyRet);
    if (emptyHandled) {
        return ACLNN_SUCCESS;
    }

    // 与正向算子对齐：输入统一转 Contiguous 后再下发
    auto partialBlock_ = l0op::Contiguous(partialBlock, uniqueExecutor.get());
    CHECK_RET(partialBlock_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto blockRes_ = l0op::Contiguous(blockRes, uniqueExecutor.get());
    CHECK_RET(blockRes_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto projWeight_ = l0op::Contiguous(projWeight, uniqueExecutor.get());
    CHECK_RET(projWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto normWeight_ = l0op::Contiguous(normWeight, uniqueExecutor.get());
    CHECK_RET(normWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradHiddenState_ = l0op::Contiguous(gradHiddenStates, uniqueExecutor.get());
    CHECK_RET(gradHiddenState_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invNorm_ = l0op::Contiguous(invNorm, uniqueExecutor.get());
    CHECK_RET(invNorm_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto probs_ = l0op::Contiguous(probs, uniqueExecutor.get());
    CHECK_RET(probs_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::BlockAttentionResidualsGrad(
        partialBlock_, blockRes_, projWeight_, normWeight_, gradHiddenState_, invNorm_, probs_, validBlockNum,
        const_cast<aclTensor *>(gradPartialBlock), const_cast<aclTensor *>(gradBlockRes),
        const_cast<aclTensor *>(gradProjWeight), const_cast<aclTensor *>(gradNormWeight), uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBlockAttentionResidualsGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "BlockAttentionResidualsGrad launch failed, ret = %d.", ret);
        return ret;
    }
    return ret;
}

#ifdef __cplusplus
}
#endif
