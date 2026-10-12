/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ring_attention_update.h"
#include "ring_attention_update.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const int64_t SOFTMAX_HEAD_DIM_NUM = 8;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const uint32_t SIZE_B32 = 4;
static const uint32_t SIZE_B16 = 2;
static const uint64_t MAX_HEAD_DIM_LIMIT_SIZE = 768;
static const uint32_t REPEAT_NUM_B32 = 64;
static const uint64_t MAX_HEAD_NUM_LIMIT_SIZE = 256;
#if (defined(NPU_ARCH) && (NPU_ARCH == 3003 || NPU_ARCH == 3113))
static const uint64_t MAX_UB_SIZE = 98304;
#else
static const uint64_t MAX_UB_SIZE = 196608;
#endif

static aclnnStatus CheckUpdateParam(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                    const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                    const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                    const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                    const aclTensor* softmaxSumOut, const uint64_t* workspaceSize,
                                    aclOpExecutor** executor)
{
    // 必须的参数指针判空
    CHECK_RET(prevAttnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(prevSoftmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(prevSoftmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(curAttnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(curSoftmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(curSoftmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attnOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxMaxOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxSumOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus InputDtypeCheck(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                   const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                   const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                   const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                   const aclTensor* softmaxSumOut)
{
    auto prevAttnDtype = prevAttnOut->GetDataType();
    auto prevMaxDtype = prevSoftmaxMax->GetDataType();
    auto prevSumDtype = prevSoftmaxSum->GetDataType();
    auto curAttnDtype = curAttnOut->GetDataType();
    auto curMaxDtype = curSoftmaxMax->GetDataType();
    auto curSumDtype = curSoftmaxSum->GetDataType();
    auto attnDtype = attnOutOut->GetDataType();
    auto maxDtype = softmaxMaxOut->GetDataType();
    auto sumDtype = softmaxSumOut->GetDataType();
    if (prevAttnDtype != curAttnDtype || attnDtype != prevAttnDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The data type of prevAttnOut[%s], curAttnOut[%s], attnOutOut[%s] are not equal.",
                op::ToString(DataType(prevAttnDtype)).GetString(), op::ToString(DataType(curAttnDtype)).GetString(),
                op::ToString(DataType(attnDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(prevAttnDtype == op::DataType::DT_FLOAT || prevAttnDtype == op::DataType::DT_FLOAT16 ||
          prevAttnDtype == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The data type of prevAttnOut/curAttnOut/attnOutOut is [%s], should be fp16, bf16 or fp32.",
                op::ToString(DataType(prevAttnDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!(prevMaxDtype == op::DataType::DT_FLOAT || prevSumDtype == op::DataType::DT_FLOAT ||
          curMaxDtype == op::DataType::DT_FLOAT || curSumDtype == op::DataType::DT_FLOAT ||
          maxDtype == op::DataType::DT_FLOAT || sumDtype == op::DataType::DT_FLOAT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The data type of prevSoftmaxMax[%s]/prevSoftmaxSum[%s]/curSoftmaxMax[%s]/"
                "curSoftmaxSum[%s]/softmaxMaxOut[%s]/softmaxSumOut[%s] should all be fp32.",
                op::ToString(DataType(prevMaxDtype)).GetString(), op::ToString(DataType(prevSumDtype)).GetString(),
                op::ToString(DataType(curMaxDtype)).GetString(), op::ToString(DataType(curSumDtype)).GetString(),
                op::ToString(DataType(maxDtype)).GetString(), op::ToString(DataType(sumDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus InputFormatCheck(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                    const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                    const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                    const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                    const aclTensor* softmaxSumOut, const aclTensor* actualSeqQlenOptional,
                                    const char* inputLayoutOptional)
{
    std::string inputLayoutStr = op::ToString(inputLayoutOptional).GetString();
    bool formatValid = prevAttnOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       prevSoftmaxMax->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       prevSoftmaxSum->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       curAttnOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       curSoftmaxMax->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       curSoftmaxSum->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       attnOutOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       softmaxMaxOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                       softmaxSumOut->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ;
    if (!formatValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input and output format only support [ND]. Actual: prevAttnOut:[%s], prevSoftmaxMax:[%s], "
                "prevSoftmaxSum:[%s], curAttnOut:[%s], curSoftmaxMax:[%s], curSoftmaxSum:[%s], attnOutOut:[%s], "
                "softmaxMaxOut:[%s], softmaxSumOut:[%s].",
                op::ToString(prevAttnOut->GetStorageFormat()).GetString(),
                op::ToString(prevSoftmaxMax->GetStorageFormat()).GetString(),
                op::ToString(prevSoftmaxSum->GetStorageFormat()).GetString(),
                op::ToString(curAttnOut->GetStorageFormat()).GetString(),
                op::ToString(curSoftmaxMax->GetStorageFormat()).GetString(),
                op::ToString(curSoftmaxSum->GetStorageFormat()).GetString(),
                op::ToString(attnOutOut->GetStorageFormat()).GetString(),
                op::ToString(softmaxMaxOut->GetStorageFormat()).GetString(),
                op::ToString(softmaxSumOut->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (inputLayoutStr == "TND") {
        formatValid = (formatValid && actualSeqQlenOptional->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ);
    }
    if (!formatValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input and output format only support [ND]. Actual: actualSeqQlenOptional:[%s].",
                op::ToString(actualSeqQlenOptional->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool CheckNullTensor(const Shape checkShape)
{
    size_t shapeSize = checkShape.GetDimNum();
    for (size_t dimIndex = 0; dimIndex < shapeSize; ++dimIndex) {
        if (checkShape.GetDim(dimIndex) == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor shape dim(%ld) is 0, the shape is not supported.", dimIndex);
            return false;
        }
    }
    return true;
}

static aclnnStatus AnalysisAxis(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum, const char* inputLayout)
{
    Shape paShape = prevAttnOut->GetViewShape();
    Shape pMaxShape = prevSoftmaxMax->GetViewShape();
    Shape pSumShape = prevSoftmaxSum->GetViewShape();
    Shape caShape = curAttnOut->GetViewShape();
    Shape cMaxShape = curSoftmaxMax->GetViewShape();
    Shape cSumShape = curSoftmaxSum->GetViewShape();
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();

    CHECK_RET(CheckNullTensor(paShape), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNullTensor(pMaxShape), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNullTensor(pSumShape), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNullTensor(caShape), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNullTensor(cMaxShape), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNullTensor(cSumShape), ACLNN_ERR_PARAM_INVALID);

    if (paShape.GetDimNum() != DIM_NUM_3 || caShape.GetDimNum() != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the shape of prevAttnOut(%ld) or curAttnOut(%ld) is not supported.",
                paShape.GetDimNum(), caShape.GetDimNum());
    }

    int headDimIndex = 0;
    if (inputLayoutStr == "SBH" && pMaxShape.GetDimNum() == DIM_NUM_4 && pSumShape.GetDimNum() == DIM_NUM_4 &&
        cMaxShape.GetDimNum() == DIM_NUM_4 && cSumShape.GetDimNum() == DIM_NUM_4) {
        headDimIndex = DIM_NUM_3;
    } else if (inputLayoutStr == "TND" && pMaxShape.GetDimNum() == DIM_NUM_3 && pSumShape.GetDimNum() == DIM_NUM_3 &&
               cMaxShape.GetDimNum() == DIM_NUM_3 && cSumShape.GetDimNum() == DIM_NUM_3) {
        headDimIndex = DIM_NUM_2;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "not support input_layout %s with dim_num prevSoftmaxMax(%lu)/prevSoftmaxSum(%lu)/"
                "curSoftmaxMax(%lu)/curSoftmaxSum(%lu)",
                inputLayout, pMaxShape.GetDimNum(), pSumShape.GetDimNum(), cMaxShape.GetDimNum(),
                cSumShape.GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (pMaxShape[headDimIndex] != SOFTMAX_HEAD_DIM_NUM || pSumShape[headDimIndex] != SOFTMAX_HEAD_DIM_NUM ||
        cMaxShape[headDimIndex] != SOFTMAX_HEAD_DIM_NUM || cSumShape[headDimIndex] != SOFTMAX_HEAD_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "the last dim of softmax_inputs(prevSoftmaxMax(%lu)/prevSoftmaxSum(%lu)/curSoftmaxMax(%lu)/"
                "curSoftmaxSum(%lu)) is not 8.",
                pMaxShape[headDimIndex], pSumShape[headDimIndex], cMaxShape[headDimIndex], cSumShape[headDimIndex]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (inputLayoutStr == "TND") {
        // 判断prevAttnOut数据类型
        const auto prevAttnDtype = prevAttnOut->GetDataType();
        uint32_t inputDataSize = 0;

        if (prevAttnDtype == op::DataType::DT_FLOAT) {
            inputDataSize = SIZE_B32;
        } else if (prevAttnDtype == op::DataType::DT_FLOAT16 || prevAttnDtype == op::DataType::DT_BF16) {
            inputDataSize = SIZE_B16;
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dtype only support fp16, fp32, bf16 currently, but got Dtype = %s",
                    op::ToString(DataType(prevAttnDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        const int64_t headNum = paShape.GetDim(1);
        const int64_t headDim = paShape.GetDim(2);

        if (headNum > MAX_HEAD_NUM_LIMIT_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Don't support this shape currently, please try to set N <= %lu!",
                    MAX_HEAD_NUM_LIMIT_SIZE);
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (headDim > MAX_HEAD_DIM_LIMIT_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Don't support this shape currently, please try to set D <= %lu!",
                    MAX_HEAD_DIM_LIMIT_SIZE);
            return ACLNN_ERR_PARAM_INVALID;
        }

        const int64_t headNumAllCount = MAX_UB_SIZE - REPEAT_NUM_B32 * (inputDataSize * 6 + 8) - REPEAT_NUM_B32 * 56;
        const int64_t headDimEach = headDim * (inputDataSize * 6 + 8) + 8 * 56;
        const int64_t loopEachMax = headNumAllCount / headDimEach;

        const int64_t headNumLoopEach = (headNum == 1) ? loopEachMax : std::min(loopEachMax, headNum);

        if (headNumLoopEach == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Don't support this shape currently, please try to use smaller D!");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus Contiguous(const aclTensor*& prevAttnOut, const aclTensor*& prevSoftmaxMax,
                              const aclTensor*& prevSoftmaxSum, const aclTensor*& curAttnOut,
                              const aclTensor*& curSoftmaxMax, const aclTensor*& curSoftmaxSum,
                              const aclTensor*& actualSeqQlenOptional, const char* inputLayoutOptional,
                              aclOpExecutor* executor)
{
    std::string inputLayoutStr = op::ToString(inputLayoutOptional).GetString();
    prevAttnOut = l0op::Contiguous(prevAttnOut, executor);
    CHECK_RET(prevAttnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    prevSoftmaxMax = l0op::Contiguous(prevSoftmaxMax, executor);
    CHECK_RET(prevSoftmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    prevSoftmaxSum = l0op::Contiguous(prevSoftmaxSum, executor);
    CHECK_RET(prevSoftmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    curAttnOut = l0op::Contiguous(curAttnOut, executor);
    CHECK_RET(curAttnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    curSoftmaxMax = l0op::Contiguous(curSoftmaxMax, executor);
    CHECK_RET(curSoftmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    curSoftmaxSum = l0op::Contiguous(curSoftmaxSum, executor);
    CHECK_RET(curSoftmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (inputLayoutStr == "TND") {
        actualSeqQlenOptional = l0op::Contiguous(actualSeqQlenOptional, executor);
        CHECK_RET(actualSeqQlenOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                                     const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                                     const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                                     const aclTensor* actualSeqQlenOptional, char* inputLayoutOptional,
                                                     const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                                     const aclTensor* softmaxSumOut, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    CHECK_RET(CheckUpdateParam(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                               attnOutOut, softmaxMaxOut, softmaxSumOut, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnRingAttentionUpdate,
                   DFX_IN(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                          actualSeqQlenOptional, inputLayoutOptional),
                   DFX_OUT(attnOutOut, softmaxMaxOut, softmaxSumOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(InputDtypeCheck(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                              attnOutOut, softmaxMaxOut, softmaxSumOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisAxis(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                           inputLayoutOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor* l0Executor = uniqueExecutor.get();
    CHECK_RET(Contiguous(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                         actualSeqQlenOptional, inputLayoutOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(InputFormatCheck(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                               attnOutOut, softmaxMaxOut, softmaxSumOut, actualSeqQlenOptional,
                               inputLayoutOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto l0RingAttentionUpdateOuts =
        l0op::RingAttentionUpdate(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                                  actualSeqQlenOptional, inputLayoutOptional, "", l0Executor);
    auto l0AttentionOutOut = l0RingAttentionUpdateOuts[0];
    auto l0SoftmaxMaxOut = l0RingAttentionUpdateOuts[1];
    auto l0SoftmaxSumOut = l0RingAttentionUpdateOuts[2];

    if (l0AttentionOutOut == nullptr || l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0AttentionOutOut or l0SoftmaxMaxOut or l0SoftmaxSumOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0AttentionOutOut, attnOutOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRingAttentionUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRingAttentionUpdate);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnRingAttentionUpdateV2GetWorkspaceSize(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                                       const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                                       const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                                       const aclTensor* actualSeqQlenOptional,
                                                       char* inputLayoutOptional, char* inputSoftmaxOutLayout,
                                                       const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                                       const aclTensor* softmaxSumOut, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    CHECK_RET(CheckUpdateParam(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                               attnOutOut, softmaxMaxOut, softmaxSumOut, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnRingAttentionUpdateV2,
                   DFX_IN(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                          actualSeqQlenOptional, inputLayoutOptional, inputSoftmaxOutLayout),
                   DFX_OUT(attnOutOut, softmaxMaxOut, softmaxSumOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(InputDtypeCheck(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                              attnOutOut, softmaxMaxOut, softmaxSumOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisAxis(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                           inputLayoutOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor* l0Executor = uniqueExecutor.get();
    CHECK_RET(Contiguous(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                         actualSeqQlenOptional, inputLayoutOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(InputFormatCheck(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                               attnOutOut, softmaxMaxOut, softmaxSumOut, actualSeqQlenOptional,
                               inputLayoutOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto l0RingAttentionUpdateOuts =
        l0op::RingAttentionUpdate(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
                                  actualSeqQlenOptional, inputLayoutOptional, inputSoftmaxOutLayout, l0Executor);
    auto l0AttentionOutOut = l0RingAttentionUpdateOuts[0];
    auto l0SoftmaxMaxOut = l0RingAttentionUpdateOuts[1];
    auto l0SoftmaxSumOut = l0RingAttentionUpdateOuts[2];

    if (l0AttentionOutOut == nullptr || l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0AttentionOutOut or l0SoftmaxMaxOut or l0SoftmaxSumOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0AttentionOutOut, attnOutOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRingAttentionUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRingAttentionUpdateV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
} // namespace

#ifdef __cplusplus
}
#endif
