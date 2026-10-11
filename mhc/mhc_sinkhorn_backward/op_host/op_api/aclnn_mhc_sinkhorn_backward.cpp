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
 * \file aclnn_mhc_sinkhorn_backward.cpp
 * \brief aclnn_mhc_sinkhorn_backward
 */

#include "aclnn_mhc_sinkhorn_backward.h"
#include "mhc_sinkhorn_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t SUPPORT_DIM_NUM_1 = 1;
static constexpr size_t SUPPORT_DIM_NUM_3 = 3;
static constexpr size_t SUPPORT_DIM_NUM_4 = 4;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* normOut, const aclTensor* sumOut,
                         const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(normOut, return false);
    OP_CHECK_NULL(sumOut, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* normOut, const aclTensor* sumOut,
                            const aclTensor* out)
{
    // 检查x的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    // 检查normOut的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(normOut, DTYPE_SUPPORT_LIST, return false);
    // 检查sumOut的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(sumOut, DTYPE_SUPPORT_LIST, return false);
    // 检查out的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* out)
{
    // 输入输出的格式需要一致
    if (gradOutput->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. gradOutput [%s], out [%s].",
                ToString(gradOutput->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* normOut, const aclTensor* sumOut,
                       const aclTensor* out)
{
    // 校验gradOutput的shape是否等于out的shape
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, out, return false);

    // gradOutput维度必须是3或4
    auto gradOutputShape = gradOutput->GetViewShape();
    auto gradOutputDim = gradOutputShape.GetDimNum();
    if (gradOutputDim != SUPPORT_DIM_NUM_3 && gradOutputDim != SUPPORT_DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input gradOutput dim must be 3 or 4, but got Dim = %ld .",
                gradOutputDim);
        return false;
    }
    // gradOutput最后两维的大小n0和n1满足：n0等于n1 且n0为4,6,8
    int64_t n0 = 0;
    int64_t n1 = 0;
    if (gradOutputDim == SUPPORT_DIM_NUM_3) {
        n0 = gradOutputShape.GetDim(DIM_ONE);
        n1 = gradOutputShape.GetDim(DIM_TWO);
    } else if (gradOutputDim == SUPPORT_DIM_NUM_4) {
        n0 = gradOutputShape.GetDim(DIM_TWO);
        n1 = gradOutputShape.GetDim(DIM_THREE);
    }
    if ((n0 != n1) || (n0 != N_VALID_4 && n0 != N_VALID_6 && n0 != N_VALID_8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "n0 must equal n1, and n must be %ld or %ld or %ld, but got n0 = %ld, n1 = %ld", N_VALID_4, N_VALID_6,
                N_VALID_8, n0, n1);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* normOut, const aclTensor* sumOut,
                                      const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, normOut, sumOut, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(gradOutput, normOut, sumOut, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(gradOutput, normOut, sumOut, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入输出format是否一致
    CHECK_RET(CheckFormat(gradOutput, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhornBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* normOut,
                                                     const aclTensor* sumOut, aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMhcSinkhornBackward, DFX_IN(gradOutput, normOut, sumOut), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, normOut, sumOut, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入转换成连续的tensor
    const aclTensor* gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* normOutContiguous = l0op::Contiguous(normOut, uniqueExecutor.get());
    CHECK_RET(normOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* sumOutContiguous = l0op::Contiguous(sumOut, uniqueExecutor.get());
    CHECK_RET(sumOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输出转换成连续的tensor
    aclTensor* outContiguous = const_cast<aclTensor*>(l0op::Contiguous(out, uniqueExecutor.get()));
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* kernelOut = l0op::MhcSinkhornBackward(gradOutputContiguous, normOutContiguous, sumOutContiguous,
                                                           outContiguous, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出outRef上
    auto viewCopyResult = l0op::ViewCopy(kernelOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhornBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcSinkhornBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
