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
 * \file aclnn_mhc_sinkhorn.cpp
 * \brief aclnn_mhc_sinkhorn
 */

#include "aclnn_mhc_sinkhorn.h"
#include "mhc_sinkhorn.h"
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
static constexpr size_t MIN_NUMITERS = 0;
static constexpr size_t MAX_NUMITERS = 100;
static constexpr size_t SUPPORT_DIM_NUM_3 = 3;
static constexpr size_t SUPPORT_DIM_NUM_4 = 4;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;
static constexpr int64_t N_ALIGN = 8;
static constexpr int64_t DOUBLE_SIZE = 2;

static bool CheckNotNull(const aclTensor *x, const aclTensor *output)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                            const aclTensor *sumOut)
{
    // 检查x的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    // 检查output的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
    if (outFlag) {
        // 检查normOut的数据类型是否在算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(normOut, DTYPE_SUPPORT_LIST, return false);
        // 检查sumOut的数据类型是否在算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(sumOut, DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                        const aclTensor *sumOut)
{
    // 输入输出的格式需要一致
    if (x->GetStorageFormat() != output->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same. x [%s], output [%s].",
                ToString(x->GetStorageFormat()).GetString(), ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *x, int64_t outFlag, const aclTensor *output, const aclTensor *normOut,
                       const aclTensor *sumOut, int64_t numIters)
{
    // 校验self的shape是否等于out的shape
    OP_CHECK_SHAPE_NOT_EQUAL(x, output, return false);

    // numIters在1~100范围内
    if (numIters <= MIN_NUMITERS || numIters > MAX_NUMITERS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numIters value error, numIters must in 1 to 100, but got numIters = %ld .",
                numIters);
        return false;
    }

    // outFlag为0或1
    if (outFlag != 0 && outFlag != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outFlag value error, outFlag must be 0 or 1, but got outFlag = %ld .",
                outFlag);
        return false;
    }

    // 维度必须是3或4
    auto xShape = x->GetViewShape();
    auto xDim = xShape.GetDimNum();
    if (xDim != SUPPORT_DIM_NUM_3 && xDim != SUPPORT_DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim must be 3 or 4, but got Dim = %ld .", xDim);
        return false;
    }

    // n0等于n1 且n为4,6,8
    int64_t n0 = 0;
    int64_t n1 = 0;
    if (xDim == SUPPORT_DIM_NUM_3) {
        n0 = xShape.GetDim(DIM_ONE);
        n1 = xShape.GetDim(DIM_TWO);
    } else if (xDim == SUPPORT_DIM_NUM_4) {
        n0 = xShape.GetDim(DIM_TWO);
        n1 = xShape.GetDim(DIM_THREE);
    }
    if ((n0 != n1) || (n0 != N_VALID_4 && n0 != N_VALID_6 && n0 != N_VALID_8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "n0 must equal n1, and n must be %ld or %ld or %ld, but got n0 = %ld, n1 = %ld", N_VALID_4, N_VALID_6,
                N_VALID_8, n0, n1);
        return false;
    }

    // outFlag为1时，校验normOut和sumOut的元素总数是否符合预期
    if (outFlag) {
        int64_t T = xShape.GetDim(DIM_ZERO);
        if (xDim == SUPPORT_DIM_NUM_4) {
            T = xShape.GetDim(DIM_ZERO) * xShape.GetDim(DIM_ONE);
        }
        int64_t n = n0;
        int64_t expectNormSize = DOUBLE_SIZE * numIters * T * n * N_ALIGN;
        int64_t expectSumSize = DOUBLE_SIZE * numIters * T * N_ALIGN;
        if (normOut->GetViewShape().GetDimNum() != 1 || normOut->GetViewShape().GetShapeSize() != expectNormSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "normOut shape must be 1-dim and size %ld, but got dim=%ld size=%ld.",
                    expectNormSize, normOut->GetViewShape().GetDimNum(), normOut->GetViewShape().GetShapeSize());
            return false;
        }
        if (sumOut->GetViewShape().GetDimNum() != 1 || sumOut->GetViewShape().GetShapeSize() != expectSumSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sumOut shape must be 1-dim and size %ld, but got dim=%ld size=%ld.",
                    expectSumSize, sumOut->GetViewShape().GetDimNum(), sumOut->GetViewShape().GetShapeSize());
            return false;
        }
    } else {
        // outFlag为0时，normOut和sumOut应为shape{1}的空输出
        if (normOut->GetViewShape().GetDimNum() != 1 || normOut->GetViewShape().GetShapeSize() != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outFlag is 0, normOut shape must be {1}, but got dim=%ld size=%ld.",
                    normOut->GetViewShape().GetDimNum(), normOut->GetViewShape().GetShapeSize());
            return false;
        }
        if (sumOut->GetViewShape().GetDimNum() != 1 || sumOut->GetViewShape().GetShapeSize() != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outFlag is 0, sumOut shape must be {1}, but got dim=%ld size=%ld.",
                    sumOut->GetViewShape().GetDimNum(), sumOut->GetViewShape().GetShapeSize());
            return false;
        }
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *x, int64_t outFlag, float eps, int64_t numIters,
                                      const aclTensor *output, const aclTensor *normOut, const aclTensor *sumOut)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, outFlag, output, normOut, sumOut), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入形状是否满足
    CHECK_RET(CheckShape(x, outFlag, output, normOut, sumOut, numIters), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入输出format是否一致
    CHECK_RET(CheckFormat(x, outFlag, output, normOut, sumOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor *x, float eps, int64_t numIters, aclTensor *output,
                                             aclTensor *normOut, aclTensor *sumOut, uint64_t *workspaceSize,
                                             aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcSinkhorn, DFX_IN(x, eps, numIters), DFX_OUT(output, normOut, sumOut));

    CHECK_RET(CheckNotNull(x, output), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 通过shape判断outFlag: shape为{1}或nullptr视为辅助输出无效(outFlag=0)
    int64_t outFlag = 1;
    bool normEmpty = (normOut == nullptr) || (normOut->GetViewShape().GetShapeSize() == 1);
    bool sumEmpty = (sumOut == nullptr) || (sumOut->GetViewShape().GetShapeSize() == 1);
    if (normEmpty && sumEmpty) {
        Shape emptyShape({1});
        if (normOut == nullptr) {
            normOut = (uniqueExecutor.get())->AllocTensor(emptyShape, x->GetDataType(), Format::FORMAT_ND);
            CHECK_RET(normOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        if (sumOut == nullptr) {
            sumOut = (uniqueExecutor.get())->AllocTensor(emptyShape, x->GetDataType(), Format::FORMAT_ND);
            CHECK_RET(sumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        outFlag = 0;
    } else if (!normEmpty && !sumEmpty) {
        outFlag = 1;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "normOut and sumOut must both be valid (shape != {1}) or both be empty (shape {1} or nullptr), "
                "but got normOut shape size %ld, sumOut shape size %ld.",
                normOut ? normOut->GetViewShape().GetShapeSize() : 0,
                sumOut ? sumOut->GetViewShape().GetShapeSize() : 0);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，参数检查
    auto ret = CheckParams(x, outFlag, eps, numIters, output, normOut, sumOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support empty input tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 将输入x转换成连续的tensor
    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor *outputContiguous = const_cast<aclTensor *>(l0op::Contiguous(output, uniqueExecutor.get()));
    CHECK_RET(outputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut =
        l0op::MhcSinkhorn(xContiguous, outFlag, eps, numIters, outputContiguous, normOut, sumOut, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出outRef上
    auto viewCopyResult = l0op::ViewCopy(kernelOut, output, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcSinkhorn);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
