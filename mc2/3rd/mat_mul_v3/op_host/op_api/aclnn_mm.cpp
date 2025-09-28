/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_mm.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "common/op_host/op_api/cube_util.h"
#include "common/op_host/op_api/matmul_util.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "common/op_api_def.h"

using namespace op;
using namespace std;
using namespace Ops::Transformer;

namespace {
static const int64_t DIMS_TWO = 2;
static const int64_t M_DIM_SELF_IDX = 0;
static const int64_t K_DIM_SELF_IDX = 1;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = (mat2Float || selfFloat) ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mat2, return false);
    return true;
}

static inline bool CheckKEqual1Support(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E &&
           GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType)
{
    bool bf16flag = CheckSocVersionIsSupportBf16();
    if (bf16flag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
        }
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        }
    }

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat2->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(mat2->GetDataType()).GetString());
        return false;
    }
    if (out != nullptr && cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
        self->GetDataType() == op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
        return false;
    }
    // self和mat2的dtype不相等时，会做promote处理。
    bool dtype_match = self->GetDataType() == mat2->GetDataType();
    if (!dtype_match) {
        OP_LOGW(
            "Self's dtype [%s] and mat2's dtype [%s] are not equal. Promotion of Data Type will be applied",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
    }
    return true;
}

static void UpdateCubeMathType(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t& cubeMathType)
{
    if (self->GetDataType() == op::DataType::DT_FLOAT16 && out->GetDataType() == op::DataType::DT_FLOAT) {
        bool ndNdNd =
            (self->GetStorageFormat() == op::Format::FORMAT_ND && mat2->GetStorageFormat() == op::Format::FORMAT_ND &&
             out->GetStorageFormat() == op::Format::FORMAT_ND);
        bool nzNzNz =
            (self->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ &&
             mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ &&
             out->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ);
        if (ndNdNd || nzNzNz) {
            OP_LOGI("Set cubeMathType for fp16 in fp32 out: %d, ori: %d", FP16FP32_KEEP_DTYPE, cubeMathType);
            cubeMathType = FP16FP32_KEEP_DTYPE;
        }
    }
}

inline static bool CheckOutputDtypeValid(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    if (self->GetDataType() == op::DataType::DT_FLOAT16 && out->GetDataType() == op::DataType::DT_FLOAT) {
        bool ndNdNd =
            (self->GetStorageFormat() == op::Format::FORMAT_ND && mat2->GetStorageFormat() == op::Format::FORMAT_ND &&
             out->GetStorageFormat() == op::Format::FORMAT_ND);
        bool nzNzNz =
            (self->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ &&
             mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ &&
             out->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ);
        if (!ndNdNd && !nzNzNz) {
            // fp16 in fp32 out that is split k template, not precision-advanced now
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Split k only support ndnd_nd or nznz_nz.");
            return false;
        }
    } else {
        OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* self, const aclTensor* mat2, bool transposeX2 = false)
{
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    int64_t mat2KDim = transposeX2 ? mat2Shape.GetDim(K_DIM_SELF_IDX) : mat2Shape.GetDim(M_DIM_SELF_IDX);
    int64_t selfKDim = selfShape.GetDim(K_DIM_SELF_IDX); // self固定不转置
    if (mat2KDim != selfKDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different, self Kdim[%ld], mat2 Kdim[%ld].",
            selfKDim, mat2KDim);
        return false;
    }
    return true;
}

static bool CheckOutputShapeValid(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(out, DIMS_TWO, return false);
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape outShape = out->GetViewShape();
    if (outShape[0] != selfShape[0] || outShape[1] != mat2Shape[1]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "output's shape is not match input, "
            "out_m[%ld] must be same with self_m[%ld], out_n[%ld] must be same with mat2_n[%ld].",
            outShape[0], selfShape[0], outShape[1], mat2Shape[1]);
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParam(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验。
    CHECK_RET(CheckDtypeValid(self, mat2, nullptr, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckOutputDtypeValid(self, mat2, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查OutputShape是否支持
    CHECK_RET(CheckOutputShapeValid(self, mat2, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查cubeMathType
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnMmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, size_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMm, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParam(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // mm/bmm
    UpdateCubeMathType(self, mat2, out, cubeMathType);
    auto matmulOut = ExecMmOp(self, mat2, cubeMathType, unique_executor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0UL;
        unique_executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace
    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
