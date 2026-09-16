/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kda_input_proj.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {

extern "C" aclnnStatus aclnnInnerKdaInputProjGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta, const aclTensor *weightGate,
    const aclTensor *weightG, const aclTensor *weightQkvScale, bool transWeightQkv, bool transWeightBeta,
    bool transWeightGate, bool transWeightG, const aclTensor *qkvOut, const aclTensor *betaOut,
    const aclTensor *gateOut, const aclTensor *gOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerKdaInputProj(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

constexpr int64_t LAST_DIM = 1;
constexpr int64_t SECOND_LAST_DIM = 2;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_IDX_ZERO = 0;
constexpr size_t DIM_IDX_ONE = 1;

op::Shape SwapLastTwoDimValue(const op::Shape &tensorShape)
{
    op::Shape swapped = tensorShape;
    const int64_t dimNum = tensorShape.GetDimNum();
    if (dimNum >= SECOND_LAST_DIM) {
        const int64_t last = tensorShape.GetDim(dimNum - LAST_DIM);
        swapped.SetDim(dimNum - LAST_DIM, tensorShape.GetDim(dimNum - SECOND_LAST_DIM));
        swapped.SetDim(dimNum - SECOND_LAST_DIM, last);
    }
    return swapped;
}

bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const int64_t dimNum = tensor->GetViewShape().GetDimNum();
    if (dimNum < SECOND_LAST_DIM) {
        return false;
    }
    const auto &strides = tensor->GetViewStrides();
    if (static_cast<int64_t>(strides.size()) < dimNum) {
        return false;
    }
    const int64_t dim1 = dimNum - LAST_DIM;
    const int64_t dim2 = dimNum - SECOND_LAST_DIM;
    if (strides[dim2] == 1 && strides[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

void NormalizeTransposedWeight(aclTensor *weight)
{
    // MatMulV3 WeightNz: only swap view shape; keep original strides (column-major of [N,K]).
    weight->SetViewShape(SwapLastTwoDimValue(weight->GetViewShape()));
}

const aclTensor *PrepareWeightForInner(const aclTensor *weight, bool &trans)
{
    trans = IsTransposeLastTwoDims(weight);
    if (!trans) {
        return weight;
    }
    aclTensor *mut = const_cast<aclTensor *>(weight);
    NormalizeTransposedWeight(mut);
    return mut;
}

inline static bool CheckNotNull(const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta,
                                const aclTensor *weightGate, const aclTensor *weightG, const aclTensor *weightQkvScale,
                                const aclTensor *qkvOut, const aclTensor *betaOut, const aclTensor *gateOut,
                                const aclTensor *gOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weightQkv, return false);
    OP_CHECK_NULL(weightBeta, return false);
    OP_CHECK_NULL(weightGate, return false);
    OP_CHECK_NULL(weightG, return false);
    OP_CHECK_NULL(weightQkvScale, return false);
    OP_CHECK_NULL(qkvOut, return false);
    OP_CHECK_NULL(betaOut, return false);
    OP_CHECK_NULL(gateOut, return false);
    OP_CHECK_NULL(gOut, return false);
    return true;
}

inline static bool CheckDtype(const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta,
                              const aclTensor *weightGate, const aclTensor *weightG, const aclTensor *weightQkvScale,
                              const aclTensor *qkvOut, const aclTensor *betaOut, const aclTensor *gateOut,
                              const aclTensor *gOut)
{
    OP_CHECK_DTYPE_NOT_MATCH(x, DataType::DT_BF16, return false);
    OP_CHECK_DTYPE_NOT_MATCH(weightQkv, DataType::DT_FLOAT8_E4M3FN, return false);
    OP_CHECK_DTYPE_NOT_MATCH(weightBeta, DataType::DT_BF16, return false);
    OP_CHECK_DTYPE_NOT_MATCH(weightGate, DataType::DT_BF16, return false);
    OP_CHECK_DTYPE_NOT_MATCH(weightG, DataType::DT_BF16, return false);
    OP_CHECK_DTYPE_NOT_MATCH(weightQkvScale, DataType::DT_FLOAT8_E8M0, return false);
    OP_CHECK_DTYPE_NOT_MATCH(qkvOut, DataType::DT_BF16, return false);
    OP_CHECK_DTYPE_NOT_MATCH(betaOut, DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gateOut, DataType::DT_BF16, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gOut, DataType::DT_BF16, return false);
    return true;
}

inline static bool IsNdLikeFormat(op::Format format)
{
    // Torch 扩展按维数映射 ND 等价 format：2D→ND，3D→NCL，4D→NCHW。
    return format == Format::FORMAT_ND || format == Format::FORMAT_NCL || format == Format::FORMAT_NCHW ||
           format == Format::FORMAT_NHWC;
}

inline static bool CheckNdFormat(const aclTensor *tensor)
{
    const auto format = tensor->GetStorageFormat();
    if (!IsNdLikeFormat(format)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "KdaInputProj only supports ND format, but got format %s.",
                op::ToString(format).GetString());
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta,
                        const aclTensor *weightGate, const aclTensor *weightG, const aclTensor *weightQkvScale,
                        const aclTensor *qkvOut, const aclTensor *betaOut, const aclTensor *gateOut,
                        const aclTensor *gOut)
{
    return CheckNdFormat(x) && CheckNdFormat(weightQkv) && CheckNdFormat(weightBeta) && CheckNdFormat(weightGate) &&
           CheckNdFormat(weightG) && CheckNdFormat(weightQkvScale) && CheckNdFormat(qkvOut) && CheckNdFormat(betaOut) &&
           CheckNdFormat(gateOut) && CheckNdFormat(gOut);
}

// Public weight view is [K, N]. K must equal x.dim1; out must be [M, N] = [x.dim0, weight.dim1].
static bool CheckWeightKAndOut(const aclTensor *x, const aclTensor *weight, const aclTensor *out,
                               const char *weightName, const char *outName)
{
    const op::Shape &xShape = x->GetViewShape();
    const op::Shape &wShape = weight->GetViewShape();
    OP_CHECK_WRONG_DIMENSION(weight, DIM_NUM_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_NUM_TWO, return false);

    const int64_t mDim = xShape.GetDim(DIM_IDX_ZERO);
    const int64_t kDim = xShape.GetDim(DIM_IDX_ONE);
    const int64_t weightK = wShape.GetDim(DIM_IDX_ZERO);
    const int64_t nDim = wShape.GetDim(DIM_IDX_ONE);
    if (weightK != kDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of %s %s does not match x %s.", weightName,
                op::ToString(wShape).GetString(), op::ToString(xShape).GetString());
        return false;
    }

    op::Shape expectOut;
    expectOut.SetDimNum(DIM_NUM_TWO);
    expectOut.SetDim(DIM_IDX_ZERO, mDim);
    expectOut.SetDim(DIM_IDX_ONE, nDim);
    if (out->GetViewShape() != expectOut) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected %s shape [%ld, %ld] (M=x.dim0, N=%s.dim1), but got %s.", outName,
                mDim, nDim, weightName, op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool CheckShapeValid(const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta,
                            const aclTensor *weightGate, const aclTensor *weightG, const aclTensor *qkvOut,
                            const aclTensor *betaOut, const aclTensor *gateOut, const aclTensor *gOut)
{
    OP_CHECK_WRONG_DIMENSION(x, DIM_NUM_TWO, return false);
    return CheckWeightKAndOut(x, weightQkv, qkvOut, "weightQkv", "qkvOut") &&
           CheckWeightKAndOut(x, weightBeta, betaOut, "weightBeta", "betaOut") &&
           CheckWeightKAndOut(x, weightGate, gateOut, "weightGate", "gateOut") &&
           CheckWeightKAndOut(x, weightG, gOut, "weightG", "gOut");
}

inline static aclnnStatus CheckInputParams(const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta,
                                           const aclTensor *weightGate, const aclTensor *weightG,
                                           const aclTensor *weightQkvScale, const aclTensor *qkvOut,
                                           const aclTensor *betaOut, const aclTensor *gateOut, const aclTensor *gOut)
{
    // 1. 空值校验
    CHECK_RET(
        CheckNotNull(x, weightQkv, weightBeta, weightGate, weightG, weightQkvScale, qkvOut, betaOut, gateOut, gOut),
        ACLNN_ERR_PARAM_NULLPTR);

    // 2. 输入类型校验
    CHECK_RET(CheckDtype(x, weightQkv, weightBeta, weightGate, weightG, weightQkvScale, qkvOut, betaOut, gateOut, gOut),
              ACLNN_ERR_PARAM_INVALID);

    // 3. format校验，只支持ND
    CHECK_RET(
        CheckFormat(x, weightQkv, weightBeta, weightGate, weightG, weightQkvScale, qkvOut, betaOut, gateOut, gOut),
        ACLNN_ERR_PARAM_INVALID);

    // 4. shape校验：左矩阵K与各权重K对应，输出[M, N]与输入对应
    CHECK_RET(CheckShapeValid(x, weightQkv, weightBeta, weightGate, weightG, qkvOut, betaOut, gateOut, gOut),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnKdaInputProjGetWorkspaceSize(const aclTensor *x, const aclTensor *weightQkv,
                                                         const aclTensor *weightBeta, const aclTensor *weightGate,
                                                         const aclTensor *weightG, const aclTensor *weightQkvScale,
                                                         const aclTensor *qkvOut, const aclTensor *betaOut,
                                                         const aclTensor *gateOut, const aclTensor *gOut,
                                                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto ret =
        CheckInputParams(x, weightQkv, weightBeta, weightGate, weightG, weightQkvScale, qkvOut, betaOut, gateOut, gOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool transQkv = false;
    bool transBeta = false;
    bool transGate = false;
    bool transG = false;
    const aclTensor *innerQkv = PrepareWeightForInner(weightQkv, transQkv);
    const aclTensor *innerBeta = PrepareWeightForInner(weightBeta, transBeta);
    const aclTensor *innerGate = PrepareWeightForInner(weightGate, transGate);
    const aclTensor *innerG = PrepareWeightForInner(weightG, transG);

    OP_LOGI("KdaInputProj infer trans from stride: qkv=%d beta=%d gate=%d g=%d.", static_cast<int32_t>(transQkv),
            static_cast<int32_t>(transBeta), static_cast<int32_t>(transGate), static_cast<int32_t>(transG));

    return aclnnInnerKdaInputProjGetWorkspaceSize(x, innerQkv, innerBeta, innerGate, innerG, weightQkvScale, transQkv,
                                                  transBeta, transGate, transG, qkvOut, betaOut, gateOut, gOut,
                                                  workspaceSize, executor);
}

extern "C" aclnnStatus aclnnKdaInputProj(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    return aclnnInnerKdaInputProj(workspace, workspaceSize, executor, stream);
}
