/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mhc_pre_sinkhorn_backward.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/aclnn_platform.h"

using namespace op;

constexpr size_t DIM_NUM_1D = 1;
constexpr size_t DIM_NUM_2D = 2;
constexpr size_t DIM_NUM_3D = 3;
constexpr size_t DIM_NUM_4D = 4;
constexpr size_t DIM_NUM_5D = 5;

#ifdef __cplusplus
extern "C" {
#endif

struct AclnnMhcPreSinkhornBackwardParams {
    const aclTensor *gradHin = nullptr;
    const aclTensor *gradHPost = nullptr;
    const aclTensor *gradHRes = nullptr;
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *hPre = nullptr;
    const aclTensor *hcBeforeNorm = nullptr;
    const aclTensor *invRms = nullptr;
    const aclTensor *sumOut = nullptr;
    const aclTensor *normOut = nullptr;
    double hcEps;

    const aclTensor *gradX = nullptr;
    const aclTensor *gradPhi = nullptr;
    const aclTensor *gradAlpha = nullptr;
    const aclTensor *gradBias = nullptr;

    const aclTensor *gradHinContiguous = nullptr;
    const aclTensor *gradHPostContiguous = nullptr;
    const aclTensor *gradHResContiguous = nullptr;
    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *biasContiguous = nullptr;
    const aclTensor *hPreContiguous = nullptr;
    const aclTensor *hcBeforeNormContiguous = nullptr;
    const aclTensor *invRmsContiguous = nullptr;
    const aclTensor *sumOutContiguous = nullptr;
    const aclTensor *normOutContiguous = nullptr;
};

class AclnnMhcPreSinkhornBackward {
public:
    static AclnnMhcPreSinkhornBackward Create()
    {
        AclnnMhcPreSinkhornBackward obj;
        return obj;
    }

    AclnnMhcPreSinkhornBackward &SetGradInput(const aclTensor *gradHin, const aclTensor *gradHPost,
                                              const aclTensor *gradHRes)
    {
        obj_.gradHin = gradHin;
        obj_.gradHPost = gradHPost;
        obj_.gradHRes = gradHRes;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                          const aclTensor *bias)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetForwardInput(const aclTensor *hPre, const aclTensor *hcBeforeNorm,
                                                 const aclTensor *invRms, const aclTensor *sumOut,
                                                 const aclTensor *normOut)
    {
        obj_.hPre = hPre;
        obj_.hcBeforeNorm = hcBeforeNorm;
        obj_.invRms = invRms;
        obj_.sumOut = sumOut;
        obj_.normOut = normOut;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetAttr(double hcEps)
    {
        obj_.hcEps = hcEps;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetOutput(const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
                                           const aclTensor *gradBias)
    {
        obj_.gradX = gradX;
        obj_.gradPhi = gradPhi;
        obj_.gradAlpha = gradAlpha;
        obj_.gradBias = gradBias;
        return *this;
    }

    AclnnMhcPreSinkhornBackwardParams Build() const
    {
        return obj_;
    }

private:
    AclnnMhcPreSinkhornBackwardParams obj_;
};

static bool CheckInputNotNullImpl(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensor is nullptr", name);
        return false;
    }
    return true;
}

static bool CheckTensorFormat(const aclTensor *tensor, const char *name)
{
    if (tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor format must be ND, but got %s", name,
                op::ToString(tensor->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormat(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return CheckTensorFormat(params.gradHin, "gradHin") && CheckTensorFormat(params.gradHPost, "gradHPost") &&
           CheckTensorFormat(params.gradHRes, "gradHRes") && CheckTensorFormat(params.x, "x") &&
           CheckTensorFormat(params.phi, "phi") && CheckTensorFormat(params.alpha, "alpha") &&
           CheckTensorFormat(params.bias, "bias") && CheckTensorFormat(params.hPre, "hPre") &&
           CheckTensorFormat(params.hcBeforeNorm, "hcBeforeNorm") && CheckTensorFormat(params.invRms, "invRms") &&
           CheckTensorFormat(params.sumOut, "sumOut") && CheckTensorFormat(params.normOut, "normOut") &&
           CheckTensorFormat(params.gradX, "gradX") && CheckTensorFormat(params.gradPhi, "gradPhi") &&
           CheckTensorFormat(params.gradAlpha, "gradAlpha") && CheckTensorFormat(params.gradBias, "gradBias");
}

static bool CheckNotNull(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return CheckInputNotNullImpl(params.gradHin, "gradHin") && CheckInputNotNullImpl(params.gradHPost, "gradHPost") &&
           CheckInputNotNullImpl(params.gradHRes, "gradHRes") && CheckInputNotNullImpl(params.x, "x") &&
           CheckInputNotNullImpl(params.phi, "phi") && CheckInputNotNullImpl(params.alpha, "alpha") &&
           CheckInputNotNullImpl(params.bias, "bias") && CheckInputNotNullImpl(params.hPre, "hPre") &&
           CheckInputNotNullImpl(params.hcBeforeNorm, "hcBeforeNorm") &&
           CheckInputNotNullImpl(params.invRms, "invRms") && CheckInputNotNullImpl(params.sumOut, "sumOut") &&
           CheckInputNotNullImpl(params.normOut, "normOut") && CheckInputNotNullImpl(params.gradX, "gradX") &&
           CheckInputNotNullImpl(params.gradPhi, "gradPhi") && CheckInputNotNullImpl(params.gradAlpha, "gradAlpha") &&
           CheckInputNotNullImpl(params.gradBias, "gradBias");
}

static bool CheckDimNum(const aclTensor *tensor, size_t expected, const char *name, bool isRange = false,
                        size_t expected2 = 0)
{
    auto dimNum = tensor->GetViewShape().GetDimNum();
    bool valid = isRange ? (dimNum == expected || dimNum == expected2) : (dimNum == expected);
    if (!valid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dim num must be %s, but got %zu", name,
                isRange ? (std::to_string(expected) + " or " + std::to_string(expected2)).c_str() :
                          std::to_string(expected).c_str(),
                dimNum);
        return false;
    }
    return true;
}

static bool CheckInputDims(const AclnnMhcPreSinkhornBackwardParams &params)
{
    bool is3D = params.x->GetViewShape().GetDimNum() == DIM_NUM_3D;
    if (is3D) {
        if (!Ops::Transformer::AclnnUtil::IsRegbase()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "3D (TND) input format is only supported on Ascend 950");
            return false;
        }
        return CheckDimNum(params.gradHin, DIM_NUM_2D, "gradHin") &&
               CheckDimNum(params.gradHPost, DIM_NUM_2D, "gradHPost") &&
               CheckDimNum(params.gradHRes, DIM_NUM_3D, "gradHRes", true, DIM_NUM_2D) &&
               CheckDimNum(params.x, DIM_NUM_3D, "x") && CheckDimNum(params.phi, DIM_NUM_2D, "phi") &&
               CheckDimNum(params.alpha, DIM_NUM_1D, "alpha") && CheckDimNum(params.bias, DIM_NUM_1D, "bias") &&
               CheckDimNum(params.hPre, DIM_NUM_2D, "hPre") &&
               CheckDimNum(params.hcBeforeNorm, DIM_NUM_2D, "hcBeforeNorm") &&
               CheckDimNum(params.invRms, DIM_NUM_2D, "invRms") && CheckDimNum(params.sumOut, DIM_NUM_3D, "sumOut") &&
               CheckDimNum(params.normOut, DIM_NUM_4D, "normOut");
    }
    return CheckDimNum(params.gradHin, DIM_NUM_3D, "gradHin") &&
           CheckDimNum(params.gradHPost, DIM_NUM_3D, "gradHPost") &&
           CheckDimNum(params.gradHRes, DIM_NUM_4D, "gradHRes", true, DIM_NUM_3D) &&
           CheckDimNum(params.x, DIM_NUM_4D, "x") && CheckDimNum(params.phi, DIM_NUM_2D, "phi") &&
           CheckDimNum(params.alpha, DIM_NUM_1D, "alpha") && CheckDimNum(params.bias, DIM_NUM_1D, "bias") &&
           CheckDimNum(params.hPre, DIM_NUM_3D, "hPre") &&
           CheckDimNum(params.hcBeforeNorm, DIM_NUM_3D, "hcBeforeNorm") &&
           CheckDimNum(params.invRms, DIM_NUM_3D, "invRms") && CheckDimNum(params.sumOut, DIM_NUM_4D, "sumOut") &&
           CheckDimNum(params.normOut, DIM_NUM_5D, "normOut");
}

static bool CheckOutputDims(const AclnnMhcPreSinkhornBackwardParams &params)
{
    bool is3D = params.x->GetViewShape().GetDimNum() == DIM_NUM_3D;
    if (is3D && !Ops::Transformer::AclnnUtil::IsRegbase()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "3D (TND) input format is only supported on Ascend 950");
        return false;
    }
    size_t gradXExpected = is3D ? DIM_NUM_3D : DIM_NUM_4D;
    return CheckDimNum(params.gradX, gradXExpected, "gradX") && CheckDimNum(params.gradPhi, DIM_NUM_2D, "gradPhi") &&
           CheckDimNum(params.gradAlpha, DIM_NUM_1D, "gradAlpha") &&
           CheckDimNum(params.gradBias, DIM_NUM_1D, "gradBias");
}

static bool CheckInputOutDims(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return CheckInputDims(params) && CheckOutputDims(params);
}

static bool CheckShapeDim(const gert::Shape &shape, uint64_t index, uint64_t expected, const char *msg)
{
    if (shape.GetDim(index) != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, and dim %lu must be %lu, actual is %ld", msg, index, expected,
                shape.GetDim(index));
        return false;
    }
    return true;
}

static bool CheckShape3D(const AclnnMhcPreSinkhornBackwardParams &params)
{
    auto &xShape = params.x->GetViewShape();
    auto &gradHInShape = params.gradHin->GetViewShape();
    uint64_t batch = gradHInShape.GetDim(0);
    uint64_t numsResidual = xShape.GetDim(1);
    uint64_t dimen = xShape.GetDim(2);

    if (!CheckShapeDim(xShape, 0, batch, "x tensor shape must be [T, N, C]") ||
        !CheckShapeDim(xShape, 1, numsResidual, "x tensor shape must be [T, N, C]") ||
        !CheckShapeDim(xShape, 2, dimen, "x tensor shape must be [T, N, C]")) {
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    auto &phiShape = params.phi->GetViewShape();
    uint64_t fusionSize = phiShape.GetDim(0);
    if (!CheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be N*C")) {
        return false;
    }

    auto &gradHPostShape = params.gradHPost->GetViewShape();
    if (!CheckShapeDim(gradHPostShape, 0, batch, "gradHPost shape must be [T, N]") ||
        !CheckShapeDim(gradHPostShape, 1, numsResidual, "gradHPost shape must be [T, N]")) {
        return false;
    }

    auto &gradHResShape = params.gradHRes->GetViewShape();
    if (!CheckShapeDim(gradHResShape, 0, batch, "gradHRes shape must be [T, N, N] or [T, N*N]")) {
        return false;
    }
    if (gradHResShape.GetDimNum() == 3) {
        if (!CheckShapeDim(gradHResShape, 1, numsResidual, "gradHRes shape must be [T, N, N]") ||
            !CheckShapeDim(gradHResShape, 2, numsResidual, "gradHRes shape must be [T, N, N]")) {
            return false;
        }
    } else {
        if (!CheckShapeDim(gradHResShape, 1, numsResidual * numsResidual, "gradHRes shape must be [T, N*N]")) {
            return false;
        }
    }

    auto &alphaShape = params.alpha->GetViewShape();
    if (!CheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    auto &biasShape = params.bias->GetViewShape();
    if (!CheckShapeDim(biasShape, 0, fusionSize, "bias tensor shape must be (2N+N^2)")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, batch, "hPre shape must be [T, N]") ||
        !CheckShapeDim(hPreShape, 1, numsResidual, "hPre shape must be [T, N]")) {
        return false;
    }

    auto &hcBeforeNormShape = params.hcBeforeNorm->GetViewShape();
    if (!CheckShapeDim(hcBeforeNormShape, 0, batch, "hcBeforeNorm shape must be [T, 2N+N^2]") ||
        !CheckShapeDim(hcBeforeNormShape, 1, fusionSize, "hcBeforeNorm shape must be [T, 2N+N^2]")) {
        return false;
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [T, 1]") ||
        !CheckShapeDim(invRmsShape, 1, 1, "invRms shape must be [T, 1]")) {
        return false;
    }

    auto &sumOutShape = params.sumOut->GetViewShape();
    if (!CheckShapeDim(sumOutShape, 1, batch, "sumOut shape must be [2*sk_iter_count, T, N]") ||
        !CheckShapeDim(sumOutShape, 2, numsResidual, "sumOut shape must be [2*sk_iter_count, T, N]")) {
        return false;
    }

    auto &normOutShape = params.normOut->GetViewShape();
    if (!CheckShapeDim(normOutShape, 1, batch, "normOut shape must be [2*sk_iter_count, T, N, N]") ||
        !CheckShapeDim(normOutShape, 2, numsResidual, "normOut shape must be [2*sk_iter_count, T, N, N]") ||
        !CheckShapeDim(normOutShape, 3, numsResidual, "normOut shape must be [2*sk_iter_count, T, N, N]")) {
        return false;
    }

    auto &gradXShape = params.gradX->GetViewShape();
    if (!CheckShapeDim(gradXShape, 0, batch, "gradX shape must be [T, N, C]") ||
        !CheckShapeDim(gradXShape, 1, numsResidual, "gradX shape must be [T, N, C]") ||
        !CheckShapeDim(gradXShape, 2, dimen, "gradX shape must be [T, N, C]")) {
        return false;
    }

    auto &gradPhiShape = params.gradPhi->GetViewShape();
    if (!CheckShapeDim(gradPhiShape, 0, fusionSize, "gradPhi shape must be [2N+N^2, N*C]") ||
        !CheckShapeDim(gradPhiShape, 1, nD, "gradPhi shape must be [2N+N^2, N*C]")) {
        return false;
    }

    auto &gradAlphaShape = params.gradAlpha->GetViewShape();
    if (!CheckShapeDim(gradAlphaShape, 0, 3, "gradAlpha shape must be (3)")) {
        return false;
    }

    auto &gradBiasShape = params.gradBias->GetViewShape();
    if (!CheckShapeDim(gradBiasShape, 0, fusionSize, "gradBias shape must be (2N+N^2)")) {
        return false;
    }

    return true;
}

static bool CheckShape(const AclnnMhcPreSinkhornBackwardParams &params)
{
    auto &xShape = params.x->GetViewShape();
    if (xShape.GetDimNum() == 3) {
        if (!Ops::Transformer::AclnnUtil::IsRegbase()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "3D (TND) input format is only supported on Ascend 950");
            return false;
        }
        return CheckShape3D(params);
    }

    auto &gradHInShape = params.gradHin->GetViewShape();
    uint64_t batch = gradHInShape.GetDim(0);
    uint64_t sequence = gradHInShape.GetDim(1);
    uint64_t dimen = gradHInShape.GetDim(2);

    if (!CheckShapeDim(xShape, 0, batch, "x tensor shape must be [B, S, N, C]") ||
        !CheckShapeDim(xShape, 1, sequence, "x tensor shape must be [B, S, N, C]")) {
        return false;
    }
    uint64_t numsResidual = xShape.GetDim(2);
    if (!CheckShapeDim(xShape, 3, dimen, "x tensor shape must be [B, S, N, C]")) {
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    auto &phiShape = params.phi->GetViewShape();
    uint64_t fusionSize = phiShape.GetDim(0);
    if (!CheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be N*C")) {
        return false;
    }

    auto &gradHPostShape = params.gradHPost->GetViewShape();
    if (!CheckShapeDim(gradHPostShape, 0, batch, "gradHPost shape must be [B, S, N]") ||
        !CheckShapeDim(gradHPostShape, 1, sequence, "gradHPost shape must be [B, S, N]") ||
        !CheckShapeDim(gradHPostShape, 2, numsResidual, "gradHPost shape must be [B, S, N]")) {
        return false;
    }

    auto &gradHResShape = params.gradHRes->GetViewShape();
    if (!CheckShapeDim(gradHResShape, 0, batch, "gradHRes shape must be [B, S, N, N] or [B, S, N*N]") ||
        !CheckShapeDim(gradHResShape, 1, sequence, "gradHRes shape must be [B, S, N, N] or [B, S, N*N]")) {
        return false;
    }
    if (gradHResShape.GetDimNum() == 4) {
        if (!CheckShapeDim(gradHResShape, 2, numsResidual, "gradHRes shape must be [B, S, N, N]") ||
            !CheckShapeDim(gradHResShape, 3, numsResidual, "gradHRes shape must be [B, S, N, N]")) {
            return false;
        }
    } else {
        if (!CheckShapeDim(gradHResShape, 2, numsResidual * numsResidual, "gradHRes shape must be [B, S, N*N]")) {
            return false;
        }
    }

    auto &alphaShape = params.alpha->GetViewShape();
    if (!CheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    auto &biasShape = params.bias->GetViewShape();
    if (!CheckShapeDim(biasShape, 0, fusionSize, "bias tensor shape must be (2N+N^2)")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, batch, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 1, sequence, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 2, numsResidual, "hPre shape must be [B, S, N]")) {
        return false;
    }

    auto &hcBeforeNormShape = params.hcBeforeNorm->GetViewShape();
    if (!CheckShapeDim(hcBeforeNormShape, 0, batch, "hcBeforeNorm shape must be [B, S, 2N+N^2]") ||
        !CheckShapeDim(hcBeforeNormShape, 1, sequence, "hcBeforeNorm shape must be [B, S, 2N+N^2]") ||
        !CheckShapeDim(hcBeforeNormShape, 2, fusionSize, "hcBeforeNorm shape must be [B, S, 2N+N^2]")) {
        return false;
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [B, S, 1]") ||
        !CheckShapeDim(invRmsShape, 1, sequence, "invRms shape must be [B, S, 1]") ||
        !CheckShapeDim(invRmsShape, 2, 1, "invRms shape must be [B, S, 1]")) {
        return false;
    }

    auto &sumOutShape = params.sumOut->GetViewShape();
    if (!CheckShapeDim(sumOutShape, 1, batch, "sumOut shape must be [2*sk_iter_count, B, S, N]") ||
        !CheckShapeDim(sumOutShape, 2, sequence, "sumOut shape must be [2*sk_iter_count, B, S, N]") ||
        !CheckShapeDim(sumOutShape, 3, numsResidual, "sumOut shape must be [2*sk_iter_count, B, S, N]")) {
        return false;
    }

    auto &normOutShape = params.normOut->GetViewShape();
    if (!CheckShapeDim(normOutShape, 1, batch, "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !CheckShapeDim(normOutShape, 2, sequence, "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !CheckShapeDim(normOutShape, 3, numsResidual, "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !CheckShapeDim(normOutShape, 4, numsResidual, "normOut shape must be [2*sk_iter_count, B, S, N, N]")) {
        return false;
    }

    auto &gradXShape = params.gradX->GetViewShape();
    if (!CheckShapeDim(gradXShape, 0, batch, "gradX shape must be [B, S, N, C]") ||
        !CheckShapeDim(gradXShape, 1, sequence, "gradX shape must be [B, S, N, C]") ||
        !CheckShapeDim(gradXShape, 2, numsResidual, "gradX shape must be [B, S, N, C]") ||
        !CheckShapeDim(gradXShape, 3, dimen, "gradX shape must be [B, S, N, C]")) {
        return false;
    }

    auto &gradPhiShape = params.gradPhi->GetViewShape();
    if (!CheckShapeDim(gradPhiShape, 0, fusionSize, "gradPhi shape must be [2N+N^2, N*C]") ||
        !CheckShapeDim(gradPhiShape, 1, nD, "gradPhi shape must be [2N+N^2, N*C]")) {
        return false;
    }

    auto &gradAlphaShape = params.gradAlpha->GetViewShape();
    if (!CheckShapeDim(gradAlphaShape, 0, 3, "gradAlpha shape must be (3)")) {
        return false;
    }

    auto &gradBiasShape = params.gradBias->GetViewShape();
    if (!CheckShapeDim(gradBiasShape, 0, fusionSize, "gradBias shape must be (2N+N^2)")) {
        return false;
    }

    return true;
}

static bool CheckDtype(const aclTensor *tensor, DataType expected, const char *name, bool isValidX = false)
{
    auto dtype = tensor->GetDataType();
    bool valid = isValidX ? (dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16) : (dtype == expected);
    if (!valid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dtype must be %s. actual is [%s].", name,
                isValidX ? "BF16 or FP16" : op::ToString(expected).GetString(), op::ToString(dtype).GetString());
        return false;
    }
    return true;
}

static bool CheckInputDtype(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return CheckDtype(params.gradHin, DataType::DT_FLOAT16, "gradHin", true) &&
           CheckDtype(params.gradHPost, DataType::DT_FLOAT, "gradHPost") &&
           CheckDtype(params.gradHRes, DataType::DT_FLOAT, "gradHRes") &&
           CheckDtype(params.x, DataType::DT_FLOAT16, "x", true) && CheckDtype(params.phi, DataType::DT_FLOAT, "phi") &&
           CheckDtype(params.alpha, DataType::DT_FLOAT, "alpha") &&
           CheckDtype(params.bias, DataType::DT_FLOAT, "bias") && CheckDtype(params.hPre, DataType::DT_FLOAT, "hPre") &&
           CheckDtype(params.hcBeforeNorm, DataType::DT_FLOAT, "hcBeforeNorm") &&
           CheckDtype(params.invRms, DataType::DT_FLOAT, "invRms") &&
           CheckDtype(params.sumOut, DataType::DT_FLOAT, "sumOut") &&
           CheckDtype(params.normOut, DataType::DT_FLOAT, "normOut");
}

static bool CheckOutputDtype(const AclnnMhcPreSinkhornBackwardParams &params)
{
    auto gradHInDtype = params.gradHin->GetDataType();
    if (params.gradX->GetDataType() != gradHInDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradX tensor dtype must match gradHin");
        return false;
    }
    return CheckDtype(params.gradPhi, DataType::DT_FLOAT, "gradPhi") &&
           CheckDtype(params.gradAlpha, DataType::DT_FLOAT, "gradAlpha") &&
           CheckDtype(params.gradBias, DataType::DT_FLOAT, "gradBias");
}

static bool CheckDtypeValid(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return CheckInputDtype(params) && CheckOutputDtype(params);
}

static aclnnStatus CheckParams(const AclnnMhcPreSinkhornBackwardParams &params)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 校验输入、输出参数维度
    CHECK_RET(CheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验输入、输出shape参数
    CHECK_RET(CheckShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    // 5. 校验format：仅支持ND格式，不支持私有格式
    CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);

    // 6. 校验N、C取值范围
    auto &xShape = params.x->GetViewShape();
    bool is3D = xShape.GetDimNum() == 3;
    int64_t nVal = is3D ? xShape.GetDim(1) : xShape.GetDim(2);
    int64_t cVal = is3D ? xShape.GetDim(2) : xShape.GetDim(3);
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        if (nVal <= 0 || nVal > 8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N must be > 0 and <= 8, but got %ld", nVal);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (nVal != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N must be 4, but got %ld", nVal);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (cVal <= 0 || cVal >= 100000 || cVal % 128 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "C must be > 0, < 100000 and divisible by 128, but got %ld", cVal);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static const aclTensor *ConvertToContiguous(const aclTensor *tensor, aclOpExecutor *executor)
{
    auto result = l0op::Contiguous(tensor, executor);
    if (result == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Contiguous result is nullptr");
    }
    return result;
}

aclnnStatus MhcSinkhornGradCovertDataContiguous(AclnnMhcPreSinkhornBackwardParams &params, aclOpExecutor *executor)
{
    params.gradHinContiguous = ConvertToContiguous(params.gradHin, executor);
    params.gradHPostContiguous = ConvertToContiguous(params.gradHPost, executor);
    params.gradHResContiguous = ConvertToContiguous(params.gradHRes, executor);
    params.xContiguous = ConvertToContiguous(params.x, executor);
    params.phiContiguous = ConvertToContiguous(params.phi, executor);
    params.alphaContiguous = ConvertToContiguous(params.alpha, executor);
    params.biasContiguous = ConvertToContiguous(params.bias, executor);
    params.hPreContiguous = ConvertToContiguous(params.hPre, executor);
    params.hcBeforeNormContiguous = ConvertToContiguous(params.hcBeforeNorm, executor);
    params.invRmsContiguous = ConvertToContiguous(params.invRms, executor);
    params.sumOutContiguous = ConvertToContiguous(params.sumOut, executor);
    params.normOutContiguous = ConvertToContiguous(params.normOut, executor);

    if (params.gradHinContiguous == nullptr || params.gradHPostContiguous == nullptr ||
        params.gradHResContiguous == nullptr || params.xContiguous == nullptr || params.phiContiguous == nullptr ||
        params.alphaContiguous == nullptr || params.biasContiguous == nullptr || params.hPreContiguous == nullptr ||
        params.hcBeforeNormContiguous == nullptr || params.invRmsContiguous == nullptr ||
        params.sumOutContiguous == nullptr || params.normOutContiguous == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static bool CopyOutput(const aclTensor *out, const aclTensor *dst, aclOpExecutor *executor)
{
    auto ret = l0op::ViewCopy(out, dst, executor);
    return ret != nullptr;
}

static aclnnStatus mhcPreSinkhornBackwardCommonProcess(AclnnMhcPreSinkhornBackwardParams &params,
                                                       aclOpExecutor *executor)
{
    auto ret = MhcSinkhornGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto outParams = l0op::MhcPreSinkhornBackward(
        params.gradHinContiguous, params.gradHPostContiguous, params.gradHResContiguous, params.xContiguous,
        params.phiContiguous, params.alphaContiguous, params.biasContiguous, params.hPreContiguous,
        params.hcBeforeNormContiguous, params.invRmsContiguous, params.sumOutContiguous, params.normOutContiguous,
        static_cast<float>(params.hcEps), executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CopyOutput(std::get<0>(outParams), params.gradX, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<1>(outParams), params.gradPhi, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<2>(outParams), params.gradAlpha, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<3>(outParams), params.gradBias, executor), ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
    const aclTensor *gradHin, const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *x,
    const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *hPre,
    const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut, const aclTensor *normOut,
    double hcEps, const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
    const aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnMhcPreSinkhornBackward,
        DFX_IN(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut),
        DFX_OUT(gradX, gradPhi, gradAlpha, gradBias));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    AclnnMhcPreSinkhornBackwardParams params = AclnnMhcPreSinkhornBackward::Create()
                                                   .SetGradInput(gradHin, gradHPost, gradHRes)
                                                   .SetInput(x, phi, alpha, bias)
                                                   .SetForwardInput(hPre, hcBeforeNorm, invRms, sumOut, normOut)
                                                   .SetAttr(hcEps)
                                                   .SetOutput(gradX, gradPhi, gradAlpha, gradBias)
                                                   .Build();

    auto ret = CheckNotNull(params) ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_NULLPTR;
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradHin->IsEmpty() || gradHPost->IsEmpty() || gradHRes->IsEmpty() || x->IsEmpty() || hPre->IsEmpty() ||
        hcBeforeNorm->IsEmpty() || invRms->IsEmpty() || sumOut->IsEmpty() || normOut->IsEmpty()) {
        OP_LOGW("[aclnnMhcPreSinkhornBackward] Input tensor is empty, skip computation and return success.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = mhcPreSinkhornBackwardCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhornBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
