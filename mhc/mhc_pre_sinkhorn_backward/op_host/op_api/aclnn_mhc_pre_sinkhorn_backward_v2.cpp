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
#include "aclnn_mhc_pre_sinkhorn_backward.h"
#include "mhc_pre_sinkhorn_backward.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/aclnn_platform.h"
#include "level0/fill.h"

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
    const aclTensor* gradHin = nullptr;
    const aclTensor* gradHPost = nullptr;
    const aclTensor* gradHRes = nullptr;
    const aclTensor* x = nullptr;
    const aclTensor* phi = nullptr;
    const aclTensor* alpha = nullptr;
    const aclTensor* bias = nullptr;
    const aclTensor* hPre = nullptr;
    const aclTensor* hcBeforeNorm = nullptr;
    const aclTensor* invRms = nullptr;
    const aclTensor* sumOut = nullptr;
    const aclTensor* normOut = nullptr;
    const aclTensor* preMix = nullptr;
    const aclTensor* gradPre = nullptr;
    double hcEps;
    bool enableHf32 = true;

    const aclTensor* gradX = nullptr;
    const aclTensor* gradPhi = nullptr;
    const aclTensor* gradAlpha = nullptr;
    const aclTensor* gradBias = nullptr;
    const aclTensor* gradPreMix = nullptr;

    const aclTensor* gradHinContiguous = nullptr;
    const aclTensor* gradHPostContiguous = nullptr;
    const aclTensor* gradHResContiguous = nullptr;
    const aclTensor* xContiguous = nullptr;
    const aclTensor* phiContiguous = nullptr;
    const aclTensor* alphaContiguous = nullptr;
    const aclTensor* biasContiguous = nullptr;
    const aclTensor* hPreContiguous = nullptr;
    const aclTensor* hcBeforeNormContiguous = nullptr;
    const aclTensor* invRmsContiguous = nullptr;
    const aclTensor* sumOutContiguous = nullptr;
    const aclTensor* normOutContiguous = nullptr;
    const aclTensor* preMixContiguous = nullptr;
    const aclTensor* gradPreContiguous = nullptr;
};

class AclnnMhcPreSinkhornBackward {
public:
    static AclnnMhcPreSinkhornBackward Create()
    {
        AclnnMhcPreSinkhornBackward obj;
        return obj;
    }

    AclnnMhcPreSinkhornBackward& SetGradInput(const aclTensor* gradHin, const aclTensor* gradHPost,
                                              const aclTensor* gradHRes)
    {
        obj_.gradHin = gradHin;
        obj_.gradHPost = gradHPost;
        obj_.gradHRes = gradHRes;
        return *this;
    }

    AclnnMhcPreSinkhornBackward& SetInput(const aclTensor* x, const aclTensor* phi, const aclTensor* alpha,
                                          const aclTensor* bias)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        return *this;
    }

    AclnnMhcPreSinkhornBackward& SetForwardInput(const aclTensor* hPre, const aclTensor* hcBeforeNorm,
                                                 const aclTensor* invRms, const aclTensor* sumOut,
                                                 const aclTensor* normOut, const aclTensor* preMix,
                                                 const aclTensor* gradPre)
    {
        obj_.hPre = hPre;
        obj_.hcBeforeNorm = hcBeforeNorm;
        obj_.invRms = invRms;
        obj_.sumOut = sumOut;
        obj_.normOut = normOut;
        obj_.preMix = preMix;
        obj_.gradPre = gradPre;
        return *this;
    }

    AclnnMhcPreSinkhornBackward& SetAttr(double hcEps, bool enableHf32 = true)
    {
        obj_.hcEps = hcEps;
        obj_.enableHf32 = enableHf32;
        return *this;
    }

    AclnnMhcPreSinkhornBackward& SetOutput(const aclTensor* gradX, const aclTensor* gradPhi, const aclTensor* gradAlpha,
                                           const aclTensor* gradBias, const aclTensor* gradPreMix)
    {
        obj_.gradX = gradX;
        obj_.gradPhi = gradPhi;
        obj_.gradAlpha = gradAlpha;
        obj_.gradBias = gradBias;
        obj_.gradPreMix = gradPreMix;
        return *this;
    }

    AclnnMhcPreSinkhornBackwardParams Build() const
    {
        return obj_;
    }

private:
    AclnnMhcPreSinkhornBackwardParams obj_;
};

static bool CheckInputNotNullImpl(const aclTensor* tensor, const char* name)
{
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensor is nullptr", name);
        return false;
    }
    return true;
}

static bool CheckTensorFormat(const aclTensor* tensor, const char* name)
{
    if (tensor != nullptr && op::IsPrivateFormat(tensor->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s's format should be ND, but got %s.", name,
                op::ToString(tensor->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormat(const AclnnMhcPreSinkhornBackwardParams& params)
{
    return CheckTensorFormat(params.gradHin, "gradHin") && CheckTensorFormat(params.gradHPost, "gradHPost") &&
           CheckTensorFormat(params.gradHRes, "gradHRes") && CheckTensorFormat(params.x, "x") &&
           CheckTensorFormat(params.phi, "phi") && CheckTensorFormat(params.alpha, "alpha") &&
           CheckTensorFormat(params.bias, "bias") && CheckTensorFormat(params.hPre, "hPre") &&
           CheckTensorFormat(params.hcBeforeNorm, "hcBeforeNorm") && CheckTensorFormat(params.invRms, "invRms") &&
           CheckTensorFormat(params.sumOut, "sumOut") && CheckTensorFormat(params.normOut, "normOut") &&
           CheckTensorFormat(params.preMix, "preMix") && CheckTensorFormat(params.gradPre, "gradPre") &&
           CheckTensorFormat(params.gradX, "gradX") && CheckTensorFormat(params.gradPhi, "gradPhi") &&
           CheckTensorFormat(params.gradAlpha, "gradAlpha") && CheckTensorFormat(params.gradBias, "gradBias") &&
           CheckTensorFormat(params.gradPreMix, "gradPreMix");
}

static bool CheckNotNull(const AclnnMhcPreSinkhornBackwardParams& params)
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

static bool CheckDimNum(const aclTensor* tensor, size_t expected, const char* name, bool isRange = false,
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

static bool CheckInputDims(const AclnnMhcPreSinkhornBackwardParams& params)
{
    bool is3D = params.x->GetViewShape().GetDimNum() == DIM_NUM_3D;
    if (is3D) {
        return CheckDimNum(params.gradHin, DIM_NUM_2D, "gradHin") &&
               CheckDimNum(params.gradHPost, DIM_NUM_2D, "gradHPost") &&
               CheckDimNum(params.gradHRes, DIM_NUM_3D, "gradHRes", true, DIM_NUM_2D) &&
               CheckDimNum(params.x, DIM_NUM_3D, "x") && CheckDimNum(params.phi, DIM_NUM_2D, "phi") &&
               CheckDimNum(params.alpha, DIM_NUM_1D, "alpha") && CheckDimNum(params.bias, DIM_NUM_1D, "bias") &&
               CheckDimNum(params.hPre, DIM_NUM_2D, "hPre") &&
               CheckDimNum(params.hcBeforeNorm, DIM_NUM_2D, "hcBeforeNorm") &&
               CheckDimNum(params.invRms, DIM_NUM_2D, "invRms") && CheckDimNum(params.sumOut, DIM_NUM_3D, "sumOut") &&
               CheckDimNum(params.normOut, DIM_NUM_4D, "normOut") &&
               (params.preMix == nullptr || CheckDimNum(params.preMix, DIM_NUM_2D, "preMix")) &&
               (params.gradPre == nullptr || CheckDimNum(params.gradPre, DIM_NUM_2D, "gradPre"));
    }
    return CheckDimNum(params.gradHin, DIM_NUM_3D, "gradHin") &&
           CheckDimNum(params.gradHPost, DIM_NUM_3D, "gradHPost") &&
           CheckDimNum(params.gradHRes, DIM_NUM_4D, "gradHRes", true, DIM_NUM_3D) &&
           CheckDimNum(params.x, DIM_NUM_4D, "x") && CheckDimNum(params.phi, DIM_NUM_2D, "phi") &&
           CheckDimNum(params.alpha, DIM_NUM_1D, "alpha") && CheckDimNum(params.bias, DIM_NUM_1D, "bias") &&
           CheckDimNum(params.hPre, DIM_NUM_3D, "hPre") &&
           CheckDimNum(params.hcBeforeNorm, DIM_NUM_3D, "hcBeforeNorm") &&
           CheckDimNum(params.invRms, DIM_NUM_3D, "invRms") && CheckDimNum(params.sumOut, DIM_NUM_4D, "sumOut") &&
           CheckDimNum(params.normOut, DIM_NUM_5D, "normOut") &&
           (params.preMix == nullptr || CheckDimNum(params.preMix, DIM_NUM_3D, "preMix")) &&
           (params.gradPre == nullptr || CheckDimNum(params.gradPre, DIM_NUM_3D, "gradPre"));
}

static bool CheckOutputDims(const AclnnMhcPreSinkhornBackwardParams& params)
{
    bool is3D = params.x->GetViewShape().GetDimNum() == DIM_NUM_3D;
    size_t gradXExpected = is3D ? DIM_NUM_3D : DIM_NUM_4D;
    size_t gradPreMixExpected = is3D ? DIM_NUM_2D : DIM_NUM_3D;
    return CheckDimNum(params.gradX, gradXExpected, "gradX") && CheckDimNum(params.gradPhi, DIM_NUM_2D, "gradPhi") &&
           CheckDimNum(params.gradAlpha, DIM_NUM_1D, "gradAlpha") &&
           CheckDimNum(params.gradBias, DIM_NUM_1D, "gradBias") &&
           (params.gradPreMix == nullptr || CheckDimNum(params.gradPreMix, gradPreMixExpected, "gradPreMix"));
}

static bool CheckInputOutDims(const AclnnMhcPreSinkhornBackwardParams& params)
{
    return CheckInputDims(params) && CheckOutputDims(params);
}

static bool CheckShapeDim(const gert::Shape& shape, uint64_t index, uint64_t expected, const char* msg)
{
    if (shape.GetDim(index) != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, and dim %lu must be %lu, actual is %ld", msg, index, expected,
                shape.GetDim(index));
        return false;
    }
    return true;
}

static bool CheckShape3D(const AclnnMhcPreSinkhornBackwardParams& params)
{
    auto& xShape = params.x->GetViewShape();
    auto& gradHInShape = params.gradHin->GetViewShape();
    uint64_t batch = gradHInShape.GetDim(0);
    uint64_t numsResidual = xShape.GetDim(1);
    uint64_t dimen = xShape.GetDim(2);

    if (!CheckShapeDim(xShape, 0, batch, "x tensor shape must be [T, N, C]") ||
        !CheckShapeDim(xShape, 1, numsResidual, "x tensor shape must be [T, N, C]") ||
        !CheckShapeDim(xShape, 2, dimen, "x tensor shape must be [T, N, C]")) {
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    auto& phiShape = params.phi->GetViewShape();
    uint64_t fusionSize = phiShape.GetDim(0);
    if (!CheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be N*C")) {
        return false;
    }

    auto& gradHPostShape = params.gradHPost->GetViewShape();
    if (!CheckShapeDim(gradHPostShape, 0, batch, "gradHPost shape must be [T, N]") ||
        !CheckShapeDim(gradHPostShape, 1, numsResidual, "gradHPost shape must be [T, N]")) {
        return false;
    }

    auto& gradHResShape = params.gradHRes->GetViewShape();
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

    auto& alphaShape = params.alpha->GetViewShape();
    if (!CheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    auto& biasShape = params.bias->GetViewShape();
    if (!CheckShapeDim(biasShape, 0, fusionSize, "bias tensor shape must be (2N+N^2)")) {
        return false;
    }

    auto& hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, batch, "hPre shape must be [T, N]") ||
        !CheckShapeDim(hPreShape, 1, numsResidual, "hPre shape must be [T, N]")) {
        return false;
    }

    if (params.preMix != nullptr) {
        auto& preMixShape = params.preMix->GetViewShape();
        if (!CheckShapeDim(preMixShape, 0, batch, "preMix shape must be [T, N]") ||
            !CheckShapeDim(preMixShape, 1, numsResidual, "preMix shape must be [T, N]")) {
            return false;
        }
    }

    if (params.gradPre != nullptr) {
        auto& gradPreShape = params.gradPre->GetViewShape();
        if (!CheckShapeDim(gradPreShape, 0, batch, "gradPre shape must be [T, N]") ||
            !CheckShapeDim(gradPreShape, 1, numsResidual, "gradPre shape must be [T, N]")) {
            return false;
        }
    }

    auto& hcBeforeNormShape = params.hcBeforeNorm->GetViewShape();
    if (!CheckShapeDim(hcBeforeNormShape, 0, batch, "hcBeforeNorm shape must be [T, 2N+N^2]") ||
        !CheckShapeDim(hcBeforeNormShape, 1, fusionSize, "hcBeforeNorm shape must be [T, 2N+N^2]")) {
        return false;
    }

    auto& invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [T, 1]") ||
        !CheckShapeDim(invRmsShape, 1, 1, "invRms shape must be [T, 1]")) {
        return false;
    }

    auto& sumOutShape = params.sumOut->GetViewShape();
    if (!CheckShapeDim(sumOutShape, 1, batch, "sumOut shape must be [2*sk_iter_count, T, N]") ||
        !CheckShapeDim(sumOutShape, 2, numsResidual, "sumOut shape must be [2*sk_iter_count, T, N]")) {
        return false;
    }

    auto& normOutShape = params.normOut->GetViewShape();
    if (!CheckShapeDim(normOutShape, 1, batch, "normOut shape must be [2*sk_iter_count, T, N, N]") ||
        !CheckShapeDim(normOutShape, 2, numsResidual, "normOut shape must be [2*sk_iter_count, T, N, N]") ||
        !CheckShapeDim(normOutShape, 3, numsResidual, "normOut shape must be [2*sk_iter_count, T, N, N]")) {
        return false;
    }

    auto& gradXShape = params.gradX->GetViewShape();
    if (!CheckShapeDim(gradXShape, 0, batch, "gradX shape must be [T, N, C]") ||
        !CheckShapeDim(gradXShape, 1, numsResidual, "gradX shape must be [T, N, C]") ||
        !CheckShapeDim(gradXShape, 2, dimen, "gradX shape must be [T, N, C]")) {
        return false;
    }

    auto& gradPhiShape = params.gradPhi->GetViewShape();
    if (!CheckShapeDim(gradPhiShape, 0, fusionSize, "gradPhi shape must be [2N+N^2, N*C]") ||
        !CheckShapeDim(gradPhiShape, 1, nD, "gradPhi shape must be [2N+N^2, N*C]")) {
        return false;
    }

    auto& gradAlphaShape = params.gradAlpha->GetViewShape();
    if (!CheckShapeDim(gradAlphaShape, 0, 3, "gradAlpha shape must be (3)")) {
        return false;
    }

    auto& gradBiasShape = params.gradBias->GetViewShape();
    if (!CheckShapeDim(gradBiasShape, 0, fusionSize, "gradBias shape must be (2N+N^2)")) {
        return false;
    }

    if (params.gradPreMix != nullptr) {
        auto& gradPreMixShape = params.gradPreMix->GetViewShape();
        if (!CheckShapeDim(gradPreMixShape, 0, batch, "gradPreMix shape must be [T, N]") ||
            !CheckShapeDim(gradPreMixShape, 1, numsResidual, "gradPreMix shape must be [T, N]")) {
            return false;
        }
    }

    return true;
}

static bool CheckShape(const AclnnMhcPreSinkhornBackwardParams& params)
{
    auto& xShape = params.x->GetViewShape();
    if (xShape.GetDimNum() == 3) {
        return CheckShape3D(params);
    }

    auto& gradHInShape = params.gradHin->GetViewShape();
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
    auto& phiShape = params.phi->GetViewShape();
    uint64_t fusionSize = phiShape.GetDim(0);
    if (!CheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be N*C")) {
        return false;
    }

    auto& gradHPostShape = params.gradHPost->GetViewShape();
    if (!CheckShapeDim(gradHPostShape, 0, batch, "gradHPost shape must be [B, S, N]") ||
        !CheckShapeDim(gradHPostShape, 1, sequence, "gradHPost shape must be [B, S, N]") ||
        !CheckShapeDim(gradHPostShape, 2, numsResidual, "gradHPost shape must be [B, S, N]")) {
        return false;
    }

    auto& gradHResShape = params.gradHRes->GetViewShape();
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

    auto& alphaShape = params.alpha->GetViewShape();
    if (!CheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    auto& biasShape = params.bias->GetViewShape();
    if (!CheckShapeDim(biasShape, 0, fusionSize, "bias tensor shape must be (2N+N^2)")) {
        return false;
    }

    auto& hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, batch, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 1, sequence, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 2, numsResidual, "hPre shape must be [B, S, N]")) {
        return false;
    }

    if (params.preMix != nullptr) {
        auto& preMixShape = params.preMix->GetViewShape();
        if (!CheckShapeDim(preMixShape, 0, batch, "preMix shape must be [B, S, N]") ||
            !CheckShapeDim(preMixShape, 1, sequence, "preMix shape must be [B, S, N]") ||
            !CheckShapeDim(preMixShape, 2, numsResidual, "preMix shape must be [B, S, N]")) {
            return false;
        }
    }

    if (params.gradPre != nullptr) {
        auto& gradPreShape = params.gradPre->GetViewShape();
        if (!CheckShapeDim(gradPreShape, 0, batch, "gradPre shape must be [B, S, N]") ||
            !CheckShapeDim(gradPreShape, 1, sequence, "gradPre shape must be [B, S, N]") ||
            !CheckShapeDim(gradPreShape, 2, numsResidual, "gradPre shape must be [B, S, N]")) {
            return false;
        }
    }

    auto& hcBeforeNormShape = params.hcBeforeNorm->GetViewShape();
    if (!CheckShapeDim(hcBeforeNormShape, 0, batch, "hcBeforeNorm shape must be [B, S, 2N+N^2]") ||
        !CheckShapeDim(hcBeforeNormShape, 1, sequence, "hcBeforeNorm shape must be [B, S, 2N+N^2]") ||
        !CheckShapeDim(hcBeforeNormShape, 2, fusionSize, "hcBeforeNorm shape must be [B, S, 2N+N^2]")) {
        return false;
    }

    auto& invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [B, S, 1]") ||
        !CheckShapeDim(invRmsShape, 1, sequence, "invRms shape must be [B, S, 1]") ||
        !CheckShapeDim(invRmsShape, 2, 1, "invRms shape must be [B, S, 1]")) {
        return false;
    }

    auto& sumOutShape = params.sumOut->GetViewShape();
    if (!CheckShapeDim(sumOutShape, 1, batch, "sumOut shape must be [2*sk_iter_count, B, S, N]") ||
        !CheckShapeDim(sumOutShape, 2, sequence, "sumOut shape must be [2*sk_iter_count, B, S, N]") ||
        !CheckShapeDim(sumOutShape, 3, numsResidual, "sumOut shape must be [2*sk_iter_count, B, S, N]")) {
        return false;
    }

    auto& normOutShape = params.normOut->GetViewShape();
    if (!CheckShapeDim(normOutShape, 1, batch, "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !CheckShapeDim(normOutShape, 2, sequence, "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !CheckShapeDim(normOutShape, 3, numsResidual, "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !CheckShapeDim(normOutShape, 4, numsResidual, "normOut shape must be [2*sk_iter_count, B, S, N, N]")) {
        return false;
    }

    auto& gradXShape = params.gradX->GetViewShape();
    if (!CheckShapeDim(gradXShape, 0, batch, "gradX shape must be [B, S, N, C]") ||
        !CheckShapeDim(gradXShape, 1, sequence, "gradX shape must be [B, S, N, C]") ||
        !CheckShapeDim(gradXShape, 2, numsResidual, "gradX shape must be [B, S, N, C]") ||
        !CheckShapeDim(gradXShape, 3, dimen, "gradX shape must be [B, S, N, C]")) {
        return false;
    }

    auto& gradPhiShape = params.gradPhi->GetViewShape();
    if (!CheckShapeDim(gradPhiShape, 0, fusionSize, "gradPhi shape must be [2N+N^2, N*C]") ||
        !CheckShapeDim(gradPhiShape, 1, nD, "gradPhi shape must be [2N+N^2, N*C]")) {
        return false;
    }

    auto& gradAlphaShape = params.gradAlpha->GetViewShape();
    if (!CheckShapeDim(gradAlphaShape, 0, 3, "gradAlpha shape must be (3)")) {
        return false;
    }

    auto& gradBiasShape = params.gradBias->GetViewShape();
    if (!CheckShapeDim(gradBiasShape, 0, fusionSize, "gradBias shape must be (2N+N^2)")) {
        return false;
    }

    if (params.gradPreMix != nullptr) {
        auto& gradPreMixShape = params.gradPreMix->GetViewShape();
        if (!CheckShapeDim(gradPreMixShape, 0, batch, "gradPreMix shape must be [B, S, N]") ||
            !CheckShapeDim(gradPreMixShape, 1, sequence, "gradPreMix shape must be [B, S, N]") ||
            !CheckShapeDim(gradPreMixShape, 2, numsResidual, "gradPreMix shape must be [B, S, N]")) {
            return false;
        }
    }

    return true;
}

static bool CheckDtype(const aclTensor* tensor, DataType expected, const char* name, bool isValidX = false)
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

static bool CheckInputDtype(const AclnnMhcPreSinkhornBackwardParams& params)
{
    // x 与 gradHin 必须同数据类型, 否则 kernel 无对应 dtype 组合的二进制
    if (params.x->GetDataType() != params.gradHin->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dtype must match gradHin. x is [%s], gradHin is [%s].",
                op::ToString(params.x->GetDataType()).GetString(),
                op::ToString(params.gradHin->GetDataType()).GetString());
        return false;
    }
    return CheckDtype(params.gradHin, DataType::DT_FLOAT16, "gradHin", true) &&
           CheckDtype(params.gradHPost, DataType::DT_FLOAT, "gradHPost") &&
           CheckDtype(params.gradHRes, DataType::DT_FLOAT, "gradHRes") &&
           CheckDtype(params.x, DataType::DT_FLOAT16, "x", true) && CheckDtype(params.phi, DataType::DT_FLOAT, "phi") &&
           CheckDtype(params.alpha, DataType::DT_FLOAT, "alpha") &&
           CheckDtype(params.bias, DataType::DT_FLOAT, "bias") && CheckDtype(params.hPre, DataType::DT_FLOAT, "hPre") &&
           CheckDtype(params.hcBeforeNorm, DataType::DT_FLOAT, "hcBeforeNorm") &&
           CheckDtype(params.invRms, DataType::DT_FLOAT, "invRms") &&
           CheckDtype(params.sumOut, DataType::DT_FLOAT, "sumOut") &&
           CheckDtype(params.normOut, DataType::DT_FLOAT, "normOut") &&
           (params.preMix == nullptr || CheckDtype(params.preMix, DataType::DT_FLOAT, "preMix")) &&
           (params.gradPre == nullptr || CheckDtype(params.gradPre, DataType::DT_FLOAT, "gradPre"));
}

static bool CheckOutputDtype(const AclnnMhcPreSinkhornBackwardParams& params)
{
    auto gradHInDtype = params.gradHin->GetDataType();
    if (params.gradX->GetDataType() != gradHInDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradX tensor dtype must match gradHin");
        return false;
    }
    return CheckDtype(params.gradPhi, DataType::DT_FLOAT, "gradPhi") &&
           CheckDtype(params.gradAlpha, DataType::DT_FLOAT, "gradAlpha") &&
           CheckDtype(params.gradBias, DataType::DT_FLOAT, "gradBias") &&
           (params.gradPreMix == nullptr || CheckDtype(params.gradPreMix, DataType::DT_FLOAT, "gradPreMix"));
}

static bool CheckDtypeValid(const AclnnMhcPreSinkhornBackwardParams& params)
{
    return CheckInputDtype(params) && CheckOutputDtype(params);
}

static aclnnStatus CheckParams(const AclnnMhcPreSinkhornBackwardParams& params)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 校验可选输入/输出组合: preMix+gradPre 成对; gradPreMix 单向联动
    //    (提前到 dim/shape/dtype 之前: 组合不合法时无需检查具体值, 避免误导性错误信息)
    if ((params.preMix == nullptr) != (params.gradPre == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Invalid optional input combination: preMix and gradPre must be passed together or omitted "
                "together. Passing only one of preMix/gradPre is not supported.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // gradPreMix 单向联动: preMix 在时 gradPreMix 必须在(否则 premix 梯度无处落盘);
    // preMix 缺省但 gradPreMix 传入是无害的——L2 内部按 hasPreMix=0 路径计算,
    // gradPreMix 由内部零值占位张量回拷(l0 始终分配), 调用方拿到全零, 与
    // "不传 gradPreMix"语义等价, 兼容存量调用方(如 atk 用例约定)传已分配
    // 缓冲区但不期望写入的用法
    if (params.preMix != nullptr && params.gradPreMix == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Invalid optional output combination: gradPreMix must be provided when preMix is provided. "
                "Passing preMix without gradPreMix is not supported.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 平台能力校验: premix 能力仅 A2/A3 (非 RegBase) 支持;
    //    A5 (RegBase: ascend950) 与 arch92 (ascend960dt) 侧 tiling/kernel 未实现
    //    preMix/gradPre 路径, 传入即拒绝
    //    (提前到 dim/shape/dtype 之前: 平台不支持时无需检查 preMix 的具体值)
    if (params.preMix != nullptr && Ops::Transformer::AclnnUtil::IsRegbase()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "preMix/gradPre are only supported on A2/A3 and not supported on A5 (ascend950) or arch92 "
                "(ascend960dt) RegBase platforms.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 校验输入、输出参数维度
    CHECK_RET(CheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 5. 校验输入、输出shape参数
    CHECK_RET(CheckShape(params), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    // 7. 校验format：仅支持ND格式，不支持私有格式
    CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);

    // 8. 校验N、C取值范围
    auto& xShape = params.x->GetViewShape();
    bool is3D = xShape.GetDimNum() == 3;
    int64_t nVal = is3D ? xShape.GetDim(1) : xShape.GetDim(2);
    int64_t cVal = is3D ? xShape.GetDim(2) : xShape.GetDim(3);
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool isArch92 = (curArch == NpuArch::DAV_9201 || curArch == NpuArch::DAV_9202);
    if (Ops::Transformer::AclnnUtil::IsRegbase() || isArch92) {
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

static const aclTensor* ConvertToContiguous(const aclTensor* tensor, aclOpExecutor* executor)
{
    auto result = l0op::Contiguous(tensor, executor);
    if (result == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Contiguous result is nullptr");
    }
    return result;
}

static aclnnStatus FillOutputZero(const aclTensor* out, aclOpExecutor* executor)
{
    op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(out->GetViewShape());
    auto shapes = executor->AllocIntArray(fillDims.data(), out->GetViewShape().GetDimNum());
    CHECK_RET(shapes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT32);
    CHECK_RET(dimTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclScalar* valueScalar = executor->AllocScalar(0);
    CHECK_RET(valueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto filled = l0op::Fill(dimTensor, valueTensor, shapes, executor);
    CHECK_RET(filled != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(filled, const_cast<aclTensor*>(out), executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus MhcSinkhornGradCovertDataContiguous(AclnnMhcPreSinkhornBackwardParams& params,
                                                       aclOpExecutor* executor)
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
    if (params.preMix != nullptr) {
        params.preMixContiguous = ConvertToContiguous(params.preMix, executor);
    }
    if (params.gradPre != nullptr) {
        params.gradPreContiguous = ConvertToContiguous(params.gradPre, executor);
    }

    if (params.gradHinContiguous == nullptr || params.gradHPostContiguous == nullptr ||
        params.gradHResContiguous == nullptr || params.xContiguous == nullptr || params.phiContiguous == nullptr ||
        params.alphaContiguous == nullptr || params.biasContiguous == nullptr || params.hPreContiguous == nullptr ||
        params.hcBeforeNormContiguous == nullptr || params.invRmsContiguous == nullptr ||
        params.sumOutContiguous == nullptr || params.normOutContiguous == nullptr ||
        (params.preMix != nullptr && params.preMixContiguous == nullptr) ||
        (params.gradPre != nullptr && params.gradPreContiguous == nullptr)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static bool CopyOutput(const aclTensor* out, const aclTensor* dst, aclOpExecutor* executor)
{
    auto ret = l0op::ViewCopy(out, dst, executor);
    return ret != nullptr;
}

static aclnnStatus mhcPreSinkhornBackwardCommonProcess(AclnnMhcPreSinkhornBackwardParams& params,
                                                       aclOpExecutor* executor)
{
    auto ret = MhcSinkhornGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 放宽后的联动语义(见 CheckParams 注释): preMix 缺省但调用方传入 gradPreMix 时,
    // hasPreMix=0 路径 kernel 不写该缓冲, L2 显式写零保证返回内容确定(全零),
    // 与"不传 gradPreMix"语义等价, 避免未初始化内存透出
    if (params.preMix == nullptr && params.gradPreMix != nullptr) {
        auto zeroRet = FillOutputZero(params.gradPreMix, executor);
        CHECK_RET(zeroRet == ACLNN_SUCCESS, zeroRet);
    }

    auto outParams = l0op::MhcPreSinkhornBackward(
        params.gradHinContiguous, params.gradHPostContiguous, params.gradHResContiguous, params.xContiguous,
        params.phiContiguous, params.alphaContiguous, params.biasContiguous, params.hPreContiguous,
        params.hcBeforeNormContiguous, params.invRmsContiguous, params.sumOutContiguous, params.normOutContiguous,
        params.preMixContiguous, params.gradPreContiguous, static_cast<float>(params.hcEps), params.enableHf32,
        executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CopyOutput(std::get<0>(outParams), params.gradX, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<1>(outParams), params.gradPhi, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<2>(outParams), params.gradAlpha, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<3>(outParams), params.gradBias, executor), ACLNN_ERR_INNER_NULLPTR);
    if (params.gradPreMix != nullptr) {
        CHECK_RET(CopyOutput(std::get<4>(outParams), params.gradPreMix, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornBackwardV2GetWorkspaceSize(
    const aclTensor* gradHin, const aclTensor* gradHPost, const aclTensor* gradHRes, const aclTensor* x,
    const aclTensor* phi, const aclTensor* alpha, const aclTensor* bias, const aclTensor* hPre,
    const aclTensor* hcBeforeNorm, const aclTensor* invRms, const aclTensor* sumOut, const aclTensor* normOut,
    const aclTensor* preMix, const aclTensor* gradPre, double hcEps, bool enableHf32, const aclTensor* gradX,
    const aclTensor* gradPhi, const aclTensor* gradAlpha, const aclTensor* gradBias, const aclTensor* gradPreMix,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreSinkhornBackwardV2,
                   DFX_IN(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                          normOut, preMix, gradPre, hcEps, enableHf32),
                   DFX_OUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    AclnnMhcPreSinkhornBackwardParams params =
        AclnnMhcPreSinkhornBackward::Create()
            .SetGradInput(gradHin, gradHPost, gradHRes)
            .SetInput(x, phi, alpha, bias)
            .SetForwardInput(hPre, hcBeforeNorm, invRms, sumOut, normOut, preMix, gradPre)
            .SetAttr(hcEps, enableHf32)
            .SetOutput(gradX, gradPhi, gradAlpha, gradBias, gradPreMix)
            .Build();

    auto ret = CheckNotNull(params) ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_NULLPTR;
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradHin->IsEmpty() || gradHPost->IsEmpty() || gradHRes->IsEmpty() || x->IsEmpty() || hPre->IsEmpty() ||
        hcBeforeNorm->IsEmpty() || invRms->IsEmpty() || sumOut->IsEmpty() || normOut->IsEmpty()) {
        OP_LOGW("[aclnnMhcPreSinkhornBackward] Input tensor is empty, skip computation and return success.");
        // 空输入（如 B=0）时梯度数学上为全 0：非空输出必须写出 0，避免未初始化内存透出给调用方
        for (const aclTensor* out : {gradX, gradPhi, gradAlpha, gradBias}) {
            if (!out->IsEmpty()) {
                CHECK_RET(FillOutputZero(out, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
            }
        }
        if (preMix != nullptr && gradPreMix != nullptr && !gradPreMix->IsEmpty()) {
            CHECK_RET(FillOutputZero(gradPreMix, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
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

aclnnStatus aclnnMhcPreSinkhornBackwardV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhornBackwardV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
