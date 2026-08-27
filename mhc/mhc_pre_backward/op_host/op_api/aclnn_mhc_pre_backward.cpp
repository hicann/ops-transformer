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
#include "mhc_pre_backward.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct AclnnMhcPreBackwardParams {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *gradHIn = nullptr;
    const aclTensor *gradHPost = nullptr;
    const aclTensor *gradHRes = nullptr;
    const aclTensor *invRms = nullptr;
    const aclTensor *hMix = nullptr;
    const aclTensor *hPre = nullptr;
    const aclTensor *hPost = nullptr;
    const aclTensor *gamma = nullptr;
    const aclTensor *gradXPostOptional = nullptr;
    float hcEps;

    const aclTensor *gradX = nullptr;
    const aclTensor *gradPhi = nullptr;
    const aclTensor *gradAlpha = nullptr;
    const aclTensor *gradBias = nullptr;
    const aclTensor *gradGamma = nullptr;

    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *gradHInContiguous = nullptr;
    const aclTensor *gradHPostContiguous = nullptr;
    const aclTensor *gradHResContiguous = nullptr;
    const aclTensor *invRmsContiguous = nullptr;
    const aclTensor *hMixContiguous = nullptr;
    const aclTensor *hPreContiguous = nullptr;
    const aclTensor *hPostContiguous = nullptr;
    const aclTensor *gammaContiguous = nullptr;
    const aclTensor *gradXPostOptionalContiguous = nullptr;
};

class AclnnMhcPreBackward {
public:
    static AclnnMhcPreBackward Create()
    {
        AclnnMhcPreBackward obj;

        return obj;
    }

    AclnnMhcPreBackward &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                  const aclTensor *gamma)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.gamma = gamma;

        return *this;
    }

    AclnnMhcPreBackward &SetForwardInput(const aclTensor *invRms, const aclTensor *hMix, const aclTensor *hPre,
                                         const aclTensor *hPost)
    {
        obj_.invRms = invRms;
        obj_.hMix = hMix;
        obj_.hPre = hPre;
        obj_.hPost = hPost;

        return *this;
    }

    AclnnMhcPreBackward &SetGradInput(const aclTensor *gradHIn, const aclTensor *gradHPost, const aclTensor *gradHRes)
    {
        obj_.gradHIn = gradHIn;
        obj_.gradHPost = gradHPost;
        obj_.gradHRes = gradHRes;
        return *this;
    }

    AclnnMhcPreBackward &SetGradXPostOptional(const aclTensor *gradXPostOptional)
    {
        obj_.gradXPostOptional = gradXPostOptional;
        return *this;
    }

    AclnnMhcPreBackward &SetAttr(float hcEps)
    {
        obj_.hcEps = hcEps;

        return *this;
    }

    AclnnMhcPreBackward &SetOutput(const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
                                   const aclTensor *gradBias, const aclTensor *gradGamma)
    {
        obj_.gradX = gradX;
        obj_.gradPhi = gradPhi;
        obj_.gradAlpha = gradAlpha;
        obj_.gradBias = gradBias;
        obj_.gradGamma = gradGamma;
        return *this;
    }

    AclnnMhcPreBackwardParams Build() const
    {
        return obj_;
    }

private:
    AclnnMhcPreBackwardParams obj_;
};

bool MhcGradCheckInputNotNullImpl(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensor is nullptr", name);
        return false;
    }
    return true;
}

bool MhcGradCheckInputNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNullImpl(params.x, "x") && MhcGradCheckInputNotNullImpl(params.phi, "phi") &&
           MhcGradCheckInputNotNullImpl(params.alpha, "alpha") &&
           MhcGradCheckInputNotNullImpl(params.gradHIn, "gradHIn") &&
           MhcGradCheckInputNotNullImpl(params.gradHPost, "gradHPost") &&
           MhcGradCheckInputNotNullImpl(params.gradHRes, "gradHRes") &&
           MhcGradCheckInputNotNullImpl(params.invRms, "invRms") && MhcGradCheckInputNotNullImpl(params.hMix, "hMix") &&
           MhcGradCheckInputNotNullImpl(params.hPre, "hPre") && MhcGradCheckInputNotNullImpl(params.hPost, "hPost") &&
           (params.gamma == nullptr || MhcGradCheckInputNotNullImpl(params.gamma, "gamma")) &&
           (params.gradXPostOptional == nullptr ||
            MhcGradCheckInputNotNullImpl(params.gradXPostOptional, "gradXPostOptional"));
}

bool MhcGradCheckOutputNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNullImpl(params.gradX, "gradX") &&
           MhcGradCheckInputNotNullImpl(params.gradPhi, "gradPhi") &&
           MhcGradCheckInputNotNullImpl(params.gradAlpha, "gradAlpha") &&
           MhcGradCheckInputNotNullImpl(params.gradBias, "gradBias") &&
           (params.gamma == nullptr || MhcGradCheckInputNotNullImpl(params.gradGamma, "gradGamma"));
}

bool MhcGradCheckNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNull(params) && MhcGradCheckOutputNotNull(params);
}

bool MhcGradCheckEmptyTensorImpl(const aclTensor *tensor, const char *name)
{
    if (tensor->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor is empty", name);
        return false;
    }
    return true;
}

bool MhcGradCheckEmptyTensor(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckEmptyTensorImpl(params.x, "x") && MhcGradCheckEmptyTensorImpl(params.phi, "phi") &&
           MhcGradCheckEmptyTensorImpl(params.alpha, "alpha") &&
           MhcGradCheckEmptyTensorImpl(params.gradHIn, "gradHIn") &&
           MhcGradCheckEmptyTensorImpl(params.gradHPost, "gradHPost") &&
           MhcGradCheckEmptyTensorImpl(params.gradHRes, "gradHRes") &&
           MhcGradCheckEmptyTensorImpl(params.invRms, "invRms") && MhcGradCheckEmptyTensorImpl(params.hMix, "hMix") &&
           MhcGradCheckEmptyTensorImpl(params.hPre, "hPre") && MhcGradCheckEmptyTensorImpl(params.hPost, "hPost") &&
           (params.gamma == nullptr || MhcGradCheckEmptyTensorImpl(params.gamma, "gamma")) &&
           (params.gradXPostOptional == nullptr ||
            MhcGradCheckEmptyTensorImpl(params.gradXPostOptional, "gradXPostOptional"));
}

bool CheckDimNum(const aclTensor *tensor, size_t expected, const char *name, bool isRange = false, size_t expected2 = 0)
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

bool MhcGradCheckInputDims(const AclnnMhcPreBackwardParams &params)
{
    return CheckDimNum(params.x, 3, "x", true, 4) && CheckDimNum(params.phi, 2, "phi") &&
           CheckDimNum(params.alpha, 1, "alpha") &&
           (params.gamma == nullptr || CheckDimNum(params.gamma, 2, "gamma")) &&
           CheckDimNum(params.gradHIn, (params.x)->GetViewShape().GetDimNum() - 1, "gradHIn") &&
           CheckDimNum(params.gradHPost, (params.x)->GetViewShape().GetDimNum() - 1, "gradHPost") &&
           CheckDimNum(params.gradHRes, (params.x)->GetViewShape().GetDimNum(), "gradHRes", true,
                       (params.x)->GetViewShape().GetDimNum() - 1) &&
           CheckDimNum(params.invRms, (params.x)->GetViewShape().GetDimNum() - 2, "invRms") &&
           CheckDimNum(params.hMix, (params.x)->GetViewShape().GetDimNum() - 1, "hMix") &&
           CheckDimNum(params.hPre, (params.x)->GetViewShape().GetDimNum() - 1, "hPre") &&
           CheckDimNum(params.hPost, (params.x)->GetViewShape().GetDimNum() - 1, "hPost") &&
           (params.gradXPostOptional == nullptr ||
            CheckDimNum(params.gradXPostOptional, (params.x)->GetViewShape().GetDimNum(), "gradXPostOptional"));
}

bool MhcGradCheckOutputDims(const AclnnMhcPreBackwardParams &params)
{
    auto gradHInDimNum = params.gradHIn->GetViewShape().GetDimNum();
    auto gradXDimNum = params.gradX->GetViewShape().GetDimNum();
    if (gradHInDimNum == 3) {
        if (gradXDimNum != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradX tensor dim num must be 4 for BSND format, but got %zu",
                    gradXDimNum);
            return false;
        }
    } else {
        if (gradXDimNum != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradX tensor dim num must be 3 for TND format, but got %zu", gradXDimNum);
            return false;
        }
    }
    return CheckDimNum(params.gradPhi, 2, "gradPhi") && CheckDimNum(params.gradAlpha, 1, "gradAlpha") &&
           CheckDimNum(params.gradBias, 1, "gradBias") &&
           (params.gamma == nullptr || CheckDimNum(params.gradGamma, 2, "gradGamma"));
}

bool MhcGradCheckInputOutDims(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputDims(params) && MhcGradCheckOutputDims(params);
}

bool CheckShapeDim(const gert::Shape &shape, uint64_t index, uint64_t expected, const char *msg)
{
    if (shape.GetDim(index) != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, and dim %lu must be %lu, actual is %ld", msg, index, expected,
                shape.GetDim(index));
        return false;
    }
    return true;
}

uint64_t Factorial(uint64_t n)
{
    if (n <= 1) {
        return 1;
    }
    uint64_t result = 1;
    for (uint64_t i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

bool IsAscend910BPlatform()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    return socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
}

bool MhcGradCheckBSNDShape(const AclnnMhcPreBackwardParams &params, uint64_t batch, uint64_t sequence, uint64_t dimen,
                           uint64_t numsResidual)
{
    auto &xShape = params.x->GetViewShape();
    if (!CheckShapeDim(xShape, 0, batch, "x tensor shape must be [B, S, N, D]") ||
        !CheckShapeDim(xShape, 1, sequence, "x tensor shape must be [B, S, N, D]") ||
        !CheckShapeDim(xShape, 2, numsResidual, "x tensor shape must be [B, S, N, D]") ||
        !CheckShapeDim(xShape, 3, dimen, "x tensor shape must be [B, S, N, D]")) {
        return false;
    }

    auto &gradHPostShape = params.gradHPost->GetViewShape();
    if (!CheckShapeDim(gradHPostShape, 0, batch, "gradHPost shape must align with gradHIn on B and S dims") ||
        !CheckShapeDim(gradHPostShape, 1, sequence, "gradHPost shape must align with gradHIn on B and S dims")) {
        return false;
    }

    // 从 phi shape 动态读取 fusionSize, 支持 N!+2N 和 N²+2N
    uint64_t fusionSize = params.phi->GetViewShape().GetDim(0);
    uint64_t factN = Factorial(numsResidual); // BSN! 形式的 factN = N!
    auto &gradHResShape = params.gradHRes->GetViewShape();
    if (gradHResShape.GetDimNum() == 4) {
        // [B, S, N, N] (BSNN 形式) - 所有平台都支持, factN = N²
        if (!CheckShapeDim(gradHResShape, 0, batch, "gradHRes shape must be [B, S, N, N]") ||
            !CheckShapeDim(gradHResShape, 1, sequence, "gradHRes shape must be [B, S, N, N]") ||
            !CheckShapeDim(gradHResShape, 2, numsResidual, "gradHRes shape must be [B, S, N, N]") ||
            !CheckShapeDim(gradHResShape, 3, numsResidual, "gradHRes shape must be [B, S, N, N]")) {
            return false;
        }
    } else if (gradHResShape.GetDimNum() == 3) {
        // [B, S, N!] (BSN! 形式) - 仅 A2 支持, A5 只支持 BSNN
        if (!IsAscend910BPlatform()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradHRes shape [B, S, N!] (BSN!) is only supported on A2 platform");
            return false;
        }
        if (!CheckShapeDim(gradHResShape, 0, batch, "gradHRes shape must be [B, S, N, N] or [B, S, N!]") ||
            !CheckShapeDim(gradHResShape, 1, sequence, "gradHRes shape must be [B, S, N, N] or [B, S, N!]") ||
            !CheckShapeDim(gradHResShape, 2, factN, "gradHRes shape must be [B, S, N, N] or [B, S, N!]")) {
            return false;
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradHRes shape dim num must be 3 (BSN!) or 4 (BSNN), but got %lu",
                gradHResShape.GetDimNum());
        return false;
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [B, S]") ||
        !CheckShapeDim(invRmsShape, 1, sequence, "invRms shape must be [B, S]")) {
        return false;
    }

    auto &hMixShape = params.hMix->GetViewShape();
    if (!CheckShapeDim(hMixShape, 0, batch, "hMix shape must be [B, S, fusionSize]") ||
        !CheckShapeDim(hMixShape, 1, sequence, "hMix shape must be [B, S, fusionSize]") ||
        !CheckShapeDim(hMixShape, 2, fusionSize, "hMix shape must be [B, S, fusionSize]")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, batch, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 1, sequence, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 2, numsResidual, "hPre shape must be [B, S, N]")) {
        return false;
    }

    auto &hPostShape = params.hPost->GetViewShape();
    if (!CheckShapeDim(hPostShape, 0, batch, "hPost shape must be [B, S, N]") ||
        !CheckShapeDim(hPostShape, 1, sequence, "hPost shape must be [B, S, N]") ||
        !CheckShapeDim(hPostShape, 2, numsResidual, "hPost shape must be [B, S, N]")) {
        return false;
    }

    auto &gradXShape = params.gradX->GetViewShape();
    if (!CheckShapeDim(gradXShape, 0, batch, "gradX shape must be [B, S, N, D]") ||
        !CheckShapeDim(gradXShape, 1, sequence, "gradX shape must be [B, S, N, D]") ||
        !CheckShapeDim(gradXShape, 2, numsResidual, "gradX shape must be [B, S, N, D]") ||
        !CheckShapeDim(gradXShape, 3, dimen, "gradX shape must be [B, S, N, D]")) {
        return false;
    }

    if (params.gradXPostOptional != nullptr) {
        auto &gradXPostShape = params.gradXPostOptional->GetViewShape();
        if (!CheckShapeDim(gradXPostShape, 0, batch, "gradXPostOptional shape must be [B, S, N, D]") ||
            !CheckShapeDim(gradXPostShape, 1, sequence, "gradXPostOptional shape must be [B, S, N, D]") ||
            !CheckShapeDim(gradXPostShape, 2, numsResidual, "gradXPostOptional shape must be [B, S, N, D]") ||
            !CheckShapeDim(gradXPostShape, 3, dimen, "gradXPostOptional shape must be [B, S, N, D]")) {
            return false;
        }
    }
    return true;
}

bool MhcGradCheckTNDShape(const AclnnMhcPreBackwardParams &params, uint64_t t, uint64_t dimen, uint64_t numsResidual)
{
    auto &xShape = params.x->GetViewShape();
    if (!CheckShapeDim(xShape, 0, t, "x tensor shape must be [T, N, D]") ||
        !CheckShapeDim(xShape, 1, numsResidual, "x tensor shape must be [T, N, D]") ||
        !CheckShapeDim(xShape, 2, dimen, "x tensor shape must be [T, N, D]")) {
        return false;
    }

    auto &gradHPostShape = params.gradHPost->GetViewShape();
    if (!CheckShapeDim(gradHPostShape, 0, t, "gradHPost shape must align with gradHIn on T dim")) {
        return false;
    }

    // 从 phi shape 动态读取 fusionSize, 支持 N!+2N 和 N²+2N
    uint64_t fusionSize = params.phi->GetViewShape().GetDim(0);
    uint64_t factN = Factorial(numsResidual); // BSN! 形式的 factN = N!
    auto &gradHResShape = params.gradHRes->GetViewShape();
    if (gradHResShape.GetDimNum() == 3) {
        // [T, N, N] (BSNN 形式) - 所有平台都支持, factN = N²
        if (!CheckShapeDim(gradHResShape, 0, t, "gradHRes shape must be [T, N, N]") ||
            !CheckShapeDim(gradHResShape, 1, numsResidual, "gradHRes shape must be [T, N, N]") ||
            !CheckShapeDim(gradHResShape, 2, numsResidual, "gradHRes shape must be [T, N, N]")) {
            return false;
        }
    } else {
        // [T, N!] (BSN! 形式) - 仅 A2 支持, A5 只支持 BSNN
        if (!IsAscend910BPlatform()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradHRes shape [T, N!] (BSN!) is only supported on A2 platform");
            return false;
        }
        if (!CheckShapeDim(gradHResShape, 0, t, "gradHRes shape must be [T, N, N] or [T, N!]") ||
            !CheckShapeDim(gradHResShape, 1, factN, "gradHRes shape must be [T, N, N] or [T, N!]")) {
            return false;
        }
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, t, "invRms shape must be [T]")) {
        return false;
    }

    auto &hMixShape = params.hMix->GetViewShape();
    if (!CheckShapeDim(hMixShape, 0, t, "hMix shape must be [T, fusionSize]") ||
        !CheckShapeDim(hMixShape, 1, fusionSize, "hMix shape must be [T, fusionSize]")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, t, "hPre shape must be [T, N]") ||
        !CheckShapeDim(hPreShape, 1, numsResidual, "hPre shape must be [T, N]")) {
        return false;
    }

    auto &hPostShape = params.hPost->GetViewShape();
    if (!CheckShapeDim(hPostShape, 0, t, "hPost shape must be [T, N]") ||
        !CheckShapeDim(hPostShape, 1, numsResidual, "hPost shape must be [T, N]")) {
        return false;
    }

    auto &gradXShape = params.gradX->GetViewShape();
    if (!CheckShapeDim(gradXShape, 0, t, "gradX shape must be [T, N, D]") ||
        !CheckShapeDim(gradXShape, 1, numsResidual, "gradX shape must be [T, N, D]") ||
        !CheckShapeDim(gradXShape, 2, dimen, "gradX shape must be [T, N, D]")) {
        return false;
    }

    if (params.gradXPostOptional != nullptr) {
        auto &gradXPostShape = params.gradXPostOptional->GetViewShape();
        if (!CheckShapeDim(gradXPostShape, 0, t, "gradXPostOptional shape must be [T, N, D]") ||
            !CheckShapeDim(gradXPostShape, 1, numsResidual, "gradXPostOptional shape must be [T, N, D]") ||
            !CheckShapeDim(gradXPostShape, 2, dimen, "gradXPostOptional shape must be [T, N, D]")) {
            return false;
        }
    }
    return true;
}

bool MhcGradCheckCommonShape(const AclnnMhcPreBackwardParams &params, uint64_t numsResidual, uint64_t dimen)
{
    auto &alphaShape = params.alpha->GetViewShape();
    if (!CheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    uint64_t nD = numsResidual * dimen;

    auto &phiShape = params.phi->GetViewShape();
    uint64_t fusionSize = phiShape.GetDim(0);
    // 校验 phi 第一维: 必须是 N!+2N 或 N²+2N
    uint64_t expectedFusionSizeFactN = Factorial(numsResidual) + 2 * numsResidual;   // N!+2N
    uint64_t expectedFusionSizeNsq = numsResidual * numsResidual + 2 * numsResidual; // N²+2N
    if (fusionSize != expectedFusionSizeFactN && fusionSize != expectedFusionSizeNsq) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor first dim must be N!+2N (%lu) or N^2+2N (%lu), but got %lu",
                expectedFusionSizeFactN, expectedFusionSizeNsq, fusionSize);
        return false;
    }
    if (!CheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be nD")) {
        return false;
    }

    // Verify hMix fusionSize matches phi
    auto &hMixShape = params.hMix->GetViewShape();
    uint64_t hMixLastDim = hMixShape.GetDim(hMixShape.GetDimNum() - 1);
    if (hMixLastDim != fusionSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hMix last dim must match phi fusionSize %lu, but got %lu", fusionSize,
                hMixLastDim);
        return false;
    }

    auto &gradPhiShape = params.gradPhi->GetViewShape();
    if (!CheckShapeDim(gradPhiShape, 0, fusionSize, "gradPhi shape first dim must match phi") ||
        !CheckShapeDim(gradPhiShape, 1, nD, "gradPhi shape second dim must be nD")) {
        return false;
    }

    auto &gradAlphaShape = params.gradAlpha->GetViewShape();
    if (!CheckShapeDim(gradAlphaShape, 0, 3, "gradAlpha shape must be [3]")) {
        return false;
    }

    auto &gradBiasShape = params.gradBias->GetViewShape();
    if (!CheckShapeDim(gradBiasShape, 0, fusionSize, "gradBias shape must match fusionSize")) {
        return false;
    }

    if (params.gamma != nullptr) {
        auto &gammaShape = params.gamma->GetViewShape();
        if (!CheckShapeDim(gammaShape, 0, numsResidual, "gamma shape must be [N, D]") ||
            !CheckShapeDim(gammaShape, 1, dimen, "gamma shape must be [N, D]")) {
            return false;
        }
        auto &gradGammaShape = params.gradGamma->GetViewShape();
        if (!CheckShapeDim(gradGammaShape, 0, numsResidual, "gradGamma shape must be [N, D]") ||
            !CheckShapeDim(gradGammaShape, 1, dimen, "gradGamma shape must be [N, D]")) {
            return false;
        }
    }
    return true;
}

bool MhcGradCheckInputOutShape(const AclnnMhcPreBackwardParams &params)
{
    auto &gradHInShape = params.gradHIn->GetViewShape();
    auto &gradHPostShape = params.gradHPost->GetViewShape();
    auto gradHInDimNum = gradHInShape.GetDimNum();

    uint64_t batch = 0;
    uint64_t sequence = 0;
    uint64_t t = 0;
    uint64_t dimen = 0;
    uint64_t numsResidual = 0;

    if (gradHInDimNum == 3) {
        batch = gradHInShape.GetDim(0);
        sequence = gradHInShape.GetDim(1);
        dimen = gradHInShape.GetDim(2);
        numsResidual = gradHPostShape.GetDim(2);
        if (!MhcGradCheckBSNDShape(params, batch, sequence, dimen, numsResidual)) {
            return false;
        }
    } else if (gradHInDimNum == 2) {
        t = gradHInShape.GetDim(0);
        dimen = gradHInShape.GetDim(1);
        numsResidual = gradHPostShape.GetDim(1);
        if (!MhcGradCheckTNDShape(params, t, dimen, numsResidual)) {
            return false;
        }
    } else {
        return false;
    }

    return MhcGradCheckCommonShape(params, numsResidual, dimen);
}

bool MhcGradIsValidXType(DataType dtype)
{
    return dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16;
}

bool CheckDtype(const aclTensor *tensor, DataType expected, const char *name, bool isValidX = false)
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

bool MhcGradCheckInputDtype(const AclnnMhcPreBackwardParams &params)
{
    return CheckDtype(params.x, DataType::DT_FLOAT16, "x", true) && CheckDtype(params.phi, DataType::DT_FLOAT, "phi") &&
           CheckDtype(params.alpha, DataType::DT_FLOAT, "alpha") &&
           (params.gamma == nullptr || CheckDtype(params.gamma, DataType::DT_FLOAT, "gamma")) &&
           CheckDtype(params.gradHIn, params.x->GetDataType(), "gradHIn") &&
           CheckDtype(params.gradHPost, DataType::DT_FLOAT, "gradHPost") &&
           CheckDtype(params.gradHRes, DataType::DT_FLOAT, "gradHRes") &&
           CheckDtype(params.invRms, DataType::DT_FLOAT, "invRms") &&
           CheckDtype(params.hMix, DataType::DT_FLOAT, "hMix") && CheckDtype(params.hPre, DataType::DT_FLOAT, "hPre") &&
           CheckDtype(params.hPost, DataType::DT_FLOAT, "hPost") &&
           (params.gradXPostOptional == nullptr ||
            CheckDtype(params.gradXPostOptional, params.x->GetDataType(), "gradXPostOptional"));
}

bool MhcGradCheckOutputDtype(const AclnnMhcPreBackwardParams &params)
{
    auto gradHInDtype = params.gradHIn->GetDataType();
    if (params.gradX->GetDataType() != gradHInDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradX tensor dtype must match gradHIn");
        return false;
    }
    return CheckDtype(params.gradPhi, DataType::DT_FLOAT, "gradPhi") &&
           CheckDtype(params.gradAlpha, DataType::DT_FLOAT, "gradAlpha") &&
           CheckDtype(params.gradBias, DataType::DT_FLOAT, "gradBias") &&
           (params.gamma == nullptr || CheckDtype(params.gradGamma, DataType::DT_FLOAT, "gradGamma"));
}

bool MhcGradCheckDtypeValid(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputDtype(params) && MhcGradCheckOutputDtype(params);
}

static bool MhcGradIsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }

    return false;
}

static bool CheckFormat(const aclTensor *tensor, const char *name)
{
    if (MhcGradIsPrivateFormat(tensor->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor format must be ND", name);
        return false;
    }
    return true;
}

bool MhcGradCheckInputFormat(const AclnnMhcPreBackwardParams &params)
{
    return CheckFormat(params.x, "x") && CheckFormat(params.phi, "phi") && CheckFormat(params.alpha, "alpha") &&
           (params.gamma == nullptr || CheckFormat(params.gamma, "gamma")) && CheckFormat(params.gradHIn, "gradHIn") &&
           CheckFormat(params.gradHPost, "gradHPost") && CheckFormat(params.gradHRes, "gradHRes") &&
           CheckFormat(params.invRms, "invRms") && CheckFormat(params.hMix, "hMix") &&
           CheckFormat(params.hPre, "hPre") && CheckFormat(params.hPost, "hPost") &&
           (params.gradXPostOptional == nullptr || CheckFormat(params.gradXPostOptional, "gradXPostOptional"));
}

bool MhcGradCheckFormat(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputFormat(params);
}

aclnnStatus MhcGradCheckParams(const AclnnMhcPreBackwardParams &params)
{
    // 1. 检查参数是否为空指针、空tensor
    CHECK_RET(MhcGradCheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(MhcGradCheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);

    // 2. 校验输入、输出参数维度
    CHECK_RET(MhcGradCheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验输入、输出shape参数
    CHECK_RET(MhcGradCheckInputOutShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(MhcGradCheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据形状是否支持
    CHECK_RET(MhcGradCheckFormat(params), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

const aclTensor *ConvertToContiguous(const aclTensor *tensor, aclOpExecutor *executor)
{
    auto result = l0op::Contiguous(tensor, executor);
    if (result == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Contiguous result is nullptr");
    }
    return result;
}

aclnnStatus MhcGradCovertDataContiguous(AclnnMhcPreBackwardParams &params, aclOpExecutor *executor)
{
    params.xContiguous = ConvertToContiguous(params.x, executor);
    params.phiContiguous = ConvertToContiguous(params.phi, executor);
    params.alphaContiguous = ConvertToContiguous(params.alpha, executor);
    params.gradHInContiguous = ConvertToContiguous(params.gradHIn, executor);
    params.gradHPostContiguous = ConvertToContiguous(params.gradHPost, executor);
    params.gradHResContiguous = ConvertToContiguous(params.gradHRes, executor);
    params.invRmsContiguous = ConvertToContiguous(params.invRms, executor);
    params.hMixContiguous = ConvertToContiguous(params.hMix, executor);
    params.hPreContiguous = ConvertToContiguous(params.hPre, executor);
    params.hPostContiguous = ConvertToContiguous(params.hPost, executor);

    if (params.xContiguous == nullptr || params.phiContiguous == nullptr || params.alphaContiguous == nullptr ||
        params.gradHInContiguous == nullptr || params.gradHPostContiguous == nullptr ||
        params.gradHResContiguous == nullptr || params.invRmsContiguous == nullptr ||
        params.hMixContiguous == nullptr || params.hPreContiguous == nullptr || params.hPostContiguous == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    if (params.gamma != nullptr) {
        params.gammaContiguous = ConvertToContiguous(params.gamma, executor);
        if (params.gammaContiguous == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (params.gradXPostOptional != nullptr) {
        params.gradXPostOptionalContiguous = ConvertToContiguous(params.gradXPostOptional, executor);
        if (params.gradXPostOptionalContiguous == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

bool CopyOutput(const aclTensor *out, const aclTensor *dst, aclOpExecutor *executor)
{
    auto ret = l0op::ViewCopy(out, dst, executor);
    return ret != nullptr;
}

static aclnnStatus mhcPreBackwardCommonProcess(AclnnMhcPreBackwardParams &params, aclOpExecutor *executor)
{
    auto ret = MhcGradCheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MhcGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto outParams =
        l0op::MhcPreBackward(params.xContiguous, params.phiContiguous, params.alphaContiguous, params.gradHInContiguous,
                             params.gradHPostContiguous, params.gradHResContiguous, params.invRmsContiguous,
                             params.hMixContiguous, params.hPreContiguous, params.hPostContiguous,
                             params.gammaContiguous, params.gradXPostOptionalContiguous, params.hcEps, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<0>(outParams), params.gradX, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<1>(outParams), params.gradPhi, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<2>(outParams), params.gradAlpha, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<3>(outParams), params.gradBias, executor), ACLNN_ERR_INNER_NULLPTR);
    if (params.gamma != nullptr) {
        CHECK_RET(CopyOutput(std::get<4>(outParams), params.gradGamma, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gradHIn,
    const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *invRms, const aclTensor *hMix,
    const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma, const aclTensor *gradXPostOptional,
    float hcEps, const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
    const aclTensor *gradBias, const aclTensor *gradGamma, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnMhcPreBackward,
        DFX_IN(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, gradXPostOptional),
        DFX_OUT(gradX, gradPhi, gradAlpha, gradBias, gradGamma));
    auto uniqueExecutor = CREATE_EXECUTOR();
    AclnnMhcPreBackwardParams params = AclnnMhcPreBackward::Create()
                                           .SetInput(x, phi, alpha, gamma)
                                           .SetGradInput(gradHIn, gradHPost, gradHRes)
                                           .SetGradXPostOptional(gradXPostOptional)
                                           .SetForwardInput(invRms, hMix, hPre, hPost)
                                           .SetAttr(hcEps)
                                           .SetOutput(gradX, gradPhi, gradAlpha, gradBias, gradGamma)
                                           .Build();
    auto ret = mhcPreBackwardCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
