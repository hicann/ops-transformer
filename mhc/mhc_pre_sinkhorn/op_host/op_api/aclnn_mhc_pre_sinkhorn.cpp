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
 * \file aclnn_mhc_pre_sinkhorn.cpp
 * \brief aclnn_mhc_pre_sinkhorn
 */

#include "aclnn_mhc_pre_sinkhorn.h"
#include "mhc_pre_sinkhorn.h"
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

static const std::initializer_list<op::DataType> DTYPE_X_SUPPORT_LIST = {op::DataType::DT_BF16,
                                                                         op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_OTHER_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static constexpr int64_t HCMULT_DEFAULT_VALUE = 4;
static constexpr int64_t NUM_ITERS_DEFAULT_VALUE = 20;

static bool IsAscend950()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
}

static bool CheckNotNull(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                         const aclTensor *hin, const aclTensor *hPost, const aclTensor *hRes)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(phi, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(hin, return false);
    OP_CHECK_NULL(hPost, return false);
    OP_CHECK_NULL(hRes, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                            const aclTensor *hin, const aclTensor *hPost, const aclTensor *hRes, const aclTensor *hPre,
                            const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut,
                            const aclTensor *normOut, bool needBackward)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_X_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(phi, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(alpha, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(hin, DTYPE_X_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(hPost, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(hRes, DTYPE_OTHER_SUPPORT_LIST, return false);

    if (needBackward) {
        OP_CHECK_DTYPE_NOT_SUPPORT(hPre, DTYPE_OTHER_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(hcBeforeNorm, DTYPE_OTHER_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(invRms, DTYPE_OTHER_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(sumOut, DTYPE_OTHER_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(normOut, DTYPE_OTHER_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckShape(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                       const aclTensor *hin, const aclTensor *hPost, const aclTensor *hRes, const aclTensor *hPre,
                       const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut,
                       const aclTensor *normOut, int64_t hcMult, int64_t numIters, bool needBackward)
{
    auto xShape = x->GetViewShape();
    auto xDim = xShape.GetDimNum();
    bool isTnd = (xDim == 3);

    if (isTnd && !IsAscend950()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "TND format (3D input) is only supported on Ascend950, but got dim = %ld on current platform.", xDim);
        return false;
    }

    if (xDim != 3 && xDim != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim must be 3 (t, n, c) or 4 (b, s, n, c), but got dim = %ld.", xDim);
        return false;
    }

    int64_t bs = 0;
    int64_t seqLen = 0;
    int64_t n = 0;
    int64_t c = 0;
    if (isTnd) {
        seqLen = xShape.GetDim(0);
        n = xShape.GetDim(1);
        c = xShape.GetDim(2);
        bs = seqLen;
    } else {
        bs = xShape.GetDim(0);
        seqLen = xShape.GetDim(1);
        n = xShape.GetDim(2);
        c = xShape.GetDim(3);
    }
    int64_t hcMix = n * n + 2 * n;

    if (hcMult != HCMULT_DEFAULT_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcMult must be in range [%ld], but got hcMult = %ld.", HCMULT_DEFAULT_VALUE,
                hcMult);
        return false;
    }

    if (n != hcMult) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x shape[2] (n=%ld) must equal hcMult (%ld), but they are different.", n,
                hcMult);
        return false;
    }

    if (numIters != NUM_ITERS_DEFAULT_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numIters must be in range [%ld], but got numIters = %ld.",
                NUM_ITERS_DEFAULT_VALUE, numIters);
        return false;
    }

    auto hinShape = hin->GetViewShape();
    if (isTnd) {
        if (hinShape.GetDimNum() != 2 || hinShape.GetDim(0) != seqLen || hinShape.GetDim(1) != c) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hin shape must be (%ld, %ld), but got (%ld, %ld).", seqLen, c,
                    hinShape.GetDim(0), hinShape.GetDim(1));
            return false;
        }
    } else {
        if (hinShape.GetDimNum() != 3 || hinShape.GetDim(0) != bs || hinShape.GetDim(1) != seqLen ||
            hinShape.GetDim(2) != c) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hin shape must be (%ld, %ld, %ld), but got (%ld, %ld, %ld).", bs, seqLen,
                    c, hinShape.GetDim(0), hinShape.GetDim(1), hinShape.GetDim(2));
            return false;
        }
    }

    auto phiShape = phi->GetViewShape();
    if (phiShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi must be 2D (hcMix, n*c), but got dim = %ld.", phiShape.GetDimNum());
        return false;
    }
    if (phiShape.GetDim(0) != hcMix || phiShape.GetDim(1) != n * c) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi shape must be (%ld, %ld), but got (%ld, %ld).", hcMix, n * c,
                phiShape.GetDim(0), phiShape.GetDim(1));
        return false;
    }

    auto alphaShape = alpha->GetViewShape();
    if (alphaShape.GetDimNum() != 1 || alphaShape.GetDim(0) != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha shape must be (3,), but got shape with dim = %ld, size = %ld.",
                alphaShape.GetDimNum(), alphaShape.GetDimNum() > 0 ? alphaShape.GetDim(0) : 0);
        return false;
    }

    auto biasShape = bias->GetViewShape();
    if (biasShape.GetDimNum() != 1 || biasShape.GetDim(0) != hcMix) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias shape must be (%ld,), but got shape with dim = %ld, size = %ld.", hcMix,
                biasShape.GetDimNum(), biasShape.GetDimNum() > 0 ? biasShape.GetDim(0) : 0);
        return false;
    }

    auto hPostShape = hPost->GetViewShape();
    if (isTnd) {
        if (hPostShape.GetDimNum() != 2 || hPostShape.GetDim(0) != seqLen || hPostShape.GetDim(1) != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost shape must be (%ld, %ld), but got (%ld, %ld).", seqLen, n,
                    hPostShape.GetDim(0), hPostShape.GetDim(1));
            return false;
        }
    } else {
        if (hPostShape.GetDimNum() != 3 || hPostShape.GetDim(0) != bs || hPostShape.GetDim(1) != seqLen ||
            hPostShape.GetDim(2) != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost shape must be (%ld, %ld, %ld), but got (%ld, %ld, %ld).", bs,
                    seqLen, n, hPostShape.GetDim(0), hPostShape.GetDim(1), hPostShape.GetDim(2));
            return false;
        }
    }

    auto hResShape = hRes->GetViewShape();
    if (isTnd) {
        if (hResShape.GetDimNum() != 2 || hResShape.GetDim(0) != seqLen || hResShape.GetDim(1) != n * n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hRes shape must be (%ld, %ld), but got (%ld, %ld).", seqLen, n * n,
                    hResShape.GetDim(0), hResShape.GetDim(1));
            return false;
        }
    } else {
        if (hResShape.GetDimNum() != 3 || hResShape.GetDim(0) != bs || hResShape.GetDim(1) != seqLen ||
            hResShape.GetDim(2) != n * n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hRes shape must be (%ld, %ld, %ld), but got (%ld, %ld, %ld).", bs, seqLen,
                    n * n, hResShape.GetDim(0), hResShape.GetDim(1), hResShape.GetDim(2));
            return false;
        }
    }

    if (needBackward) {
        auto hPreShape = hPre->GetViewShape();
        if (isTnd) {
            if (hPreShape.GetDimNum() != 2 || hPreShape.GetDim(0) != seqLen || hPreShape.GetDim(1) != n) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre shape must be (%ld, %ld), but got (%ld, %ld).", seqLen, n,
                        hPreShape.GetDim(0), hPreShape.GetDim(1));
                return false;
            }
        } else {
            if (hPreShape.GetDimNum() != 3 || hPreShape.GetDim(0) != bs || hPreShape.GetDim(1) != seqLen ||
                hPreShape.GetDim(2) != n) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre shape must be (%ld, %ld, %ld), but got (%ld, %ld, %ld).", bs,
                        seqLen, n, hPreShape.GetDim(0), hPreShape.GetDim(1), hPreShape.GetDim(2));
                return false;
            }
        }

        auto hcBeforeNormShape = hcBeforeNorm->GetViewShape();
        if (isTnd) {
            if (hcBeforeNormShape.GetDimNum() != 2 || hcBeforeNormShape.GetDim(0) != seqLen ||
                hcBeforeNormShape.GetDim(1) != hcMix) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcBeforeNorm shape must be (%ld, %ld), but got (%ld, %ld).", seqLen,
                        hcMix, hcBeforeNormShape.GetDim(0), hcBeforeNormShape.GetDim(1));
                return false;
            }
        } else {
            if (hcBeforeNormShape.GetDimNum() != 3 || hcBeforeNormShape.GetDim(0) != bs ||
                hcBeforeNormShape.GetDim(1) != seqLen || hcBeforeNormShape.GetDim(2) != hcMix) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcBeforeNorm shape must be (%ld, %ld, %ld), but got (%ld, %ld, %ld).",
                        bs, seqLen, hcMix, hcBeforeNormShape.GetDim(0), hcBeforeNormShape.GetDim(1),
                        hcBeforeNormShape.GetDim(2));
                return false;
            }
        }

        auto invRmsShape = invRms->GetViewShape();
        if (isTnd) {
            if (invRmsShape.GetDimNum() != 2 || invRmsShape.GetDim(0) != seqLen || invRmsShape.GetDim(1) != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms shape must be (%ld, 1), but got (%ld, %ld).", seqLen,
                        invRmsShape.GetDim(0), invRmsShape.GetDim(1));
                return false;
            }
        } else {
            if (invRmsShape.GetDimNum() != 3 || invRmsShape.GetDim(0) != bs || invRmsShape.GetDim(1) != seqLen ||
                invRmsShape.GetDim(2) != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms shape must be (%ld, %ld, 1), but got (%ld, %ld, %ld).", bs,
                        seqLen, invRmsShape.GetDim(0), invRmsShape.GetDim(1), invRmsShape.GetDim(2));
                return false;
            }
        }

        auto sumOutShape = sumOut->GetViewShape();
        int64_t expectedSumOutDim0 = numIters * 2;
        if (isTnd) {
            if (sumOutShape.GetDimNum() != 3 || sumOutShape.GetDim(0) != expectedSumOutDim0 ||
                sumOutShape.GetDim(1) != seqLen || sumOutShape.GetDim(2) != n) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sumOut shape must be (%ld, %ld, %ld), but got (%ld, %ld, %ld).",
                        expectedSumOutDim0, seqLen, n, sumOutShape.GetDim(0), sumOutShape.GetDim(1),
                        sumOutShape.GetDim(2));
                return false;
            }
        } else {
            if (sumOutShape.GetDimNum() != 4 || sumOutShape.GetDim(0) != expectedSumOutDim0 ||
                sumOutShape.GetDim(1) != bs || sumOutShape.GetDim(2) != seqLen || sumOutShape.GetDim(3) != n) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "sumOut shape must be (%ld, %ld, %ld, %ld), but got (%ld, %ld, %ld, %ld).", expectedSumOutDim0,
                        bs, seqLen, n, sumOutShape.GetDim(0), sumOutShape.GetDim(1), sumOutShape.GetDim(2),
                        sumOutShape.GetDim(3));
                return false;
            }
        }

        auto normOutShape = normOut->GetViewShape();
        int64_t expectedNormOutDim0 = numIters * 2;
        if (isTnd) {
            if (normOutShape.GetDimNum() != 4 || normOutShape.GetDim(0) != expectedNormOutDim0 ||
                normOutShape.GetDim(1) != seqLen || normOutShape.GetDim(2) != n || normOutShape.GetDim(3) != n) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "normOut shape must be (%ld, %ld, %ld, %ld), but got (%ld, %ld, %ld, %ld).",
                        expectedNormOutDim0, seqLen, n, n, normOutShape.GetDim(0), normOutShape.GetDim(1),
                        normOutShape.GetDim(2), normOutShape.GetDim(3));
                return false;
            }
        } else {
            if (normOutShape.GetDimNum() != 5 || normOutShape.GetDim(0) != expectedNormOutDim0 ||
                normOutShape.GetDim(1) != bs || normOutShape.GetDim(2) != seqLen || normOutShape.GetDim(3) != n ||
                normOutShape.GetDim(4) != n) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "normOut shape must be (%ld, %ld, %ld, %ld, %ld), but got (%ld, %ld, %ld, %ld, %ld).",
                        expectedNormOutDim0, bs, seqLen, n, n, normOutShape.GetDim(0), normOutShape.GetDim(1),
                        normOutShape.GetDim(2), normOutShape.GetDim(3), normOutShape.GetDim(4));
                return false;
            }
        }
    }

    return true;
}

static bool IsFormatSupport(const aclTensor *tensor, const std::string &tensorName)
{
    if (tensor != nullptr && op::IsPrivateFormat(tensor->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s's format should be ND, but got %s.", tensorName.c_str(),
                op::ToString(tensor->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                        const aclTensor *hin, const aclTensor *hPost, const aclTensor *hRes, const aclTensor *hPre,
                        const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut,
                        const aclTensor *normOut, bool needBackward)
{
    CHECK_RET(IsFormatSupport(x, "x"), false);
    CHECK_RET(IsFormatSupport(phi, "phi"), false);
    CHECK_RET(IsFormatSupport(alpha, "alpha"), false);
    CHECK_RET(IsFormatSupport(bias, "bias"), false);
    CHECK_RET(IsFormatSupport(hin, "hin"), false);
    CHECK_RET(IsFormatSupport(hPost, "hPost"), false);
    CHECK_RET(IsFormatSupport(hRes, "hRes"), false);

    if (needBackward) {
        CHECK_RET(IsFormatSupport(hPre, "hPre"), false);
        CHECK_RET(IsFormatSupport(hcBeforeNorm, "hcBeforeNorm"), false);
        CHECK_RET(IsFormatSupport(invRms, "invRms"), false);
        CHECK_RET(IsFormatSupport(sumOut, "sumOut"), false);
        CHECK_RET(IsFormatSupport(normOut, "normOut"), false);
    }

    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                      const aclTensor *bias, const aclTensor *hin, const aclTensor *hPost,
                                      const aclTensor *hRes, const aclTensor *hPre, const aclTensor *hcBeforeNorm,
                                      const aclTensor *invRms, const aclTensor *sumOut, const aclTensor *normOut,
                                      int64_t hcMult, int64_t numIters, bool needBackward)
{
    CHECK_RET(CheckNotNull(x, phi, alpha, bias, hin, hPost, hRes), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, phi, alpha, bias, hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut,
                              needBackward),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, phi, alpha, bias, hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, hcMult,
                         numIters, needBackward),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckFormat(x, phi, alpha, bias, hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, needBackward),
        ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                                const aclTensor *bias, int64_t hcMult, int64_t numIters, double hcEps,
                                                double normEps, bool needBackward, aclTensor *hin, aclTensor *hPost,
                                                aclTensor *hRes, aclTensor *hPre, aclTensor *hcBeforeNorm,
                                                aclTensor *invRms, aclTensor *sumOut, aclTensor *normOut,
                                                uint64_t *workspaceSize, aclOpExecutor **executor)

{
    L2_DFX_PHASE_1(aclnnMhcPreSinkhorn, DFX_IN(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, needBackward),
                   DFX_OUT(hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, phi, alpha, bias, hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, hcMult,
                           numIters, needBackward);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // Check if input tensors are empty
    if (x->IsEmpty() || hin->IsEmpty() || hPost->IsEmpty() || hRes->IsEmpty()) {
        OP_LOGW("[aclnnMhcPreSinkhorn] Input tensor is empty, skip computation and return success.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *phiContiguous = l0op::Contiguous(phi, uniqueExecutor.get());
    CHECK_RET(phiContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *alphaContiguous = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_RET(alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut = l0op::MhcPreSinkhorn(
        xContiguous, phiContiguous, alphaContiguous, biasContiguous, hcMult, numIters, hcEps, normEps, needBackward,
        hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhorn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
