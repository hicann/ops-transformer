/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fa_case.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 测试用例.
 */

#include <utility>
#include "aclnn_fa_case.h"
#include "tests/utils/log.h"
#include "aclnn_flash_attention_score.h"
#include "aclnn_flash_attention_score_grad.h"

using namespace ops::adv::tests::fa;

bool FasTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *exp = &cs->mForward.mExp;

    aclnnStatus ret;
    if (mAclnnParam->pseType == 1) {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnPse.GetAclTensor(),
                mAclnnParam->aclnnDropMask.GetAclTensor(), mAclnnParam->aclnnPaddingMask.GetAclTensor(),
                mAclnnParam->aclnnAttenMask.GetAclTensor(), mAclnnParam->aclnnPrefixIntAry, mAclnnParam->scale,
                mAclnnParam->keepProb, mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->aclnnSoftmaxMax.GetAclTensor(), mAclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxRes.GetAclTensor(), mAclnnParam->aclnnAttenRes.GetAclTensor(), workSpaceSize,
                opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionScoreGetWorkspaceSize failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnPse.GetAclTensor(),
                mAclnnParam->aclnnDropMask.GetAclTensor(), mAclnnParam->aclnnPaddingMask.GetAclTensor(),
                mAclnnParam->aclnnAttenMask.GetAclTensor(), mAclnnParam->aclnnPrefixIntAry,
                mAclnnParam->aclnnActualSeqQLenIntAry, mAclnnParam->aclnnActualSeqKvLenIntAry, mAclnnParam->scale,
                mAclnnParam->keepProb, mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->aclnnSoftmaxMax.GetAclTensor(), mAclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxRes.GetAclTensor(), mAclnnParam->aclnnAttenRes.GetAclTensor(), workSpaceSize,
                opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionVarLenScoreGetWorkspaceSize failed, ERROR: %d", ret));
        }
    } else {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnPse.GetAclTensor(),
                mAclnnParam->aclnnDropMask.GetAclTensor(), mAclnnParam->aclnnPaddingMask.GetAclTensor(),
                mAclnnParam->aclnnAttenMask.GetAclTensor(), mAclnnParam->aclnnPrefixIntAry,
                mAclnnParam->qStartIdxOptionalIntAry, mAclnnParam->kvStartIdxOptionalIntAry, mAclnnParam->scale,
                mAclnnParam->keepProb, mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->pseType, mAclnnParam->aclnnSoftmaxMax.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxSum.GetAclTensor(), mAclnnParam->aclnnSoftmaxRes.GetAclTensor(),
                mAclnnParam->aclnnAttenRes.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionScoreV2GetWorkspaceSize failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnPse.GetAclTensor(),
                mAclnnParam->aclnnDropMask.GetAclTensor(), mAclnnParam->aclnnPaddingMask.GetAclTensor(),
                mAclnnParam->aclnnAttenMask.GetAclTensor(), mAclnnParam->aclnnPrefixIntAry,
                mAclnnParam->aclnnActualSeqQLenIntAry, mAclnnParam->aclnnActualSeqKvLenIntAry,
                mAclnnParam->qStartIdxOptionalIntAry, mAclnnParam->kvStartIdxOptionalIntAry, mAclnnParam->scale,
                mAclnnParam->keepProb, mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->pseType, mAclnnParam->aclnnSoftmaxMax.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxSum.GetAclTensor(), mAclnnParam->aclnnSoftmaxRes.GetAclTensor(),
                mAclnnParam->aclnnAttenRes.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize failed, ERROR: %d", ret));
        }
    }
    return ret == ACL_SUCCESS;
}

bool FasKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *ctx = &cs->mAclnnForwardCtx;

    aclnnStatus ret;
    if (mAclnnParam->pseType == 1) {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScore(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                           ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScore failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionVarLenScore(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                 ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionVarLenScore failed, ERROR: %d", ret));
        }
    } else {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreV2(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                             ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScoreV2 failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionVarLenScoreV2(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                   ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionVarLenScoreV2 failed, ERROR: %d", ret));
        }
    }
    return ret == ACL_SUCCESS;
}

bool FagTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *exp = &cs->mReverse.mExp;

    aclnnStatus ret;
    if (mAclnnParam->pseType == 1) {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGradGetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnDy.GetAclTensor(),
                mAclnnParam->aclnnPse.GetAclTensor(), mAclnnParam->aclnnDropMask.GetAclTensor(),
                mAclnnParam->aclnnPaddingMask.GetAclTensor(), mAclnnParam->aclnnAttenMask.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxMax.GetAclTensor(), mAclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxRes.GetAclTensor(), mAclnnParam->aclnnAttenRes.GetAclTensor(),
                mAclnnParam->aclnnPrefixIntAry, mAclnnParam->scale, mAclnnParam->keepProb, mAclnnParam->preTokens,
                mAclnnParam->nxtTokens, mAclnnParam->n1, const_cast<char *>(mAclnnParam->layout.c_str()),
                mAclnnParam->innerPrecise, mAclnnParam->sparseMode, mAclnnParam->aclnnDq.GetAclTensor(),
                mAclnnParam->aclnnDk.GetAclTensor(), mAclnnParam->aclnnDv.GetAclTensor(),
                mAclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionScoreGradGetWorkspaceSize failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnDy.GetAclTensor(),
                mAclnnParam->aclnnPse.GetAclTensor(), mAclnnParam->aclnnDropMask.GetAclTensor(),
                mAclnnParam->aclnnPaddingMask.GetAclTensor(), mAclnnParam->aclnnAttenMask.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxMax.GetAclTensor(), mAclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxRes.GetAclTensor(), mAclnnParam->aclnnAttenRes.GetAclTensor(),
                mAclnnParam->aclnnPrefixIntAry, mAclnnParam->aclnnActualSeqQLenIntAry,
                mAclnnParam->aclnnActualSeqKvLenIntAry, mAclnnParam->scale, mAclnnParam->keepProb,
                mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->aclnnDq.GetAclTensor(), mAclnnParam->aclnnDk.GetAclTensor(),
                mAclnnParam->aclnnDv.GetAclTensor(), mAclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize failed, ERROR: %d", ret));
        }
    } else {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnDy.GetAclTensor(),
                mAclnnParam->aclnnPse.GetAclTensor(), mAclnnParam->aclnnDropMask.GetAclTensor(),
                mAclnnParam->aclnnPaddingMask.GetAclTensor(), mAclnnParam->aclnnAttenMask.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxMax.GetAclTensor(), mAclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxRes.GetAclTensor(), mAclnnParam->aclnnAttenRes.GetAclTensor(),
                mAclnnParam->aclnnPrefixIntAry, mAclnnParam->qStartIdxOptionalIntAry,
                mAclnnParam->kvStartIdxOptionalIntAry, mAclnnParam->scale, mAclnnParam->keepProb,
                mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->pseType, mAclnnParam->aclnnDq.GetAclTensor(), mAclnnParam->aclnnDk.GetAclTensor(),
                mAclnnParam->aclnnDv.GetAclTensor(), mAclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionScoreGradV2GetWorkspaceSize failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
                mAclnnParam->aclnnQuery.GetAclTensor(), mAclnnParam->aclnnKey.GetAclTensor(),
                mAclnnParam->aclnnValue.GetAclTensor(), mAclnnParam->aclnnDy.GetAclTensor(),
                mAclnnParam->aclnnPse.GetAclTensor(), mAclnnParam->aclnnDropMask.GetAclTensor(),
                mAclnnParam->aclnnPaddingMask.GetAclTensor(), mAclnnParam->aclnnAttenMask.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxMax.GetAclTensor(), mAclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                mAclnnParam->aclnnSoftmaxRes.GetAclTensor(), mAclnnParam->aclnnAttenRes.GetAclTensor(),
                mAclnnParam->aclnnPrefixIntAry, mAclnnParam->aclnnActualSeqQLenIntAry,
                mAclnnParam->aclnnActualSeqKvLenIntAry, mAclnnParam->qStartIdxOptionalIntAry,
                mAclnnParam->kvStartIdxOptionalIntAry, mAclnnParam->scale, mAclnnParam->keepProb,
                mAclnnParam->preTokens, mAclnnParam->nxtTokens, mAclnnParam->n1,
                const_cast<char *>(mAclnnParam->layout.c_str()), mAclnnParam->innerPrecise, mAclnnParam->sparseMode,
                mAclnnParam->pseType, mAclnnParam->aclnnDq.GetAclTensor(), mAclnnParam->aclnnDk.GetAclTensor(),
                mAclnnParam->aclnnDv.GetAclTensor(), mAclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
                   LOG_ERR("aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize failed, ERROR: %d", ret));
        }
    }
    return ret == ACL_SUCCESS;
}

bool FagKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *ctx = &cs->mAclnnReverseCtx;

    aclnnStatus ret;
    if (mAclnnParam->pseType == 1) {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGrad(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                               ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScoreGrad failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGrad(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                        ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionUnpaddingScoreGrad failed, ERROR: %d", ret));
        }
    } else {
        if (!mAclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGradV2(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                 ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScoreGradV2 failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGradV2(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                          ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionUnpaddingScoreGradV2 failed, ERROR: %d", ret));
        }
    }
    return ret == ACL_SUCCESS;
}

AclnnFaCase::AclnnFaCase()
    : FaCase(), mAclnnForwardCtx(AclnnContext()), mAclnnReverseCtx(AclnnContext()), mAclnnParam(AclnnFaParam())
{
}

AclnnFaCase::AclnnFaCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion forward, OpInfoWithSocversion reverse,
                         const AclnnFaParam &param, int32_t tilingTemplatePriority)
    : FaCase(name, enable, dbgInfo, std::move(forward), std::move(reverse), FaParam(), tilingTemplatePriority),
      mAclnnParam(param)
{
}

bool AclnnFaCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnFaCase::InitOpInfo()
{
    if (!FaCase::InitOpInfo()) {
        return false;
    }
    auto rst = mAclnnForwardCtx.SetOpName(this->mForward.mName.c_str());
    rst = rst && mAclnnForwardCtx.SetTilingRunCbf(FasTilingRunCbf);
    rst = rst && mAclnnForwardCtx.SetKernelRunCbf(FasKernelRunCbf);
    rst = rst && mAclnnForwardCtx.SetOutputs({&mAclnnParam.aclnnSoftmaxMax, &mAclnnParam.aclnnSoftmaxSum,
                                              &mAclnnParam.aclnnSoftmaxRes, &mAclnnParam.aclnnAttenRes});
    rst = rst && mForward.SetContext(&mAclnnForwardCtx);
    rst = rst && mAclnnReverseCtx.SetOpName(this->mReverse.mName.c_str());
    rst = rst && mAclnnReverseCtx.SetTilingRunCbf(FagTilingRunCbf);
    rst = rst && mAclnnReverseCtx.SetKernelRunCbf(FagKernelRunCbf);
    rst = rst && mAclnnReverseCtx.SetOutputs(
                     {&mAclnnParam.aclnnDq, &mAclnnParam.aclnnDk, &mAclnnParam.aclnnDv, &mAclnnParam.aclnnDpse});
    rst = rst && mReverse.SetContext(&mAclnnReverseCtx);
    return rst;
}

bool AclnnFaCase::InitCurrentCasePtr()
{
    CaseWithSocversion::mCurrentCasePtr = this;
    return true;
}
