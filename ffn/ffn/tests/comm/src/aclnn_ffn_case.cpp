/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_ffn_case.cpp
 * \brief FFN Aclnn 测试用例.
 */

#include <utility>
#include "tests/utils/tests_utils_log.h"
#include "aclnn_ffn.h"
#include "aclnn_ffn_v2.h"
#include "aclnn_ffn_v3.h"
#include "aclnn_ffn_case.h"

using namespace ops::adv::tests::ffn;

bool FFNTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnFFNCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnParam->mAclnnFFNVersion == AclnnFFNVersion::V1) {
        ret = aclnnFFNGetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight1.GetAclTensor(),
            aclnnParam->aclnnWeight2.GetAclTensor(), aclnnParam->aclnnExpertTokensIntAry,
            aclnnParam->aclnnBias1.GetAclTensor(), aclnnParam->aclnnBias2.GetAclTensor(),
            aclnnParam->aclnnScale.GetAclTensor(), aclnnParam->aclnnOffset.GetAclTensor(),
            aclnnParam->aclnnDeqScale1.GetAclTensor(), aclnnParam->aclnnDeqScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantScale1.GetAclTensor(), aclnnParam->aclnnAntiquantScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantOffset1.GetAclTensor(), aclnnParam->aclnnAntiquantOffset2.GetAclTensor(),
            aclnnParam->mActivation.c_str(), aclnnParam->mInnerPrecise, aclnnParam->aclnnY.GetAclTensor(),
            workSpaceSize, opExecutor);
    } else if (aclnnParam->mAclnnFFNVersion == AclnnFFNVersion::V2) {
        ret = aclnnFFNV2GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight1.GetAclTensor(),
            aclnnParam->aclnnWeight2.GetAclTensor(), aclnnParam->aclnnExpertTokensIntAry,
            aclnnParam->aclnnBias1.GetAclTensor(), aclnnParam->aclnnBias2.GetAclTensor(),
            aclnnParam->aclnnScale.GetAclTensor(), aclnnParam->aclnnOffset.GetAclTensor(),
            aclnnParam->aclnnDeqScale1.GetAclTensor(), aclnnParam->aclnnDeqScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantScale1.GetAclTensor(), aclnnParam->aclnnAntiquantScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantOffset1.GetAclTensor(), aclnnParam->aclnnAntiquantOffset2.GetAclTensor(),
            aclnnParam->mActivation.c_str(), aclnnParam->mInnerPrecise, aclnnParam->mTokensIndexFlag,
            aclnnParam->aclnnY.GetAclTensor(), workSpaceSize, opExecutor);
    } else if (aclnnParam->mAclnnFFNVersion == AclnnFFNVersion::V3) {
        ret = aclnnFFNV3GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight1.GetAclTensor(),
            aclnnParam->aclnnWeight2.GetAclTensor(), aclnnParam->aclnnExpertTokens.GetAclTensor(),
            aclnnParam->aclnnBias1.GetAclTensor(), aclnnParam->aclnnBias2.GetAclTensor(),
            aclnnParam->aclnnScale.GetAclTensor(), aclnnParam->aclnnOffset.GetAclTensor(),
            aclnnParam->aclnnDeqScale1.GetAclTensor(), aclnnParam->aclnnDeqScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantScale1.GetAclTensor(), aclnnParam->aclnnAntiquantScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantOffset1.GetAclTensor(), aclnnParam->aclnnAntiquantOffset2.GetAclTensor(),
            aclnnParam->mActivation.c_str(), aclnnParam->mInnerPrecise, aclnnParam->mTokensIndexFlag,
            aclnnParam->aclnnY.GetAclTensor(), workSpaceSize, opExecutor);
    }

    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFFNGetWorkspaceSize failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

bool FFNKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnFFNCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;
    auto *aclnnCtx = &cs->mAclnnCtx;

    aclnnStatus ret = ACL_SUCCESS;
    if (aclnnParam->mAclnnFFNVersion == AclnnFFNVersion::V1) {
        ret = aclnnFFN(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                       aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->mAclnnFFNVersion == AclnnFFNVersion::V2) {
        ret = aclnnFFNV2(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                         aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->mAclnnFFNVersion == AclnnFFNVersion::V3) {
        ret = aclnnFFNV3(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                         aclnnCtx->GetAclRtStream());
    }
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFFN failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

AclnnFFNCase::AclnnFFNCase() : FFNCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnFFNParam())
{
}

AclnnFFNCase::AclnnFFNCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, AclnnFFNParam aclnnParam,
                           int32_t tilingTemplatePriority)
    : FFNCase(name, enable, dbgInfo, std::move(opInfo), Param(), tilingTemplatePriority),
      mAclnnParam(std::move(aclnnParam))
{
}

bool AclnnFFNCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnFFNCase::InitOpInfo()
{
    if (!FFNCase::InitOpInfo()) {
        return false;
    }

    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(FFNTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(FFNKernelRunCbf);
    rst = rst && mAclnnCtx.SetOutputs({&mAclnnParam.aclnnY});
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnFFNCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool AclnnFFNCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
