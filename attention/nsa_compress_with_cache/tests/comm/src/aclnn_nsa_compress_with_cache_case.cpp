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
 * \file aclnn_nsa_compress_with_cache_case.cpp
 * \brief NsaCompressWithCache Aclnn 测试用例.
 */

#include "aclnn_nsa_compress_with_cache_case.h"
#include <utility>
#include "tests/utils/tests_utils_log.h"
#include "aclnn_nsa_compress_with_cache.h"

using namespace ops::adv::tests::NsaCompressWithCache;

bool NsaCompressWithCacheTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnNsaCompressWithCacheCase *>(curCase);
    auto *mAclnnParam = &cs->mAclnnParam;
    auto *exp = &cs->mOpInfo.mExp;

    aclnnStatus ret;

    ret = aclnnNsaCompressWithCacheGetWorkspaceSize(
        mAclnnParam->aclnnInput.GetAclTensor(), mAclnnParam->aclnnWeight.GetAclTensor(),
        mAclnnParam->aclnnSlotMapping.GetAclTensor(), mAclnnParam->aclnnActualSeqLenIntAry,
        mAclnnParam->aclnnBlockTable.GetAclTensor(), const_cast<char *>(mAclnnParam->mLayoutOptional.c_str()),
        mAclnnParam->mCompressBlockSize, mAclnnParam->mCompressStride, mAclnnParam->mActSeqLenType,
        mAclnnParam->mPageBlockSize, mAclnnParam->aclnnOutputCache.GetAclTensor(), workSpaceSize, opExecutor);
    LOG_IF(ret != ACL_SUCCESS && exp->mSuccess,
           LOG_ERR("aclnnNsaCompressWithCacheGetWorkspaceSize failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

bool NsaCompressWithCacheKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnNsaCompressWithCacheCase *>(curCase);
    // auto *mAclnnParam = &cs->mAclnnParam;
    auto *ctx = &cs->mAclnnCtx;

    aclnnStatus ret;

    ret = aclnnNsaCompressWithCache(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                    ctx->GetAclRtStream());
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnNsaCompressWithCache failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}


AclnnNsaCompressWithCacheCase::AclnnNsaCompressWithCacheCase()
    : NsaCompressWithCacheCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnNsaCompressWithCacheParam())
{
}

AclnnNsaCompressWithCacheCase::AclnnNsaCompressWithCacheCase(const char *name, bool enable, const char *dbgInfo,
                                                             OpInfo opInfo, const AclnnNsaCompressWithCacheParam &param,
                                                             int32_t tilingTemplatePriority)
    : NsaCompressWithCacheCase(name, enable, dbgInfo, std::move(opInfo), NsaCompressWithCacheParam(),
                               tilingTemplatePriority),
      mAclnnParam(param)
{
}

bool AclnnNsaCompressWithCacheCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnNsaCompressWithCacheCase::InitOpInfo()
{
    if (!NsaCompressWithCacheCase::InitOpInfo()) {
        return false;
    }
    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(NsaCompressWithCacheTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(NsaCompressWithCacheKernelRunCbf);
    rst = rst && mAclnnCtx.SetOutputs({&mAclnnParam.aclnnOutputCache});
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnNsaCompressWithCacheCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}


bool AclnnNsaCompressWithCacheCase::Run()
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
