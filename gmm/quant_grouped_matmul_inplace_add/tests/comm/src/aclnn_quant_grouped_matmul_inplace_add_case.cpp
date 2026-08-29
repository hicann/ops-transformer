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
 * \file aclnn_quant_grouped_matmul_inplace_add_case.cpp
 * \brief QGMMInplaceAdd Aclnn 测试用例.
 */

#include <utility>
#include "tests/utils/tests_utils_log.h"
#include "aclnn_quant_grouped_matmul_inplace_add.h"
#include "aclnn_quant_grouped_matmul_inplace_add_case.h"

using namespace ops::adv::tests::quant_grouped_matmul_inplace_add;

namespace {
bool QGMMInplaceAddTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnQGMMInplaceAddCase *>(curCase);
    auto *aclnnParam = &cs->mAclnnParam;

    aclnnStatus ret = ACL_SUCCESS;
    ret = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(
            aclnnParam->x1, aclnnParam->aclnnX2.GetAclTensor(),
            aclnnParam->scale1, aclnnParam->aclnnScale2.GetAclTensor(),
            aclnnParam->aclnnGroupList.GetAclTensor(), aclnnParam->aclnnY.GetAclTensor(),
            aclnnParam->mGroupListType, aclnnParam->mGroupSize, workSpaceSize, opExecutor);
    return ret == ACL_SUCCESS;
}

bool QGMMInplaceAddKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnQGMMInplaceAddCase *>(curCase);
    auto *aclnnCtx = &cs->mAclnnCtx;

    aclnnStatus ret = ACL_SUCCESS;
    ret = aclnnQuantGroupedMatmulInplaceAdd(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                                 aclnnCtx->GetAclRtStream());
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnQGMMInplaceAdd failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}
}

AclnnQGMMInplaceAddCase::AclnnQGMMInplaceAddCase()
    : QuantGroupedMatmulInplaceAddCase(), mAclnnCtx(AclnnContext()), mAclnnParam(AclnnQGMMInplaceAddParam())
{
}

AclnnQGMMInplaceAddCase::AclnnQGMMInplaceAddCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo,
    AclnnQGMMInplaceAddParam aclnnParam, int32_t tilingTemplatePriority)
    : QuantGroupedMatmulInplaceAddCase(name, enable, dbgInfo, std::move(opInfo), Param(), tilingTemplatePriority),
      mAclnnParam(std::move(aclnnParam))
{
}

bool AclnnQGMMInplaceAddCase::InitParam()
{
    return mAclnnParam.Init();
}

bool AclnnQGMMInplaceAddCase::InitOpInfo()
{
    if (!QuantGroupedMatmulInplaceAddCase::InitOpInfo()) {
        return false;
    }

    auto rst = mAclnnCtx.SetOpName(this->mOpInfo.mName.c_str());
    rst = rst && mAclnnCtx.SetTilingRunCbf(QGMMInplaceAddTilingRunCbf);
    rst = rst && mAclnnCtx.SetKernelRunCbf(QGMMInplaceAddKernelRunCbf);
    rst = rst && mOpInfo.SetContext(&mAclnnCtx);
    return rst;
}

bool AclnnQGMMInplaceAddCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool AclnnQGMMInplaceAddCase::Run()
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
