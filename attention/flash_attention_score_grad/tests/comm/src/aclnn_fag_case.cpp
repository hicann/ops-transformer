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
 * \file aclnn_fag_case.cpp
 * \brief
 */

#include "aclnn_fag_case.h"
#include <utility>

using namespace ops::adv::tests::fag;

AclnnFagCase::AclnnFagCase() : AclnnFagCase("Undefined", true, "", OpInfoWithSocversion(), AclnnFaParam())
{
}

AclnnFagCase::AclnnFagCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion reverse, AclnnFaParam param)
    : AclnnFaCase(name, enable, dbgInfo, OpInfoWithSocversion(), std::move(reverse), std::move(param),
                  kTilingTemplatePriority_Invalid)
{
}

bool AclnnFagCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mReverse.ProcessTiling(mName, this->socVersion)) {
        return false;
    }
    if (!mReverse.ProcessKernel(mName)) {
        return false;
    }
    return true;
}
