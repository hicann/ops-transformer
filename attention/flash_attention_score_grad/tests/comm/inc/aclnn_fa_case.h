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
 * \file aclnn_fa_case.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 测试用例.
 */

#pragma once

#include "fa_case.h"
#include "tests/utils/op_info_with_socversion.h"
#include "tests/utils/aclnn_context.h"
#include "aclnn_fa_param.h"

namespace ops::adv::tests::fa {

class AclnnFaCase : public ops::adv::tests::fa::FaCase {
public:
    using AclnnContext = ops::adv::tests::utils::AclnnContext;
    using AclnnFaParam = ops::adv::tests::fa::AclnnFaParam;

public:
    /* 算子控制信息 */
    AclnnContext mAclnnForwardCtx;
    AclnnContext mAclnnReverseCtx;

    /* 输入/输出 参数 */
    AclnnFaParam mAclnnParam;

public:
    AclnnFaCase();
    AclnnFaCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion forward, OpInfoWithSocversion reverse,
                const AclnnFaParam &param, int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::fa
