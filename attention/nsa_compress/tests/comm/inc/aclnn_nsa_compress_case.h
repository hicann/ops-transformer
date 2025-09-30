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
 * \file aclnn_nsa_compress_case.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 测试用例.
 */

#ifndef UTEST_ACLNN_NSA_COMPRESS_CASE_H
#define UTEST_ACLNN_NSA_COMPRESS_CASE_H

#include "nsa_compress_case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/aclnn_context.h"
#include "aclnn_nsa_compress_param.h"

namespace ops::adv::tests::NsaCompress {

class AclnnNsaCompressCase : public ops::adv::tests::NsaCompress::NsaCompressCase {
public:
    using AclnnContext = ops::adv::tests::utils::AclnnContext;
    using AclnnNsaCompressParam = ops::adv::tests::NsaCompress::AclnnNsaCompressParam;

public:
    /* 算子控制信息 */
    AclnnContext mAclnnCtx;

    /* 输入/输出 参数 */
    AclnnNsaCompressParam mAclnnParam;

public:
    AclnnNsaCompressCase();
    AclnnNsaCompressCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo,
                         const AclnnNsaCompressParam &param,
                         int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    bool Run() override;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};


} // namespace ops::adv::tests::NsaCompress
#endif // UTEST_ACLNN_NSA_COMPRESS_PARAM_H