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
 * \file aclnn_nsa_compress_with_cache_case.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 测试用例.
 */

#pragma once

#include "nsa_compress_with_cache_case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/aclnn_context.h"
#include "aclnn_nsa_compress_with_cache_param.h"

namespace ops::adv::tests::NsaCompressWithCache {

class AclnnNsaCompressWithCacheCase : public ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheCase {
public:
    using AclnnContext = ops::adv::tests::utils::AclnnContext;
    using AclnnNsaCompressWithCacheParam = ops::adv::tests::NsaCompressWithCache::AclnnNsaCompressWithCacheParam;

public:
    /* 算子控制信息 */
    AclnnContext mAclnnCtx;

    /* 输入/输出 参数 */
    AclnnNsaCompressWithCacheParam mAclnnParam;

public:
    AclnnNsaCompressWithCacheCase();
    AclnnNsaCompressWithCacheCase(const char* name, bool enable, const char* dbgInfo, OpInfo opInfo,
                                  const AclnnNsaCompressWithCacheParam& param,
                                  int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    bool Run() override;

protected:
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::NsaCompressWithCache
