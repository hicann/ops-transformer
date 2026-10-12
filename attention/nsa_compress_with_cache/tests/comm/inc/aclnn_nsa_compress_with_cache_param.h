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
 * \file aclnn_nsa_compress_with_cache_param.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 参数信息.
 */

#pragma once

#include "nsa_compress_with_cache_case.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::NsaCompressWithCache {

class AclnnNsaCompressWithCacheParam : public ops::adv::tests::NsaCompressWithCache::NsaCompressWithCacheParam {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    /* 输入输出 */
    AclnnTensor aclnnInput, aclnnWeight, aclnnActSeqLenOptional, aclnnOutputCache, aclnnSlotMapping, aclnnBlockTable;
    aclIntArray* aclnnActualSeqLenIntAry = nullptr;

public:
    AclnnNsaCompressWithCacheParam() = default;
    AclnnNsaCompressWithCacheParam(int64_t batchSize, int64_t headNum, int64_t headDim, ge::DataType dtype,
                                   std::vector<int64_t> actSeqLenList, LayoutType layoutType, int64_t compressBlockSize,
                                   int64_t compressStride, int64_t actSeqLenType, int64_t pageBlockSize);
    ~AclnnNsaCompressWithCacheParam();

    bool Init() override;
};

} // namespace ops::adv::tests::NsaCompressWithCache
