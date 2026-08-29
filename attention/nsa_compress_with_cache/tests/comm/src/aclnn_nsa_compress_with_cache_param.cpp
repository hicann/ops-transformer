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
 * \file aclnn_nsa_compress_with_cache_param.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 参数信息.
 */

#include "aclnn_nsa_compress_with_cache_param.h"
#include <utility>
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/tests_utils_log.h"

namespace {
template <class T> bool InitAclIntArray(aclIntArray **intArray, std::vector<T> &hostData)
{
    if (intArray == nullptr) {
        LOG_ERR("intArray nil.");
        return false;
    }
    if (*intArray != nullptr) {
        auto ret = aclDestroyIntArray(*intArray);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret), *intArray = nullptr);
    }
    if (hostData.empty()) {
        return true;
    }
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
    if (*intArray == nullptr) {
        LOG_ERR("aclCreateIntArray failed.");
        return false;
    }
    return true;
}
} // namespace

using namespace ops::adv::tests::NsaCompressWithCache;

AclnnNsaCompressWithCacheParam::AclnnNsaCompressWithCacheParam(int64_t batchSize, int64_t headNum, int64_t headDim,
                                                               ge::DataType dtype, std::vector<int64_t> actSeqLenList,
                                                               LayoutType layoutType, int64_t compressBlockSize,
                                                               int64_t compressStride, int64_t actSeqLenType,
                                                               int64_t pageBlockSize)
    : NsaCompressWithCacheParam(batchSize, headNum, headDim, dtype, std::move(actSeqLenList), layoutType,
                                compressBlockSize, compressStride, actSeqLenType, pageBlockSize),
      aclnnActualSeqLenIntAry(nullptr)
{
}

AclnnNsaCompressWithCacheParam::~AclnnNsaCompressWithCacheParam()
{
    if (aclnnActualSeqLenIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnActualSeqLenIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret),
                    aclnnActualSeqLenIntAry = nullptr);
    }
}

bool AclnnNsaCompressWithCacheParam::Init()
{
    if (!NsaCompressWithCacheParam::Init()) {
        return false;
    }
    aclnnInput = ops::adv::tests::utils::AclnnTensor(mInput);
    aclnnWeight = ops::adv::tests::utils::AclnnTensor(mWeight);
    aclnnActSeqLenOptional = ops::adv::tests::utils::AclnnTensor(mActSeqLenOptional);
    aclnnOutputCache = ops::adv::tests::utils::AclnnTensor(mOutputCache);
    aclnnSlotMapping = ops::adv::tests::utils::AclnnTensor(mSlotMapping);
    aclnnBlockTable = ops::adv::tests::utils::AclnnTensor(mBlockTableOptional);

    if (!InitAclIntArray(&aclnnActualSeqLenIntAry, mActSeqLenList)) {
        return false;
    }

    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);

    for (auto *it : {&aclnnInput, &aclnnWeight, &aclnnOutputCache, &aclnnSlotMapping, &aclnnBlockTable}) {
        it->FreeDevData();
        if (it->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = it->AllocDevData(0, 0);
        if (devData == nullptr) {
            return false;
        }
        if (it->IsOutput()) {
            continue;
        }
        std::string filePath = std::string(cs->GetRootPath()) + it->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!it->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    return true;
}
