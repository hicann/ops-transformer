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
 * \file aclnn_ffn_param.cpp
 * \brief FFN Aclnn 参数信息.
 */

#include "aclnn_ffn_param.h"
#include <utility>
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/tests_utils_log.h"

using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;

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

using namespace ops::adv::tests::ffn;

AclnnFFNParam::AclnnFFNParam(std::vector<Tensor> inputs, std::vector<int64_t> expertTokensData, std::string activation,
                             int32_t innerPrecise, int32_t outputDtype, FunctionType functionType,
                             AclnnFFNVersion aclnnFFNVersion, bool tokensIndexFlag)
    : Param(std::move(inputs), std::move(expertTokensData), activation, innerPrecise, outputDtype, tokensIndexFlag),
      mFunctionType(functionType), mAclnnFFNVersion(aclnnFFNVersion)
{
}


AclnnFFNParam::~AclnnFFNParam()
{
    if (aclnnExpertTokensIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnExpertTokensIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclnnExpertTokensIntAry failed, ERROR: %d", ret),
                    aclnnExpertTokensIntAry = nullptr);
    }
}

bool AclnnFFNParam::Init()
{
    aclnnX = ops::adv::tests::utils::AclnnTensor(mTensors["x"]);
    aclnnWeight1 = ops::adv::tests::utils::AclnnTensor(mTensors["weight1"]);
    aclnnWeight2 = ops::adv::tests::utils::AclnnTensor(mTensors["weight2"]);
    auto iter = mTensors.find("bias1");
    if (iter != mTensors.end()) {
        aclnnBias1 = ops::adv::tests::utils::AclnnTensor(mTensors["bias1"]);
    }
    iter = mTensors.find("bias2");
    if (iter != mTensors.end()) {
        aclnnBias2 = ops::adv::tests::utils::AclnnTensor(mTensors["bias2"]);
    }
    aclnnY = ops::adv::tests::utils::AclnnTensor(mTensors["y"]);

    if (mFunctionType == FunctionType::QUANT) {
        aclnnScale = ops::adv::tests::utils::AclnnTensor(mTensors["scale"]);
        aclnnOffset = ops::adv::tests::utils::AclnnTensor(mTensors["offset"]);
        aclnnDeqScale1 = ops::adv::tests::utils::AclnnTensor(mTensors["deqScale1"]);
        aclnnDeqScale2 = ops::adv::tests::utils::AclnnTensor(mTensors["deqScale2"]);
    } else if (mFunctionType == FunctionType::ANTIQUANT) {
        aclnnAntiquantScale1 = ops::adv::tests::utils::AclnnTensor(mTensors["antiquant_scale1"]);
        aclnnAntiquantScale2 = ops::adv::tests::utils::AclnnTensor(mTensors["antiquant_scale2"]);
        aclnnAntiquantOffset1 = ops::adv::tests::utils::AclnnTensor(mTensors["antiquant_offset1"]);
        aclnnAntiquantOffset2 = ops::adv::tests::utils::AclnnTensor(mTensors["antiquant_offset2"]);
    }
    auto ret = InitExpertTokens();
    LOG_IF_EXPR(ret == false, LOG_ERR("InitExpertTokens faild"), return false);

    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);

    for (auto *t :
         {&aclnnX, &aclnnWeight1, &aclnnWeight2, &aclnnBias1, &aclnnBias2, &aclnnScale, &aclnnOffset, &aclnnDeqScale1,
          &aclnnDeqScale2, &aclnnAntiquantScale1, &aclnnAntiquantScale2, &aclnnAntiquantOffset1, &aclnnAntiquantOffset1,
          &aclnnAntiquantOffset2, &aclnnY, &aclnnExpertTokens}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return false;
        }
        std::string filePath = std::string(cs->GetRootPath()) + t->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!t->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    return true;
}

bool AclnnFFNParam::InitExpertTokens()
{
    auto iter = mTensors.find("expertTokens");
    if (iter == mTensors.end()) {
        return true;
    }

    if (mAclnnFFNVersion == AclnnFFNVersion::V3) {
        if (mExpertTokensData.size() > 0) {
            size_t dataSize = mExpertTokensData.size() * sizeof(int64_t);
            std::string fileName = "expertToken.bin";
            if (!WriteFile(fileName, mExpertTokensData.data(), dataSize)) {
                LOG_ERR("Write expertToken data to file[%s] failed", fileName.c_str());
                return false;
            }
        }
        aclnnExpertTokens = ops::adv::tests::utils::AclnnTensor(mTensors["expertTokens"]);
    } else if (!InitAclIntArray(&aclnnExpertTokensIntAry, mExpertTokensData)) {
        return false;
    }

    return true;
}