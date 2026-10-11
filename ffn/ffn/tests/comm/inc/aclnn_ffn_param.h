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
 * \file aclnn_ffn_param.h
 * \brief FFN Aclnn 参数信息.
 */

#ifndef UTEST_ACLNN_FFN_PARAM_H
#define UTEST_ACLNN_FFN_PARAM_H

#include "ffn_case.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::ffn {

class AclnnFFNParam : public ops::adv::tests::ffn::Param {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    enum class FunctionType {
        NO_QUANT,
        QUANT,
        ANTIQUANT,
    };

    enum class AclnnFFNVersion {
        V1,
        V2,
        V3,
    };

public:
    FunctionType mFunctionType = FunctionType::NO_QUANT;
    AclnnFFNVersion mAclnnFFNVersion = AclnnFFNVersion::V1;
    /* 输入输出 */
    AclnnTensor aclnnX, aclnnWeight1, aclnnWeight2, aclnnExpertTokens, aclnnBias1, aclnnBias2, aclnnScale, aclnnOffset,
        aclnnDeqScale1, aclnnDeqScale2, aclnnAntiquantScale1, aclnnAntiquantScale2, aclnnAntiquantOffset1,
        aclnnAntiquantOffset2, aclnnY;
    aclIntArray* aclnnExpertTokensIntAry = nullptr;

public:
    AclnnFFNParam() = default;
    AclnnFFNParam(std::vector<Tensor> inputs, std::vector<int64_t> expertTokensData, std::string activation,
                  int32_t innerPrecise, int32_t outputDtype, FunctionType functionType, AclnnFFNVersion aclnnFFNVersion,
                  bool tokensIndexFlag = false);

    ~AclnnFFNParam();

    bool Init();

private:
    bool InitExpertTokens();
};

} // namespace ops::adv::tests::ffn
#endif // UTEST_ACLNN_FFN_PARAM_H
