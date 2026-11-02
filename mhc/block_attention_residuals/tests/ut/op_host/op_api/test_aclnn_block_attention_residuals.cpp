/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../../../../op_host/op_api/aclnn_block_attention_residuals.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class BlockAttentionResidualsOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "BlockAttentionResidualsOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "BlockAttentionResidualsOpapiUt TearDown" << endl;
    }
};

namespace {
constexpr int64_t TOKEN_NUM = 2;
constexpr int64_t BLOCK_NUM = 2;
constexpr int64_t HIDDEN_SIZE = 16;
constexpr int64_t BACKWARD_WIDTH = BLOCK_NUM + 1;
constexpr int64_t VALID_BLOCK_NUM = -1;
constexpr double NORM_EPS = 1e-6;

using TensorDescPtr = std::unique_ptr<TensorDesc>;

TensorDescPtr MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype)
{
    TensorDescPtr tensor(new TensorDesc(shape, dtype, ACL_FORMAT_ND));
    tensor->ValueRange(-1, 1);
    return tensor;
}

struct ResidualsTensors {
    TensorDescPtr partialBlock;
    TensorDescPtr blockRes;
    TensorDescPtr projWeight;
    TensorDescPtr normWeight;
    TensorDescPtr hiddenStates;
    TensorDescPtr invNorm;
    TensorDescPtr probs;
};

ResidualsTensors MakeValidTensors()
{
    ResidualsTensors tensors;
    tensors.partialBlock = MakeTensor({TOKEN_NUM, HIDDEN_SIZE}, ACL_FLOAT16);
    tensors.blockRes = MakeTensor({TOKEN_NUM, BLOCK_NUM, HIDDEN_SIZE}, ACL_FLOAT16);
    tensors.projWeight = MakeTensor({HIDDEN_SIZE}, ACL_FLOAT16);
    tensors.normWeight = MakeTensor({HIDDEN_SIZE}, ACL_FLOAT16);
    tensors.hiddenStates = MakeTensor({TOKEN_NUM, HIDDEN_SIZE}, ACL_FLOAT16);
    tensors.invNorm = MakeTensor({TOKEN_NUM, BACKWARD_WIDTH}, ACL_FLOAT);
    tensors.probs = MakeTensor({TOKEN_NUM, BACKWARD_WIDTH}, ACL_FLOAT);
    return tensors;
}

enum class MissingTensor {
    None,
    PartialBlock,
    BlockRes,
    ProjWeight,
    NormWeight,
    HiddenStates,
    InvNorm,
    Probs
};

aclnnStatus RunGetWorkspaceSize(ResidualsTensors& tensors, MissingTensor missing, bool needBackward)
{
    auto partialBlock = tensors.partialBlock->ToAclType();
    auto blockRes = tensors.blockRes->ToAclType();
    auto projWeight = tensors.projWeight->ToAclType();
    auto normWeight = tensors.normWeight->ToAclType();
    auto hiddenStates = tensors.hiddenStates->ToAclType();
    auto invNorm = tensors.invNorm->ToAclType();
    auto probs = tensors.probs->ToAclType();
    auto pick = [missing](MissingTensor current, auto* tensor) { return missing == current ? nullptr : tensor; };
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnBlockAttentionResidualsGetWorkspaceSize(
        pick(MissingTensor::PartialBlock, partialBlock.get()), pick(MissingTensor::BlockRes, blockRes.get()),
        pick(MissingTensor::ProjWeight, projWeight.get()), pick(MissingTensor::NormWeight, normWeight.get()),
        VALID_BLOCK_NUM, NORM_EPS, needBackward, pick(MissingTensor::HiddenStates, hiddenStates.get()),
        pick(MissingTensor::InvNorm, invNorm.get()), pick(MissingTensor::Probs, probs.get()), &workspaceSize,
        &executor);
}
} // namespace

TEST_F(BlockAttentionResidualsOpapiUt, aclnn_block_attention_residuals_missing_required_tensor_returns_nullptr)
{
    const std::vector<std::pair<const char*, MissingTensor>> cases = {
        {"partialBlock", MissingTensor::PartialBlock}, {"blockRes", MissingTensor::BlockRes},
        {"projWeight", MissingTensor::ProjWeight},     {"normWeight", MissingTensor::NormWeight},
        {"hiddenStates", MissingTensor::HiddenStates},
    };
    for (const auto& [name, missing] : cases) {
        auto tensors = MakeValidTensors();
        const auto actual = RunGetWorkspaceSize(tensors, missing, false);
        EXPECT_EQ(actual, ACLNN_ERR_PARAM_NULLPTR) << name;
    }
}

TEST_F(BlockAttentionResidualsOpapiUt, aclnn_block_attention_residuals_missing_backward_output_returns_nullptr)
{
    const std::vector<std::pair<const char*, MissingTensor>> cases = {
        {"invNorm", MissingTensor::InvNorm},
        {"probs", MissingTensor::Probs},
    };
    for (const auto& [name, missing] : cases) {
        auto tensors = MakeValidTensors();
        const auto actual = RunGetWorkspaceSize(tensors, missing, true);
        EXPECT_EQ(actual, ACLNN_ERR_PARAM_NULLPTR) << name;
    }
}
