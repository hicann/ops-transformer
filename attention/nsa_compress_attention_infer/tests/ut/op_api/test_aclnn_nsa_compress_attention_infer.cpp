/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_nsa_compress_attention_infer.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/op_errno.h"

namespace {
TEST(NsaCompressAttentionInferApiTest, RejectsMismatchedValueDtypeAtWorkspaceQuery)
{
    auto query = TensorDesc({1, 8, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto key = TensorDesc({4, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto value = TensorDesc({4, 128, 64}, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    auto blockTable = TensorDesc({1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto cmpSeqLengths = IntArrayDesc(std::vector<int64_t>{128}).ToAclType();
    auto output = TensorDesc({1, 8, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto topKOutput = TensorDesc({1, 1, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    ASSERT_TRUE(query && key && value && blockTable && cmpSeqLengths && output && topKOutput);

    auto& errorManager = ErrorManager::GetInstance();
    (void)errorManager.GetRawErrorMessages();
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnNsaCompressAttentionInferGetWorkspaceSize(
        query.get(), key.get(), value.get(), nullptr, blockTable.get(), nullptr, cmpSeqLengths.get(), nullptr, nullptr,
        8, 1, 64, 1, 32, 16, 1.0, layout, 128, 0, output.get(), topKOutput.get(), &workspaceSize, &executor);

    EXPECT_EQ(status, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
    const auto errors = errorManager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, "EZ1001");
    EXPECT_NE(errors.front().error_message.find("Input K or V data type is invalid"), std::string::npos);
}

TEST(NsaCompressAttentionInferApiTest, ChecksTndBatchSizeIndependentlyOfPackedTokenCount)
{
    const struct {
        int64_t batchSize;
        int64_t tokens;
        int64_t querySeqLen;
        aclnnStatus expectedStatus;
    } cases[] = {
        {2500, 10000, 4, ACLNN_SUCCESS},    {2501, 10004, 4, ACLNN_SUCCESS},
        {10000, 40000, 4, ACLNN_SUCCESS},   {10001, 10001, 1, ACLNN_ERR_PARAM_INVALID},
        {0, 1, 1, ACLNN_ERR_PARAM_INVALID},
    };

    auto key = TensorDesc({1, 128, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto value = TensorDesc({1, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    ASSERT_TRUE(key && value);
    for (const auto& testCase : cases) {
        const int64_t tokens = testCase.tokens;
        SCOPED_TRACE(testing::Message() << "B=" << testCase.batchSize << ", T=" << tokens);
        auto query = TensorDesc({tokens, 1, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
        auto blockTable = TensorDesc({testCase.batchSize, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        auto queryLengths = IntArrayDesc(std::vector<int64_t>(testCase.batchSize, testCase.querySeqLen)).ToAclType();
        auto cmpLengths = IntArrayDesc(std::vector<int64_t>(testCase.batchSize, 3)).ToAclType();
        auto selLengths = IntArrayDesc(std::vector<int64_t>(testCase.batchSize, 64)).ToAclType();
        auto output = TensorDesc({tokens, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
        auto topKOutput = TensorDesc({tokens, 1, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        ASSERT_TRUE(query && blockTable && queryLengths && cmpLengths && selLengths && output && topKOutput);

        char layout[] = "TND";
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        const auto status = aclnnNsaCompressAttentionInferGetWorkspaceSize(
            query.get(), key.get(), value.get(), nullptr, blockTable.get(), queryLengths.get(), cmpLengths.get(),
            selLengths.get(), nullptr, 1, 1, 64, 1, 32, 16, 0.088388, layout, 128, 0, output.get(), topKOutput.get(),
            &workspaceSize, &executor);
        EXPECT_EQ(status, testCase.expectedStatus);
        EXPECT_EQ(executor != nullptr, testCase.expectedStatus == ACLNN_SUCCESS);
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }
}
} // namespace
