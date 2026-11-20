/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <string>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "opdev/op_errno.h"
#include "../../../op_host/op_api/aclnn_nsa_compress_with_cache.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class NsaCompressWithCacheApiTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressWithCacheApiTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressWithCacheApiTest TearDown" << std::endl;
    }
};

TEST_F(NsaCompressWithCacheApiTest, aclnn_nsa_compress_with_cache_A1_fp16)
{
    constexpr int64_t batchSize = 4;
    constexpr int64_t pageBlockSize = 128;
    constexpr int64_t maxSeqLen = 512;
    constexpr int64_t blockNumPerBatch = maxSeqLen / pageBlockSize;
    constexpr int64_t blocksNum = blockNumPerBatch * batchSize;
    constexpr int64_t headsNum = 24;
    constexpr int64_t headsDim = 192;
    constexpr int64_t compressBlockSize = 32;
    constexpr int64_t compressStride = 16;

    auto input =
        TensorDesc({blocksNum, pageBlockSize, headsNum, headsDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight = TensorDesc({compressBlockSize, headsNum}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto slotMapping = TensorDesc({batchSize}, ACL_INT32, ACL_FORMAT_ND);
    IntArrayDesc actSeqLen(std::vector<int64_t>{32, 0, 0, 0});
    auto blockTable = TensorDesc({batchSize, blockNumPerBatch}, ACL_INT32, ACL_FORMAT_ND);
    char layout[] = "TND";
    auto outputCache = TensorDesc({maxSeqLen, headsNum, headsDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompressWithCache,
                        INPUT(input, weight, slotMapping, actSeqLen, blockTable, layout, compressBlockSize,
                              compressStride, 1, pageBlockSize),
                        OUTPUT(outputCache));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// compressStride<=0 拦截在 host tiling（见 op_host tiling UT），aclnn GetWorkspaceSize
// 仅做空指针/dtype/format/contiguous 校验，不在此阶段断言 stride 非法（issue 7713）。

namespace {
enum class NullParam {
    Input,
    Weight,
    SlotMapping,
    OutputCache,
    WorkspaceSize,
    Executor
};

class NsaCompressWithCacheNullptrTest : public testing::TestWithParam<NullParam> {};

TEST_P(NsaCompressWithCacheNullptrTest, ReturnsParamNullptr)
{
    auto input = TensorDesc({1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto weight = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto slotMapping = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto outputCache = TensorDesc({1, 128, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    ASSERT_NE(input.get(), nullptr);
    ASSERT_NE(weight.get(), nullptr);
    ASSERT_NE(slotMapping.get(), nullptr);
    ASSERT_NE(outputCache.get(), nullptr);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    char layout[] = "BSND";
    const auto param = GetParam();

    // Keep every other required pointer valid to test each API argument independently.
    const auto ret = aclnnNsaCompressWithCacheGetWorkspaceSize(
        param == NullParam::Input ? nullptr : input.get(), param == NullParam::Weight ? nullptr : weight.get(),
        param == NullParam::SlotMapping ? nullptr : slotMapping.get(), nullptr, nullptr, layout, 16, 16, 0, 128,
        param == NullParam::OutputCache ? nullptr : outputCache.get(),
        param == NullParam::WorkspaceSize ? nullptr : &workspaceSize,
        param == NullParam::Executor ? nullptr : &executor);

    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(workspaceSize, 0U);
    EXPECT_EQ(executor, nullptr);
}

std::string NullParamName(const testing::TestParamInfo<NullParam>& info)
{
    switch (info.param) {
        case NullParam::Input:
            return "Input";
        case NullParam::Weight:
            return "Weight";
        case NullParam::SlotMapping:
            return "SlotMapping";
        case NullParam::OutputCache:
            return "OutputCache";
        case NullParam::WorkspaceSize:
            return "WorkspaceSize";
        case NullParam::Executor:
            return "Executor";
    }
    return "Unknown";
}

INSTANTIATE_TEST_SUITE_P(RequiredPointers, NsaCompressWithCacheNullptrTest,
                         testing::Values(NullParam::Input, NullParam::Weight, NullParam::SlotMapping,
                                         NullParam::OutputCache, NullParam::WorkspaceSize, NullParam::Executor),
                         NullParamName);

TEST(NsaCompressWithCacheParamTest, EmptyTensorRemainsParamInvalid)
{
    auto input = TensorDesc({0, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto weight = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto slotMapping = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto outputCache = TensorDesc({1, 128, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    ASSERT_NE(input.get(), nullptr);
    ASSERT_NE(weight.get(), nullptr);
    ASSERT_NE(slotMapping.get(), nullptr);
    ASSERT_NE(outputCache.get(), nullptr);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    char layout[] = "BSND";
    // Optional nulls must not be mistaken for missing required arguments.
    EXPECT_EQ(
        aclnnNsaCompressWithCacheGetWorkspaceSize(input.get(), weight.get(), slotMapping.get(), nullptr, nullptr,
                                                  layout, 16, 16, 0, 128, outputCache.get(), &workspaceSize, &executor),
        ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}
} // namespace
