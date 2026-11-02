/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_chunk_kda_fwd.h"
#include "../../../op_host/op_api/chunk_kda_fwd.h"
#include "stub/chunk_kda_launcher_test_hook.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kSeqLen = 64;
constexpr int64_t kHeadNum = 2;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kChunkSize = 64;

struct AclTensorDeleter {
    void operator()(aclTensor* value) const
    {
        if (value != nullptr) {
            (void)aclDestroyTensor(value);
        }
    }
};

using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;

AclTensorPtr MakeTensor(std::initializer_list<int64_t> shape, aclDataType dtype)
{
    auto tensor = TensorDesc(std::vector<int64_t>(shape), dtype, ACL_FORMAT_ND).ToAclType();
    return AclTensorPtr(tensor.release(), AclTensorDeleter{});
}

aclnnStatus RunWorkspaceCase(const std::string& layout, int64_t chunkSize, bool includeALog = true)
{
    auto q = TensorDesc({kBatch, kSeqLen, kHeadNum, kHeadDim}, ACL_BF16, ACL_FORMAT_ND);
    auto k = TensorDesc({kBatch, kSeqLen, kHeadNum, kHeadDim}, ACL_BF16, ACL_FORMAT_ND);
    auto v = TensorDesc({kBatch, kSeqLen, kHeadNum, kHeadDim}, ACL_BF16, ACL_FORMAT_ND);
    auto g = TensorDesc({kBatch, kSeqLen, kHeadNum, kHeadDim}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta = TensorDesc({kBatch, kSeqLen, kHeadNum}, ACL_FLOAT, ACL_FORMAT_ND);
    auto aLog = TensorDesc({kHeadNum}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dtBias = TensorDesc({kHeadNum * kHeadDim}, ACL_FLOAT, ACL_FORMAT_ND);
    auto attn = TensorDesc({kBatch, kSeqLen, kHeadNum, kHeadDim}, ACL_BF16, ACL_FORMAT_ND);
    auto aqk = TensorDesc({kBatch, kHeadNum, kSeqLen, chunkSize}, ACL_BF16, ACL_FORMAT_ND);
    auto akk = TensorDesc({kBatch, kHeadNum, kSeqLen, chunkSize}, ACL_BF16, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    if (includeALog) {
        auto ut =
            OP_API_UT(aclnnChunkKdaFwd,
                      INPUT(q, k, v, g, beta, aLog, dtBias, nullptr, nullptr, nullptr, layout.c_str(), 1.0 / 16.0,
                            chunkSize, true, -5.0, true, false),
                      OUTPUT(attn, nullptr, nullptr, aqk, akk, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }
    auto ut = OP_API_UT(aclnnChunkKdaFwd,
                        INPUT(q, k, v, g, beta, nullptr, dtBias, nullptr, nullptr, nullptr, layout.c_str(), 1.0 / 16.0,
                              chunkSize, true, -5.0, true, false),
                        OUTPUT(attn, nullptr, nullptr, aqk, akk, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

std::vector<ErrorManager::ErrorItem> DrainErrorItems()
{
    return ErrorManager::GetInstance().GetRawErrorMessages();
}

bool HasErrorCode(const std::vector<ErrorManager::ErrorItem>& items, const std::string& code)
{
    return std::any_of(items.begin(), items.end(), [&code](const auto& item) { return item.error_id == code; });
}

using RealConvertToTensor = const aclTensor* (*)(aclOpExecutor*, const aclIntArray*, op::DataType);

RealConvertToTensor GetRealConvertToTensor()
{
    static RealConvertToTensor realConvert = []() {
        void* handle = dlopen("libnnopbase.so", RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            return static_cast<RealConvertToTensor>(nullptr);
        }
        return reinterpret_cast<RealConvertToTensor>(
            dlsym(handle, "_ZN13aclOpExecutor15ConvertToTensorEPK11aclIntArrayN2ge8DataTypeE"));
    }();
    return realConvert;
}

class ChunkKdaFwdErrorCodeTest : public testing::Test {
protected:
    void SetUp() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        chunk_kda_ut::SetNextLauncherStatus(ACLNN_SUCCESS);
        (void)DrainErrorItems();
    }

    void TearDown() override
    {
        chunk_kda_ut::ClearLauncherStatus();
        chunk_kda_ut::ClearConvertToTensorFailCall();
        (void)DrainErrorItems();
    }
};
} // namespace

const aclTensor* aclOpExecutor::ConvertToTensor(const aclIntArray* value, op::DataType dataType)
{
    if (chunk_kda_ut::ShouldFailConvertToTensor()) {
        OP_LOGE(ACLNN_ERR_INNER, "aclOpExecutor::ConvertToTensor fail.");
        return nullptr;
    }
    static const RealConvertToTensor realConvert = GetRealConvertToTensor();
    if (realConvert == nullptr) {
        return nullptr;
    }
    return realConvert(this, value, dataType);
}

TEST_F(ChunkKdaFwdErrorCodeTest, invalid_layout_reports_param_error)
{
    EXPECT_EQ(RunWorkspaceCase("INVALID", kChunkSize), ACLNN_ERR_PARAM_INVALID);
    const auto errors = DrainErrorItems();
    EXPECT_TRUE(HasErrorCode(errors, "EZ1001"));
    EXPECT_FALSE(HasErrorCode(errors, "EZ9999"));
}

TEST_F(ChunkKdaFwdErrorCodeTest, invalid_chunk_size_reports_param_error)
{
    EXPECT_EQ(RunWorkspaceCase("BSND", 32), ACLNN_ERR_PARAM_INVALID);
    const auto errors = DrainErrorItems();
    EXPECT_TRUE(HasErrorCode(errors, "EZ1001"));
    EXPECT_FALSE(HasErrorCode(errors, "EZ9999"));
}

TEST_F(ChunkKdaFwdErrorCodeTest, missing_gate_input_returns_param_invalid)
{
    EXPECT_EQ(RunWorkspaceCase("BSND", kChunkSize, false), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ChunkKdaFwdErrorCodeTest, launcher_failure_returns_null_outputs_without_error_report)
{
    chunk_kda_ut::SetNextLauncherStatus(ACLNN_ERR_PARAM_NULLPTR);
    auto q = MakeTensor({1}, ACL_BF16);
    auto k = MakeTensor({1}, ACL_BF16);
    auto v = MakeTensor({1}, ACL_BF16);
    auto g = MakeTensor({1}, ACL_FLOAT);
    auto beta = MakeTensor({1}, ACL_FLOAT);
    auto attn = MakeTensor({1}, ACL_BF16);
    auto aqk = MakeTensor({1}, ACL_BF16);
    auto akk = MakeTensor({1}, ACL_BF16);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);

    const auto result =
        l0op::KdaChunkForward(q.get(), k.get(), v.get(), g.get(), beta.get(), nullptr, nullptr, nullptr, nullptr,
                              nullptr, 1.0, 64, false, true, false, -5.0, attn.get(), nullptr, nullptr, aqk.get(),
                              akk.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, executor.get());

    if (chunk_kda_ut::HasPendingLauncherStatus()) {
        GTEST_SKIP() << "Launcher test hook is inactive for the linked KdaChunkForward implementation.";
    }
    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](const aclTensor* tensor) { return tensor == nullptr; }));
    EXPECT_TRUE(DrainErrorItems().empty());
}

TEST_F(ChunkKdaFwdErrorCodeTest, launcher_success_preserves_output_aliases)
{
    auto q = MakeTensor({1}, ACL_BF16);
    auto k = MakeTensor({1}, ACL_BF16);
    auto v = MakeTensor({1}, ACL_BF16);
    auto g = MakeTensor({1}, ACL_FLOAT);
    auto beta = MakeTensor({1}, ACL_FLOAT);
    std::array<AclTensorPtr, 11> outputs;
    for (auto& output : outputs) {
        output = MakeTensor({1}, ACL_FLOAT);
    }
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);

    const auto result =
        l0op::KdaChunkForward(q.get(), k.get(), v.get(), g.get(), beta.get(), nullptr, nullptr, nullptr, nullptr,
                              nullptr, 1.0, 64, false, true, false, -5.0, outputs[0].get(), outputs[1].get(),
                              outputs[2].get(), outputs[3].get(), outputs[4].get(), outputs[5].get(), outputs[6].get(),
                              outputs[7].get(), outputs[8].get(), outputs[9].get(), outputs[10].get(), executor.get());

    for (size_t index = 0; index < outputs.size(); ++index) {
        EXPECT_EQ(result[index], outputs[index].get());
    }
    EXPECT_TRUE(DrainErrorItems().empty());
}

TEST_F(ChunkKdaFwdErrorCodeTest, metadata_conversion_success_preserves_output_aliases)
{
    auto q = MakeTensor({1}, ACL_BF16);
    auto k = MakeTensor({1}, ACL_BF16);
    auto v = MakeTensor({1}, ACL_BF16);
    auto g = MakeTensor({1}, ACL_FLOAT);
    auto beta = MakeTensor({1}, ACL_FLOAT);
    auto attn = MakeTensor({1}, ACL_BF16);
    auto aqk = MakeTensor({1}, ACL_BF16);
    auto akk = MakeTensor({1}, ACL_BF16);
    int64_t cuSeqlensData[] = {0, kSeqLen};
    int64_t chunkIndicesData[] = {0, 0};
    auto cuSeqlens = aclCreateIntArray(cuSeqlensData, 2);
    auto chunkIndices = aclCreateIntArray(chunkIndicesData, 2);
    ASSERT_NE(cuSeqlens, nullptr);
    ASSERT_NE(chunkIndices, nullptr);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);

    const auto result =
        l0op::KdaChunkForward(q.get(), k.get(), v.get(), g.get(), beta.get(), nullptr, nullptr, nullptr, cuSeqlens,
                              chunkIndices, 1.0, 64, false, true, false, -5.0, attn.get(), nullptr, nullptr, aqk.get(),
                              akk.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, executor.get());

    EXPECT_EQ(result[0], attn.get());
    EXPECT_EQ(result[3], aqk.get());
    EXPECT_EQ(result[4], akk.get());
    EXPECT_TRUE(DrainErrorItems().empty());
    aclDestroyIntArray(cuSeqlens);
    aclDestroyIntArray(chunkIndices);
}

TEST_F(ChunkKdaFwdErrorCodeTest, cu_seqlens_conversion_failure_preserves_lower_error)
{
    auto q = MakeTensor({1}, ACL_BF16);
    auto k = MakeTensor({1}, ACL_BF16);
    auto v = MakeTensor({1}, ACL_BF16);
    auto g = MakeTensor({1}, ACL_FLOAT);
    auto beta = MakeTensor({1}, ACL_FLOAT);
    auto attn = MakeTensor({1}, ACL_BF16);
    auto aqk = MakeTensor({1}, ACL_BF16);
    auto akk = MakeTensor({1}, ACL_BF16);
    int64_t cuSeqlensData[] = {0, kSeqLen};
    auto cuSeqlens = aclCreateIntArray(cuSeqlensData, 2);
    ASSERT_NE(cuSeqlens, nullptr);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);
    chunk_kda_ut::SetConvertToTensorFailCall(1);

    const auto result =
        l0op::KdaChunkForward(q.get(), k.get(), v.get(), g.get(), beta.get(), nullptr, nullptr, nullptr, cuSeqlens,
                              nullptr, 1.0, 64, false, true, false, -5.0, attn.get(), nullptr, nullptr, aqk.get(),
                              akk.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, executor.get());

    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](const aclTensor* tensor) { return tensor == nullptr; }));
    const auto errors = DrainErrorItems();
    EXPECT_EQ(std::count_if(errors.begin(), errors.end(), [](const auto& item) { return item.error_id == "EZ9999"; }),
              1);
    EXPECT_FALSE(HasErrorCode(errors, "EZ1001"));
    aclDestroyIntArray(cuSeqlens);
}

TEST_F(ChunkKdaFwdErrorCodeTest, chunk_indices_conversion_failure_preserves_lower_error)
{
    auto q = MakeTensor({1}, ACL_BF16);
    auto k = MakeTensor({1}, ACL_BF16);
    auto v = MakeTensor({1}, ACL_BF16);
    auto g = MakeTensor({1}, ACL_FLOAT);
    auto beta = MakeTensor({1}, ACL_FLOAT);
    auto attn = MakeTensor({1}, ACL_BF16);
    auto aqk = MakeTensor({1}, ACL_BF16);
    auto akk = MakeTensor({1}, ACL_BF16);
    int64_t cuSeqlensData[] = {0, kSeqLen};
    int64_t chunkIndicesData[] = {0, 0};
    auto cuSeqlens = aclCreateIntArray(cuSeqlensData, 2);
    auto chunkIndices = aclCreateIntArray(chunkIndicesData, 2);
    ASSERT_NE(cuSeqlens, nullptr);
    ASSERT_NE(chunkIndices, nullptr);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);
    chunk_kda_ut::SetConvertToTensorFailCall(2);

    const auto result =
        l0op::KdaChunkForward(q.get(), k.get(), v.get(), g.get(), beta.get(), nullptr, nullptr, nullptr, cuSeqlens,
                              chunkIndices, 1.0, 64, false, true, false, -5.0, attn.get(), nullptr, nullptr, aqk.get(),
                              akk.get(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, executor.get());

    EXPECT_TRUE(std::all_of(result.begin(), result.end(), [](const aclTensor* tensor) { return tensor == nullptr; }));
    const auto errors = DrainErrorItems();
    EXPECT_EQ(std::count_if(errors.begin(), errors.end(), [](const auto& item) { return item.error_id == "EZ9999"; }),
              1);
    EXPECT_FALSE(HasErrorCode(errors, "EZ1001"));
    aclDestroyIntArray(cuSeqlens);
    aclDestroyIntArray(chunkIndices);
}

TEST_F(ChunkKdaFwdErrorCodeTest, phase2_null_executor_does_not_add_internal_error)
{
    const auto ret = aclnnChunkKdaFwd(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_ERR_INNER);
    const auto errors = DrainErrorItems();
    EXPECT_TRUE(HasErrorCode(errors, "EZ1001"));
    EXPECT_EQ(std::count_if(errors.begin(), errors.end(), [](const auto& item) { return item.error_id == "EZ1001"; }),
              1);
    EXPECT_NE(std::find_if(errors.begin(), errors.end(),
                           [](const auto& item) { return item.error_message.find("executor") != std::string::npos; }),
              errors.end());
    EXPECT_FALSE(HasErrorCode(errors, "EZ9999"));
    EXPECT_FALSE(HasErrorCode(errors, "EZ9903"));
}
