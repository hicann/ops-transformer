/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_dequant_rope_quant_kvcache.h"
#include "../error_code_test_utils.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/data_type_utils.h"
#include "opdev/platform.h"
#include "aclnnInner_dequant_rope_quant_kvcache.h"

// Process-global state defined by the API UT platform stub.
namespace op {
extern SocVersion g_socVersion;
extern NpuArch g_npuArchOverride;
} // namespace op

namespace {
using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor*)>;
constexpr const char* API_NAME = "aclnnDequantRopeQuantKvcacheGetWorkspaceSize";

class PlatformStateGuard {
public:
    PlatformStateGuard()
        : socVersion_(op::g_socVersion),
          npuArchOverride_(op::g_npuArchOverride)
    {}

    ~PlatformStateGuard()
    {
        op::g_socVersion = socVersion_;
        op::g_npuArchOverride = npuArchOverride_;
    }

    PlatformStateGuard(const PlatformStateGuard&) = delete;
    PlatformStateGuard& operator=(const PlatformStateGuard&) = delete;

private:
    op::SocVersion socVersion_;
    NpuArch npuArchOverride_;
};

class DequantRopeQuantKvcacheApi : public testing::Test {
protected:
    void SetUp() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        drqk_test::ClearErrorMessages();
    }

    void TearDown() override
    {
        drqk_test::ClearErrorMessages();
    }

private:
    PlatformStateGuard platformState_;
};

struct ApiInputs {
    TensorPtr x;
    TensorPtr cos;
    TensorPtr sin;
    TensorPtr kCache = TensorDesc({1, 1024, 1, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclType();
    TensorPtr vCache = TensorDesc({1, 1024, 1, 128}, ACL_INT8, ACL_FORMAT_ND).ToAclType();
    TensorPtr indices = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    TensorPtr scale = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    TensorPtr weightScale = TensorDesc({1280}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    TensorPtr q;
    TensorPtr k;
    TensorPtr v;
    std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> splits;

    ApiInputs(aclDataType xType, aclDataType cosType, aclDataType sinType)
        : x(TensorDesc({1, 320, 1280}, xType, ACL_FORMAT_ND).ToAclType()),
          cos(TensorDesc({1, 320, 1, 128}, cosType, ACL_FORMAT_ND).ToAclType()),
          sin(TensorDesc({1, 320, 1, 128}, sinType, ACL_FORMAT_ND).ToAclType()),
          q(TensorDesc({1, 320, 8, 128}, cosType, ACL_FORMAT_ND).ToAclType()),
          k(TensorDesc({1, 320, 1, 128}, cosType, ACL_FORMAT_ND).ToAclType()),
          v(TensorDesc({1, 320, 1, 128}, cosType, ACL_FORMAT_ND).ToAclType()),
          splits(nullptr, aclDestroyIntArray)
    {
        const int64_t sizes[] = {1024, 128, 128};
        splits.reset(aclCreateIntArray(sizes, 3));
    }

    aclnnStatus Run(const aclTensor* xArg, const aclTensor* cosArg, const aclTensor* sinArg, uint64_t* workspaceSize,
                    aclOpExecutor** executor)
    {
        char quantMode[] = "static";
        char layout[] = "BSND";
        char cacheMode[] = "contiguous";
        return aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
            xArg, cosArg, sinArg, kCache.get(), vCache.get(), indices.get(), scale.get(), scale.get(), nullptr, nullptr,
            x->GetDataType() == op::DataType::DT_INT32 ? weightScale.get() : nullptr, nullptr, nullptr, splits.get(),
            quantMode, layout, true, cacheMode, q.get(), k.get(), v.get(), workspaceSize, executor);
    }
};

// Isolate forwarding from runtime initialization, following the private-name API routing UT pattern.
struct ForwardProbe {
    std::array<const aclTensor*, 16> tensors{};
    const aclIntArray* splits = nullptr;
    aclnnStatus status = ACLNN_SUCCESS;
    unsigned firstCalls = 0;
    unsigned secondCalls = 0;
    uint64_t* workspaceOutput = nullptr;
    aclOpExecutor** executorOutput = nullptr;
    uint64_t workspaceSize = 64;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    aclrtStream stream = nullptr;
} probe;

aclnnStatus ProbeGetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* kCache,
                                  aclTensor* vCache, const aclTensor* indices, const aclTensor* scaleK,
                                  const aclTensor* scaleV, const aclTensor* offsetK, const aclTensor* offsetV,
                                  const aclTensor* weightScale, const aclTensor* activationScale, const aclTensor* bias,
                                  const aclIntArray* splits, char* quantMode, char* layout, bool kvOutput,
                                  char* cacheMode, const aclTensor* q, const aclTensor* k, const aclTensor* v,
                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    ++probe.firstCalls;
    const std::array<const aclTensor*, 16> actual = {x,      cos,    sin,     kCache,  vCache,      indices,
                                                     scaleK, scaleV, offsetK, offsetV, weightScale, activationScale,
                                                     bias,   q,      k,       v};
    EXPECT_EQ(actual, probe.tensors);
    EXPECT_EQ(splits, probe.splits);
    EXPECT_STREQ(quantMode, "static");
    EXPECT_STREQ(layout, "BSND");
    EXPECT_STREQ(cacheMode, "contiguous");
    EXPECT_TRUE(kvOutput);
    EXPECT_EQ(workspaceSize, probe.workspaceOutput);
    EXPECT_EQ(executor, probe.executorOutput);
    EXPECT_EQ(*workspaceSize, 0U);
    EXPECT_EQ(*executor, nullptr);
    *workspaceSize = probe.workspaceSize;
    *executor = probe.executor;
    return probe.status;
}

aclnnStatus ProbeRun(void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream)
{
    ++probe.secondCalls;
    EXPECT_EQ(workspace, probe.workspace);
    EXPECT_EQ(size, probe.workspaceSize);
    EXPECT_EQ(executor, probe.executor);
    EXPECT_EQ(stream, probe.stream);
    return probe.status;
}
} // namespace

// Compile the unchanged production body with private API/dependency names. The diagnostic tests below
// still call the real public API. This routing seam requires no device or shared framework changes.
#define API_NAME DRQK_ROUTE_API_NAME
#define aclnnDequantRopeQuantKvcacheGetWorkspaceSize DrqkRouteGetWorkspaceSize
#define aclnnDequantRopeQuantKvcache DrqkRouteRun
#define aclnnInnerDequantRopeQuantKvcacheGetWorkspaceSize ProbeGetWorkspaceSize
#define aclnnInnerDequantRopeQuantKvcache ProbeRun
#include "../../../../op_host/op_api/aclnn_dequant_rope_quant_kvcache.cpp"
#undef aclnnInnerDequantRopeQuantKvcache
#undef aclnnInnerDequantRopeQuantKvcacheGetWorkspaceSize
#undef aclnnDequantRopeQuantKvcache
#undef aclnnDequantRopeQuantKvcacheGetWorkspaceSize
#undef API_NAME

namespace {
TEST_F(DequantRopeQuantKvcacheApi, RejectInconsistentDtypesWithNamedReason)
{
    const std::array<std::array<aclDataType, 3>, 7> cases = {{{ACL_BF16, ACL_FLOAT16, ACL_FLOAT16},
                                                              {ACL_BF16, ACL_FLOAT16, ACL_BF16},
                                                              {ACL_BF16, ACL_BF16, ACL_FLOAT16},
                                                              {ACL_FLOAT16, ACL_BF16, ACL_BF16},
                                                              {ACL_INT32, ACL_FLOAT16, ACL_BF16},
                                                              {ACL_INT32, ACL_BF16, ACL_FLOAT16},
                                                              {ACL_INT32, ACL_FLOAT, ACL_FLOAT}}};
    for (const auto& types : cases) {
        SCOPED_TRACE(testing::PrintToString(types));
        ApiInputs inputs(types[0], types[1], types[2]);
        uint64_t workspaceSize = 12345;
        // Failure must clear both outputs without using or freeing this sentinel.
        auto* executor = reinterpret_cast<aclOpExecutor*>(&workspaceSize);
        EXPECT_EQ(inputs.Run(inputs.x.get(), inputs.cos.get(), inputs.sin.get(), &workspaceSize, &executor),
                  ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(workspaceSize, 0U);
        EXPECT_EQ(executor, nullptr);
        const std::string constraint =
            types[0] == ACL_INT32 ?
                "cos.dtype and sin.dtype must match and be FLOAT16 or BFLOAT16 when x.dtype is INT32" :
                "x.dtype, cos.dtype and sin.dtype must be consistent when x.dtype is FLOAT16 or BFLOAT16";
        const std::string reason = constraint +
                                   ". Actual dtypes: x.dtype=" + op::ToString(inputs.x->GetDataType()).GetString() +
                                   ", cos.dtype=" + op::ToString(inputs.cos->GetDataType()).GetString() +
                                   ", sin.dtype=" + op::ToString(inputs.sin->GetDataType()).GetString();
        drqk_test::ExpectSingleError("EZ0037",
                                     {{"op_name", API_NAME}, {"param_name", "x, cos, sin"}, {"reason", reason}});
    }
}

TEST_F(DequantRopeQuantKvcacheApi, RejectUnsupportedXDtype)
{
    ApiInputs inputs(ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT16);
    uint64_t workspaceSize = 12345;
    auto* executor = reinterpret_cast<aclOpExecutor*>(&workspaceSize);
    EXPECT_EQ(inputs.Run(inputs.x.get(), inputs.cos.get(), inputs.sin.get(), &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspaceSize, 0U);
    EXPECT_EQ(executor, nullptr);
    drqk_test::ExpectSingleError("EZ0019", {{"op_name", API_NAME},
                                            {"param_name", "x"},
                                            {"incorrect_dtype", "DT_FLOAT"},
                                            {"correct_dtype", "FLOAT16, BFLOAT16 or INT32"}});
}

TEST_F(DequantRopeQuantKvcacheApi, RejectRequiredNullPointersWithNamedReason)
{
    const std::array<const char*, 5> names = {"x", "cos", "sin", "workspaceSize", "executor"};
    for (size_t i = 0; i < names.size(); ++i) {
        SCOPED_TRACE(names[i]);
        ApiInputs inputs(ACL_BF16, ACL_BF16, ACL_BF16);
        uint64_t workspaceSize = 12345;
        aclOpExecutor* executor = nullptr;
        EXPECT_EQ(inputs.Run(i == 0 ? nullptr : inputs.x.get(), i == 1 ? nullptr : inputs.cos.get(),
                             i == 2 ? nullptr : inputs.sin.get(), i == 3 ? nullptr : &workspaceSize,
                             i == 4 ? nullptr : &executor),
                  ACLNN_ERR_PARAM_NULLPTR);
        drqk_test::ExpectSingleError("EZ0026", {{"op_name", API_NAME},
                                                {"param_name", names[i]},
                                                {"incorrect_value", "nullptr"},
                                                {"reason", "must not be nullptr"}});
    }
}

TEST_F(DequantRopeQuantKvcacheApi, ForwardSupportedDtypesToGeneratedInterface)
{
    const std::array<std::array<aclDataType, 2>, 4> cases = {
        {{ACL_FLOAT16, ACL_FLOAT16}, {ACL_BF16, ACL_BF16}, {ACL_INT32, ACL_FLOAT16}, {ACL_INT32, ACL_BF16}}};
    for (const auto& types : cases) {
        SCOPED_TRACE(testing::PrintToString(types));
        ApiInputs inputs(types[0], types[1], types[1]);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        probe = {};
        probe.tensors = {inputs.x.get(),
                         inputs.cos.get(),
                         inputs.sin.get(),
                         inputs.kCache.get(),
                         inputs.vCache.get(),
                         inputs.indices.get(),
                         inputs.scale.get(),
                         inputs.scale.get(),
                         nullptr,
                         nullptr,
                         types[0] == ACL_INT32 ? inputs.weightScale.get() : nullptr,
                         nullptr,
                         nullptr,
                         inputs.q.get(),
                         inputs.k.get(),
                         inputs.v.get()};
        probe.splits = inputs.splits.get();
        probe.workspaceOutput = &workspaceSize;
        probe.executorOutput = &executor;
        probe.executor = reinterpret_cast<aclOpExecutor*>(&probe);
        char quantMode[] = "static";
        char layout[] = "BSND";
        char cacheMode[] = "contiguous";
        for (auto status : {ACLNN_SUCCESS, ACLNN_ERR_INNER}) {
            probe.status = status;
            EXPECT_EQ(DrqkRouteGetWorkspaceSize(
                          inputs.x.get(), inputs.cos.get(), inputs.sin.get(), inputs.kCache.get(), inputs.vCache.get(),
                          inputs.indices.get(), inputs.scale.get(), inputs.scale.get(), nullptr, nullptr,
                          probe.tensors[10], nullptr, nullptr, inputs.splits.get(), quantMode, layout, true, cacheMode,
                          inputs.q.get(), inputs.k.get(), inputs.v.get(), &workspaceSize, &executor),
                      status);
            EXPECT_EQ(workspaceSize, probe.workspaceSize);
            EXPECT_EQ(executor, probe.executor);
            drqk_test::ExpectNoError();
        }
        EXPECT_EQ(probe.firstCalls, 2U);
    }
}

TEST_F(DequantRopeQuantKvcacheApi, ForwardExecutionArgumentsAndStatus)
{
    probe = {};
    probe.workspace = &probe;
    probe.executor = reinterpret_cast<aclOpExecutor*>(&probe);
    probe.stream = reinterpret_cast<aclrtStream>(&probe);
    for (auto status : {ACLNN_SUCCESS, ACLNN_ERR_INNER}) {
        probe.status = status;
        EXPECT_EQ(DrqkRouteRun(probe.workspace, probe.workspaceSize, probe.executor, probe.stream), status);
        drqk_test::ExpectNoError();
    }
    EXPECT_EQ(probe.secondCalls, 2U);
}
} // namespace
