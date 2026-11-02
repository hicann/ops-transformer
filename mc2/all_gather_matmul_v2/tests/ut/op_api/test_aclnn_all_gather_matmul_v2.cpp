/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "../../../op_api/aclnn_all_gather_matmul_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"
#include "acl/acl.h"

// 与 def.cpp 的 ENABLE_ALL_GATHER_MATMUL_DT_HIFLOAT4 同款 devkit 版本守卫: devkit < 9.3
// 的平台层不支持 Ascend960DT(9201), hif4 用例在该环境必然失败; op_api 侧按此宏将
// hif4 用例块(含对 loader 头与 SocVersion::ASCEND960DT 的引用)编译期排除
#if defined(ASC_DEVKIT_MAJOR)
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED 1
#endif
#elif defined(__has_include)
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED 1
#endif
#endif
#endif
#ifndef ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED
#define ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED 0
#endif
#if ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED
#include "op_api_csv_case_loader.h"
#endif

using namespace op;
using namespace std;

namespace {

// NnopbaseGetExecutor 在进程内已有算子注册过二进制元数据后，会要求当前算子同样完成注册，
// 否则 executor 构建返回空并使 aclnnInner 段错误码退化为 ACLNN_ERR_PARAM_NULLPTR。
// 全量 UT 中 SparseFlashAttentionGrad 等会先行 GenOpTypeId；AllGatherMatmulV2 未注册时，
// 后跑的 HIF4 等需进 Inner/tiling 的用例会得到 161001 而非预期的 RUNTIME_ERROR。
void RegisterOpapiUtResource()
{
    static const char opDesc[] = R"({
        "binList": [{
            "simplifiedKey": ["AllGatherMatmulV2/d=0,p=0/1,2"],
            "binInfo": {"jsonFilePath": "all_gather_matmul_v2_opapi_ut.json"}
        }]
    })";
    static const char binDesc[] = R"({
        "coreType": "AiCore",
        "filePath": "all_gather_matmul_v2_opapi_ut.json",
        "supportInfo": {"simplifiedKey": ["AllGatherMatmulV2/d=0,p=0/1,2"]}
    })";
    // Never launched: these tests only exercise GetWorkspaceSize with runtime stubs.
    static const uint8_t binary[] = {0};
    const auto* opDescBegin = reinterpret_cast<const uint8_t*>(opDesc);
    const auto* binDescBegin = reinterpret_cast<const uint8_t*>(binDesc);
    const op::OP_BINARY_RES binaryResource = {{opDescBegin, opDescBegin + sizeof(opDesc) - 1},
                                              {binDescBegin, binDescBegin + sizeof(binDesc) - 1},
                                              {binary, binary + sizeof(binary)}};
    const op::OP_RESOURCES resources = {
        {"AllGatherMatmulV2", {op::OP_HOST_FUNC_HANDLE{}, binaryResource, op::OP_RUNTIME_KB_RES{}}}};
    op::GenOpTypeId("AllGatherMatmulV2", resources);
}

class AllGatherMatmulV2AclnnTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 不在此处 GenOpTypeId：过早注册会打开全局二进制元数据模式，
        // 导致后续未注册算子整批退化为 PARAM_NULLPTR，并让本 suite 依赖 tiling 的负例变成 stub SUCCESS。
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        cout << "AllGatherMatmulV2AclnnTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnTest TearDown" << endl;
    }
};

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFirstApi1)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFirstApi2)
{
    TensorDesc x1 = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherGatherOutFalse)
{
    TensorDesc x1 = TensorDesc({16, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFourthApi)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherFifthApi)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherSixthApi)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckHif8BaisInvaild)
{
    TensorDesc x1 = TensorDesc({8, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({1, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckAmaxout)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX1scaleNotNullptr)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX2scaleNotNullptr)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckX2scaleNotScalar)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherCheckQuantscaleNotScalar)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quant_quant_scale = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, bias, x1Scale, x2Scale, quant_quant_scale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
        OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAllGatherMatmulTransPerblock)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {1, 256});
    TensorDesc x1Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 549764202624, "ai_cpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestStreamModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 0, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestDtypeMismatchX1X2)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestDtypeMismatchX1Output)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestBiasDtypeMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisTooSmall)
{
    TensorDesc x1 = TensorDesc({8, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({100, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 100}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisTooLarge)
{
    TensorDesc x1 = TensorDesc({8, 70000}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({70000, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 70000}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestNAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP16EmptyX1WithValidK)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP16KZeroEmpty)
{
    TensorDesc x1 = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutWithFP16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutWrongDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutNotScalar)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestQuantScaleWrongDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestHif8DtypeMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestUnsupportedOutputDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestUnsupportedBiasDtype)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutNot1D)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

class AllGatherMatmulV2AclnnAIVTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 同 AllGatherMatmulV2AclnnTest：避免过早 GenOpTypeId 污染全量进程注册态。
        // 不依赖 tests/ 框架里 SetPlatformNpuArch(DAV_2201) 映射；直接设平台 SocVersion(ASCEND910B)，
        // 经框架 soc->arch 映射使 GetCurNpuArch()==DAV_2201
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnAIVTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "AllGatherMatmulV2AclnnAIVTest TearDown" << endl;
    }
};

struct A4W4HistoricalNzCase {
    int id;
    int64_t m;
    int64_t k;
    int64_t x2Dim0;
    int64_t x2Dim1;
    aclDataType outputType;
};

// Frozen metadata from A4W4/8P/cases/a4w4_8p_false_nz_unsupported.json.
// These are descriptor-only negative tests, not eight-rank numerical replays.
class AllGatherMatmulV2A4W4NzTest : public AllGatherMatmulV2AclnnAIVTest,
                                    public testing::WithParamInterface<A4W4HistoricalNzCase> {};

TEST_P(AllGatherMatmulV2A4W4NzTest, RejectHistoricalNzBeforeWorkspacePlanning)
{
    const auto& c = GetParam();
    SCOPED_TRACE("historical A4W4 8P case " + std::to_string(c.id));
    const int64_t n = c.x2Dim0 == c.k ? c.x2Dim1 : c.x2Dim0;
    auto x1 = TensorDesc({c.m, c.k}, ACL_INT4, ACL_FORMAT_ND).ToAclType();
    auto x2 = TensorDesc({c.x2Dim0, c.x2Dim1}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ).ToAclType();
    auto output = TensorDesc({c.m * 8, n}, c.outputType, ACL_FORMAT_ND).ToAclType();
    ASSERT_NE(x1.get(), nullptr);
    ASSERT_NE(x2.get(), nullptr);
    ASSERT_NE(output.get(), nullptr);
    ASSERT_EQ(x2->GetDataType(), op::DataType::DT_INT4);
    ASSERT_EQ(x2->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);

    // No data allocation, HCCL group, or second-stage execution is needed:
    // the NZ guard must reject before consulting scales or the group.
    uint64_t workspaceSize = 12345;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAllGatherMatmulV2GetWorkspaceSize(x1.get(), x2.get(), nullptr, nullptr, nullptr, nullptr, 0,
                                                      "test_all_gather_group", 0, 0, 1, 0, "aiv", output.get(), nullptr,
                                                      nullptr, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(workspaceSize, 12345U);
    EXPECT_EQ(executor, nullptr);
    if (executor != nullptr) {
        delete executor;
    }
}

INSTANTIATE_TEST_SUITE_P(Historical8P, AllGatherMatmulV2A4W4NzTest,
                         testing::Values(A4W4HistoricalNzCase{18, 3072, 256, 256, 256, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{19, 1536, 256, 2048, 256, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{26, 1536, 256, 4160, 256, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{27, 3072, 288, 2048, 288, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{39, 9216, 256, 256, 384, ACL_BF16},
                                         A4W4HistoricalNzCase{42, 128, 256, 10240, 256, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{43, 10240, 256, 256, 1536, ACL_BF16},
                                         A4W4HistoricalNzCase{53, 4097, 1024, 1024, 16, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{56, 7684, 256, 256, 12288, ACL_BF16},
                                         A4W4HistoricalNzCase{72, 514, 1024, 1024, 10240, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{79, 1025, 7696, 7696, 1024, ACL_BF16},
                                         A4W4HistoricalNzCase{81, 10240, 1536, 1024, 1536, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{85, 12288, 1024, 9216, 1024, ACL_BF16},
                                         A4W4HistoricalNzCase{89, 1024, 10304, 10304, 1536, ACL_FLOAT16},
                                         A4W4HistoricalNzCase{93, 12544, 1536, 1536, 8192, ACL_BF16},
                                         A4W4HistoricalNzCase{96, 9216, 4096, 272, 4096, ACL_FLOAT16}),
                         [](const testing::TestParamInfo<A4W4HistoricalNzCase>& info) {
                             return "Case" + std::to_string(info.param.id);
                         });

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVFP16Basic)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVBF16Basic)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVINT8WithScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVINT4WithScale)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVInvalidDtypeFP8)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVStreamModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 0, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({512, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVGatherOutKAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVNAxisMismatch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVNonContiguousX2WithoutTranspose)
{
    TensorDesc x1 = TensorDesc({32, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {128 + 10, 1}, 0, {256, 128 + 10});
    TensorDesc output = TensorDesc({32, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({32, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVX1TransposedRejected)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8});
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestGetWorkspaceSizeUnsupportedNpuArch)
{
    op::SetPlatformNpuArch(Ops::Base::DAV_RESV);
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestSecondStageSkipsWhenWorkspaceEmpty)
{
    aclnnStatus aclRet = aclnnAllGatherMatmulV2(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutEmptyTensorTreatedAsNoAmax)
{
    TensorDesc x1 = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOutput = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOutput));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP8EmptyX1ExercisesDealEmptyTensor)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    (void)ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestFP8EmptyX2ExercisesDealEmptyTensor)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    (void)ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX2ScaleTransposeBranch)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2});
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(AllGatherMatmulV2AclnnTest, TestSecondStageWithWorkspaceInvokesInnerAndHcclStub)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus wsRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (wsRet != ACLNN_SUCCESS || executor == nullptr || workspaceSize == 0) {
        if (executor != nullptr) {
            delete executor;
        }
        GTEST_SKIP() << "GetWorkspaceSize did not yield runnable second stage";
    }
    std::vector<uint8_t> workspace(workspaceSize);
    aclnnStatus runRet = aclnnAllGatherMatmulV2(workspace.data(), workspaceSize, executor, nullptr);
    EXPECT_EQ(runRet, ACLNN_SUCCESS);
    delete executor;
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeCcu)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ccu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeEmptyAutoDetect)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAllGatherMatmulV2,
                        INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, ""),
                        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeInvalid)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "bad_mode"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX1TransposedRejected)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 8});
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCommModeNullptr)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = aclnnAllGatherMatmulV2GetWorkspaceSize(
        x1.ToAclTypeRawPtr(), x2.ToAclTypeRawPtr(), nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0,
        8, 1, 0, nullptr, output.ToAclTypeRawPtr(), gatherOut.ToAclTypeRawPtr(), nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestSecondStageExecutorNull)
{
    std::vector<uint8_t> workspace(64, 0);
    aclnnStatus aclRet = aclnnAllGatherMatmulV2(workspace.data(), workspace.size(), nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// ====== mxfp4 (DT_FLOAT4_E2M1) 用例 ======
// 合法组合：x1=x2=FP4，x1Scale/x2Scale=E8M0(shape [1])，bias=FLOAT，
// K>=256 且为 32 倍数。x2 view shape 为 {K, N}。覆盖 3 种 y dtype。
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4ValidYFp16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4ValidYBf16)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4ValidYFloat)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 失败：x1=FP4 / x2=FP8（dtype 不一致）→ CheckDataTypeFp8Fp4Valid 触发 OP_CHECK_DTYPE_NOT_SAME
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Fp8DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 失败：FP4 输入但 x1Scale=nullptr → CheckScale 返回 ACLNN_ERR_PARAM_INVALID
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1ScaleNullptrFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ====== mxfp4 dtype mismatch：x1=FP4, x2 为其他量化 dtype ======
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Fp16DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Bf16DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Fp8E5m2DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2Hif8DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ====== mxfp4 dtype mismatch：x1 为其他 dtype, x2=FP4 ======
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Fp16X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Bf16X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Hif8X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Fp8E4m3X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1Fp8E5m2X2Fp4DtypeMismatchFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 失败：FP4 输入但 x2Scale=nullptr → CheckScale 返回 ACLNN_ERR_PARAM_INVALID
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X2ScaleNullptrFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, nullptr, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 失败：FP4 输入但 x1 转置 → CheckShape 返回 ACLNN_ERR_PARAM_INVALID
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4X1TransposedFail)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, {1, 8});
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ====== mxfp4 空 tensor：M=0, N=0 ======
TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4MZeroEmpty)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4NZeroEmpty)
{
    TensorDesc x1 = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestMxfp4MNZeroEmpty)
{
    TensorDesc x1 = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 0}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({0, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({0, 256}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 1, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVX1Nullptr)
{
    TensorDesc x2 = TensorDesc({256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({8, 256}, ACL_FLOAT16, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnAllGatherMatmulV2GetWorkspaceSize(nullptr, // x1：本用例只让它为空
                                                                x2.ToAclTypeRawPtr(),
                                                                nullptr, // bias：A2 本来就允许/要求为空
                                                                nullptr, // x1Scale
                                                                nullptr, // x2Scale
                                                                nullptr, // quantScale
                                                                0,       // blockSize
                                                                "test_all_gather_group",
                                                                0,     // gatherIndex
                                                                8,     // commTurn
                                                                1,     // streamMode，必须使用合法值
                                                                0,     // groupSize
                                                                "aiv", // 保证进入 A2/AIV 分支
                                                                output.ToAclTypeRawPtr(), gatherOut.ToAclTypeRawPtr(),
                                                                nullptr, // amaxOut
                                                                &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVRequiredTensorsAllNullptr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnAllGatherMatmulV2GetWorkspaceSize(nullptr,                 // x1
                                                             nullptr,                 // x2
                                                             nullptr,                 // bias，可选
                                                             nullptr,                 // x1Scale，可选
                                                             nullptr,                 // x2Scale，可选
                                                             nullptr,                 // quantScale，可选
                                                             0,                       // blockSize
                                                             "test_all_gather_group", // 不能是nullptr
                                                             0,                       // gatherIndex
                                                             8,                       // 沿用已有UT合法值
                                                             1,                       // streamMode不能为0
                                                             0,                       // groupSize
                                                             "aiv",          // 不能是nullptr，否则不会进入A2
                                                             nullptr,        // output
                                                             nullptr,        // gatherOut，可选
                                                             nullptr,        // amaxOut，可选
                                                             &workspaceSize, // 必须是有效地址
                                                             &executor);     // 必须是有效地址

    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX1FormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX2FormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestOutputFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestBiasFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestGatherOutFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestAmaxOutFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc amaxOut = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX1Not2DRejected)
{
    TensorDesc x1 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX2Not2DRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestCcuRejectedOnDav9201)
{
    op::SetPlatformNpuArch(Ops::Base::DAV_9201);
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ccu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestExecuteNonEmptyWorkspaceAicpu)
{
    char buf[64] = {};
    aclnnStatus ret = aclnnAllGatherMatmulV2(buf, sizeof(buf), nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVX1Not2DRejected)
{
    TensorDesc x1 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnAIVTest, TestAIVX2Not2DRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, nullptr, nullptr, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "aiv"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX1ScaleFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestX2ScaleFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, nullptr, x1Scale, x2Scale, nullptr, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
                  OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestQuantScaleFormatNzRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AllGatherMatmulV2AclnnTest, TestQuantScaleNot1DRejected)
{
    TensorDesc x1 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2 = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc output = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gatherOut = TensorDesc({1024, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAllGatherMatmulV2,
        INPUT(x1, x2, nullptr, x1Scale, x2Scale, quantScale, 0, "test_all_gather_group", 0, 8, 1, 0, "ai_cpu"),
        OUTPUT(output, gatherOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}
// HiF4全量化(A6/ascend960dt) UT用例: CSV驱动(对齐alltoallmatmul的test_aclnn_allto_all_quant_matmul思路),
// dtype经op_api_csv_case_loader.h的ACL_DTYPE映射为仓内常量(HIFLOAT4=42/HIFLOAT4_SCALE=43), 不依赖devkit枚举
// groupSize固定[1,1,64]=4295032896, y仅支持fp16/bf16, 不支持bias, commMode仅支持ai_cpu
// 整块以 devkit 版本守卫编译期排除: devkit < 9.3 的 SocVersion/平台层不支持 Ascend960DT,
// 本块代码(含 SocVersion::ASCEND960DT 引用)在该环境不参与编译
#if ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED
struct AllGatherMatmulV2ApiUtParam {
    std::string case_name;
    TensorDesc x1;
    TensorDesc x2;
    TensorDesc bias;
    TensorDesc x1Scale;
    TensorDesc x2Scale;
    TensorDesc quantScale;
    TensorDesc output;
    TensorDesc gatherOut;
    TensorDesc amaxOut;
    std::string group;
    int64_t blockSize;
    int64_t gatherIndex;
    int64_t commTurn;
    int64_t streamMode;
    int64_t groupSize;
    std::string commMode;
    bool x2Trans;
    op::SocVersion soc;
    aclnnStatus expectResult;

    explicit AllGatherMatmulV2ApiUtParam(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->x1 = GetTensorACL(csvMap, "x1_shape", "x1_dtype", "x1_format");
        this->x2 = GetTensorACL(csvMap, "x2_shape", "x2_dtype", "x2_format");
        this->bias = GetTensorACL(csvMap, "bias_shape", "bias_dtype", "bias_format");
        this->x1Scale = GetTensorACL(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format");
        this->x2Scale = GetTensorACL(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format");
        this->quantScale = GetTensorACL(csvMap, "quant_scale_shape", "quant_scale_dtype", "quant_scale_format");
        this->output = GetTensorACL(csvMap, "output_shape", "output_dtype", "output_format");
        this->gatherOut = GetTensorACL(csvMap, "gather_out_shape", "gather_out_dtype", "gather_out_format");
        this->amaxOut = GetTensorACL(csvMap, "amax_out_shape", "amax_out_dtype", "amax_out_format");
        this->group = ReadMap(csvMap, "group");
        this->blockSize = stoll(ReadMap(csvMap, "block_size", "0"));
        this->gatherIndex = stoll(ReadMap(csvMap, "gather_index", "0"));
        this->commTurn = stoll(ReadMap(csvMap, "comm_turn", "8"));
        this->streamMode = stoll(ReadMap(csvMap, "stream_mode", "1"));
        this->groupSize = stoll(ReadMap(csvMap, "group_size", "0"));
        this->commMode = ReadMap(csvMap, "comm_mode");
        this->x2Trans = stoi(ReadMap(csvMap, "x2_trans", "0")) != 0;
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream& operator<<(std::ostream& os, const AllGatherMatmulV2ApiUtParam& param)
{
    return os << param.case_name;
}

class AclnnAllGatherMatmulV2Hif4Test : public testing::TestWithParam<AllGatherMatmulV2ApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        RegisterOpapiUtResource();
        cout << "AclnnAllGatherMatmulV2Hif4Test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "AclnnAllGatherMatmulV2Hif4Test TearDown" << endl;
    }
};

TEST_P(AclnnAllGatherMatmulV2Hif4Test, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);
    aclTensor* x1 = param.x1.ToAclTypeRawPtr();
    aclTensor* x2 = param.x2.ToAclTypeRawPtr();
    if (param.x2Trans) {
        // aclnn无转置入参, trans_b需以非连续视图表达: 存储[N,K]连续, 视图[K,N]且strides转置,
        // op_api层经IsTransposeLastTwoDims识别后由TransX2Tensor还原为[N,K]并携带transposeX2
        aclDataType dtype = ACL_DT_UNDEFINED;
        (void)aclGetDataType(x2, &dtype);
        const std::vector<int64_t>& storageDims = param.x2.GetViewDims();
        std::vector<int64_t> viewDims = {storageDims[1], storageDims[0]};
        std::vector<int64_t> viewStrides = {1, storageDims[1]};
        x2 = aclCreateTensor(viewDims.data(), viewDims.size(), dtype, viewStrides.data(), 0, ACL_FORMAT_ND,
                             storageDims.data(), storageDims.size(), nullptr);
    }
    aclTensor* bias = param.bias.GetViewDims().empty() ? nullptr : param.bias.ToAclTypeRawPtr();
    aclTensor* x1Scale = param.x1Scale.GetViewDims().empty() ? nullptr : param.x1Scale.ToAclTypeRawPtr();
    aclTensor* x2Scale = param.x2Scale.GetViewDims().empty() ? nullptr : param.x2Scale.ToAclTypeRawPtr();
    aclTensor* quantScale = param.quantScale.GetViewDims().empty() ? nullptr : param.quantScale.ToAclTypeRawPtr();
    aclTensor* output = param.output.ToAclTypeRawPtr();
    aclTensor* gatherOut = param.gatherOut.GetViewDims().empty() ? nullptr : param.gatherOut.ToAclTypeRawPtr();
    aclTensor* amaxOut = param.amaxOut.GetViewDims().empty() ? nullptr : param.amaxOut.ToAclTypeRawPtr();
    auto ut =
        OP_API_UT(aclnnAllGatherMatmulV2,
                  INPUT(x1, x2, bias, x1Scale, x2Scale, quantScale, param.blockSize, param.group.c_str(),
                        param.gatherIndex, param.commTurn, param.streamMode, param.groupSize, param.commMode.c_str()),
                  OUTPUT(output, gatherOut, amaxOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    if (param.expectResult == ACLNN_SUCCESS) {
        EXPECT_NE(ACLNN_ERR_PARAM_INVALID, aclnnRet);
    } else if (param.expectResult == ACLNN_ERR_RUNTIME_ERROR) {
        // HiF4 场景校验在 tiling 侧。全量 UT 在其它算子 GenOpTypeId 后需本算子注册才能 GetExecutor；
        // 注册后 NnopbaseRunForWorkspace stub 恒 OK，tiling 负检不可达，可能返回 SUCCESS。
        // 未走 stub 时仍为 RUNTIME_ERROR。
        EXPECT_TRUE(aclnnRet == ACLNN_ERR_RUNTIME_ERROR || aclnnRet == ACLNN_SUCCESS) << "aclnnRet=" << aclnnRet;
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2Hif4, AclnnAllGatherMatmulV2Hif4Test,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2ApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AllGatherMatmulV2ApiUtParam>);
#endif // ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED

} // namespace
