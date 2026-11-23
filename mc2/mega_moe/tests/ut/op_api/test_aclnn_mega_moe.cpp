/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn_mega_moe.h"

using namespace op;
using namespace std;

class AclnnMegaMoeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        std::cout << "MegaMoe AclnnMegaMoeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        std::cout << "MegaMoe AclnnMegaMoeTest TearDown" << std::endl;
    }
};

static aclnnStatus RunActivationParamCase(const char* activation, const std::vector<float>& activationParams)
{
    auto contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto xDesc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topkIdsDesc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topkWeightsDesc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    TensorListDesc weight1Descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2Descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weightScales1Descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weightScales2Descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    auto yDesc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expertTokenNumsDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activationParamsDesc = FloatArrayDesc(activationParams);

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(contextDesc, xDesc, topkIdsDesc, topkWeightsDesc, weight1Descs, weight2Descs, weightScales1Descs,
              weightScales2Descs, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "", 0, activation, activationParamsDesc, 0, 2, 0, 0),
        OUTPUT(yDesc, expertTokenNumsDesc));

    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

static aclnnStatus RunArch22SharedQuantCase(bool isInt4)
{
    constexpr int64_t kMoeExpertNum = 2;
    constexpr int64_t kSharedExpertNum = 2;
    const aclDataType weightDtype = isInt4 ? ACL_INT32 : ACL_INT8;
    const int64_t packedFactor = isInt4 ? 8 : 1;

    auto contextDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto xDesc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topkIdsDesc = TensorDesc({1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto topkWeightsDesc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1Descs(kMoeExpertNum,
                                TensorDesc({2048, 1024 / packedFactor}, weightDtype, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2Descs(kMoeExpertNum,
                                TensorDesc({1024, 1024 / packedFactor}, weightDtype, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weightScales1Descs(kMoeExpertNum, TensorDesc({2048}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weightScales2Descs(kMoeExpertNum, TensorDesc({1024}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2));

    TensorListDesc sharedWeight1Descs(
        kSharedExpertNum, TensorDesc({2048, 1024 / packedFactor}, weightDtype, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc sharedWeight2Descs(
        kSharedExpertNum, TensorDesc({1024, 1024 / packedFactor}, weightDtype, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc sharedWeightScales1Descs(kSharedExpertNum,
                                            TensorDesc({2048}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc sharedWeightScales2Descs(kSharedExpertNum,
                                            TensorDesc({1024}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc bias1Descs(kMoeExpertNum, TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc bias2Descs(kMoeExpertNum, TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc sharedBias1Descs(kSharedExpertNum, TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc sharedBias2Descs(kSharedExpertNum, TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1));

    auto yDesc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto expertTokenNumsDesc = TensorDesc({kMoeExpertNum}, ACL_INT32, ACL_FORMAT_ND);
    auto activationParamsDesc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    uint64_t workspaceSize = 0;
    if (isInt4) {
        auto ut = OP_API_UT(
            aclnnMegaMoe,
            INPUT(contextDesc, xDesc, topkIdsDesc, topkWeightsDesc, weight1Descs, weight2Descs, weightScales1Descs,
                  weightScales2Descs, bias1Descs, bias2Descs, nullptr, nullptr, sharedWeight1Descs, sharedWeight2Descs,
                  sharedWeightScales1Descs, sharedWeightScales2Descs, sharedBias1Descs, sharedBias2Descs, nullptr,
                  kMoeExpertNum, 2, 2097152, 0, 0, 0, 0, 0, "", 0, "swiglu", activationParamsDesc, 0, 2, 0, 0),
            OUTPUT(yDesc, expertTokenNumsDesc));
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(contextDesc, xDesc, topkIdsDesc, topkWeightsDesc, weight1Descs, weight2Descs, weightScales1Descs,
              weightScales2Descs, nullptr, nullptr, nullptr, nullptr, sharedWeight1Descs, sharedWeight2Descs,
              sharedWeightScales1Descs, sharedWeightScales2Descs, nullptr, nullptr, nullptr, kMoeExpertNum, 2, 2097152,
              0, 0, 0, 0, 0, "", 0, "swiglu", activationParamsDesc, 0, 2, 0, 0),
        OUTPUT(yDesc, expertTokenNumsDesc));
    return ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(AclnnMegaMoeTest, activation_param_domain)
{
    constexpr float kMax = std::numeric_limits<float>::max();

    // The op_api layer no longer duplicates activation-parameter validation; it lives on the
    // tiling side (see ActivationParamValidationWithLargeIntermediateHidden in
    // test_mega_moe_tiling.cpp), which every aclnn path goes through. Here we only verify that
    // valid parameter sets are accepted and forwarded to the mocked inner op, which returns
    // ACLNN_ERR_RUNTIME_ERROR in this test environment. Rejection of invalid parameter sets
    // (wrong count, NaN/Inf, non-positive scale) is covered by the arch-specific tiling UTs.
    EXPECT_EQ(RunActivationParamCase("swigluoai", {kMax, 0.0f, -1.0f}), ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(RunActivationParamCase("situglu", {4.0f}), ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(RunActivationParamCase("situglu", {4.0f, 25.0f}), ACLNN_ERR_RUNTIME_ERROR);
}

TEST_F(AclnnMegaMoeTest, arch22_int8_shared_expert_inputs_are_forwarded)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
    const aclnnStatus ret = RunArch22SharedQuantCase(false);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    // Like activation_param_domain, this is a wrapper acceptance smoke test:
    // the mock runtime cannot execute the inner op. Shape validation is covered
    // by the arch22 tiling tests; numerical correctness requires device tests.
    EXPECT_EQ(ret, ACLNN_ERR_RUNTIME_ERROR);
}

TEST_F(AclnnMegaMoeTest, arch22_packed_int32_weight_lists_wrapper_smoke)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
    const aclnnStatus ret = RunArch22SharedQuantCase(true);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    // ACLNN forwards the packed INT32 routed and shared weight lists as supplied.
    // This smoke test checks the error returned by the mocked inner operator.
    EXPECT_EQ(ret, ACLNN_ERR_RUNTIME_ERROR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_context)
{
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(nullptr, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs, weight_scales1_descs,
              weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "", 0, "swiglu", activation_params_desc, 0, 2, 0, 0),
        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_x)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, nullptr, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "",
                              0, "swiglu", activation_params_desc, 0, 2, 0, 0),
                        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_topk_ids)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(context_desc, x_desc, nullptr, topk_weights_desc, weight1_descs, weight2_descs, weight_scales1_descs,
              weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "", 0, "swiglu", activation_params_desc, 0, 2, 0, 0),
        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_topk_weights)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(context_desc, x_desc, topk_ids_desc, nullptr, weight1_descs, weight2_descs, weight_scales1_descs,
              weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "", 0, "swiglu", activation_params_desc, 0, 2, 0, 0),
        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_weight1)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, nullptr, weight2_descs, weight_scales1_descs,
              weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "", 0, "swiglu", activation_params_desc, 0, 2, 0, 0),
        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_weight2)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(
        aclnnMegaMoe,
        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, nullptr, weight_scales1_descs,
              weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "", 0, "swiglu", activation_params_desc, 0, 2, 0, 0),
        OUTPUT(y_desc, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_y_out)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto expert_token_nums_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "",
                              0, "swiglu", activation_params_desc, 0, 2, 0, 0),
                        OUTPUT(nullptr, expert_token_nums_desc));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMegaMoeTest, dav3510_nullptr_expert_token_nums_out)
{
    auto context_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto topk_ids_desc = TensorDesc({128, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 3);
    auto topk_weights_desc = TensorDesc({128, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

    TensorListDesc weight1_descs(4, TensorDesc({2048, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight2_descs(4, TensorDesc({4096, 1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ValueRange(-1, 1));
    TensorListDesc weight_scales1_descs(4, TensorDesc({2048}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));
    TensorListDesc weight_scales2_descs(4, TensorDesc({4096}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(0, 2));

    auto y_desc = TensorDesc({128, 4096}, ACL_BF16, ACL_FORMAT_ND);
    auto activation_params_desc = FloatArrayDesc(std::vector<float>{std::numeric_limits<float>::max()});

    auto ut = OP_API_UT(aclnnMegaMoe,
                        INPUT(context_desc, x_desc, topk_ids_desc, topk_weights_desc, weight1_descs, weight2_descs,
                              weight_scales1_descs, weight_scales2_descs, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 16, 4, 2097152, 0, 0, 0, 0, 0, "",
                              0, "swiglu", activation_params_desc, 0, 2, 0, 0),
                        OUTPUT(y_desc, nullptr));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
