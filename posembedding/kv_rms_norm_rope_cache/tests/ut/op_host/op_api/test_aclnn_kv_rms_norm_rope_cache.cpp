/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_kv_rms_norm_rope_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_kv_rms_norm_rope_cache_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_kv_rms_norm_rope_cache_test TearDown" << endl;
    }
};

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_01)
{
    TensorDesc kv_desc = TensorDesc({181, 1, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({181, 1}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({181, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc k_rope_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_desc = TensorDesc({181, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    char* cache_mode = "Norm";
    bool is_output_kv = false;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              nullptr, nullptr, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(k_rope_desc, ckv_desc));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_02_null_proto)
{
    TensorDesc kv_desc = TensorDesc({181, 1, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({181, 1}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({181, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({181, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    char* cache_mode = "Norm";
    bool is_output_kv = false;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              nullptr, nullptr, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_03_pa_bnsd_scale)
{
    TensorDesc kv_desc = TensorDesc({64, 1, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({64}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({576, 128, 1, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({576, 128, 1, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc k_rope_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc ckv_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc k_rope_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_desc = TensorDesc({64, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    char* cache_mode = "PA_BNSD";
    bool is_output_kv = false;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              k_rope_scale_desc, ckv_scale_desc, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(k_rope_desc, ckv_desc));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_04_pa_blk_bnsd_offset)
{
    TensorDesc kv_desc = TensorDesc({64, 1, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({64}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({576, 128, 1, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({576, 128, 1, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc k_rope_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc ckv_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc k_rope_offset_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc ckv_offset_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc k_rope_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_desc = TensorDesc({64, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    char* cache_mode = "PA_BLK_BNSD";
    bool is_output_kv = false;

    auto ut = OP_API_UT(
        aclnnKvRmsNormRopeCache,
        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc, k_rope_scale_desc,
              ckv_scale_desc, k_rope_offset_desc, ckv_offset_desc, epsilon, cache_mode, is_output_kv),
        OUTPUT(k_rope_desc, ckv_desc));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_05_pa_nz)
{
    TensorDesc kv_desc = TensorDesc({72, 2, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({72, 2, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({72, 2, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({144}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({72, 2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({72, 2, 128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc k_rope_desc = TensorDesc({72, 2, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_desc = TensorDesc({72, 2, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    char* cache_mode = "PA_NZ";
    bool is_output_kv = false;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              nullptr, nullptr, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(k_rope_desc, ckv_desc));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_06_pa_blk_nz_null_proto)
{
    TensorDesc kv_desc = TensorDesc({72, 2, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({72, 2, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({72, 2, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({144}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({72, 2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({72, 2, 128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    char* cache_mode = "PA_BLK_NZ";
    bool is_output_kv = false;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              nullptr, nullptr, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_07_output_kv_empty_cache_mode)
{
    TensorDesc kv_desc = TensorDesc({64, 1, 1, 576}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({64}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({64, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc k_rope_desc = TensorDesc({64, 1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc ckv_desc = TensorDesc({64, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    char* cache_mode = "";
    bool is_output_kv = true;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              nullptr, nullptr, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(k_rope_desc, ckv_desc));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_08_bf16)
{
    TensorDesc kv_desc = TensorDesc({64, 1, 1, 576}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc gamma_desc = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({64, 1, 1, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({64, 1, 1, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({64}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc kpe_cache_desc = TensorDesc({64, 1, 1, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc ckv_cache_desc = TensorDesc({64, 1, 1, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc k_rope_desc = TensorDesc({64, 1, 1, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc ckv_desc = TensorDesc({64, 1, 1, 512}, ACL_BF16, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    char* cache_mode = "Norm";
    bool is_output_kv = false;

    auto ut = OP_API_UT(aclnnKvRmsNormRopeCache,
                        INPUT(kv_desc, gamma_desc, cos_desc, sin_desc, index_desc, kpe_cache_desc, ckv_cache_desc,
                              nullptr, nullptr, nullptr, nullptr, epsilon, cache_mode, is_output_kv),
                        OUTPUT(k_rope_desc, ckv_desc));
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_kv_rms_norm_rope_cache_test, kv_rms_norm_rope_cache_case_09_run_api_wrapper)
{
    aclnnStatus acl_ret = aclnnKvRmsNormRopeCache(nullptr, 0, nullptr, nullptr);
}
