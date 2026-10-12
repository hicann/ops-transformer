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
#include <vector>
#include <float.h>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_qkv_rms_norm_rope_cache.h"

#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

int batch_size;
int seq_len;
int Nqkv;
int Nq;
int Nk;
int Nv;
int dim;
int block_num;
int block_size;

class l2_qkv_rms_norm_rope_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_qkv_rms_norm_rope_cache_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_qkv_rms_norm_rope_cache_test TearDown" << endl;
    }
};

TEST_F(l2_qkv_rms_norm_rope_cache_test, qkv_rms_norm_rope_cache_case_01)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    TensorDesc qkv_desc = TensorDesc({batch_size * seq_len, Nqkv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc qGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({batch_size * seq_len}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc qOut_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_desc = TensorDesc({block_num, Nk * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc vCache_desc = TensorDesc({block_num, Nv * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc kScale_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vScale_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);

    char* cache_mode = "PA_NZ";
    auto qkv_size = IntArrayDesc(vector<int64_t>{batch_size, seq_len, Nqkv, dim});
    auto head_nums = IntArrayDesc(vector<int64_t>{Nq, Nk, Nv});
    double epsilon = 1e-6;

    TensorDesc qOut_proto_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_proto_desc = TensorDesc({batch_size * seq_len, Nk * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc vCache_proto_desc = TensorDesc({batch_size * seq_len, Nv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCache,
        INPUT(qkv_desc, qGamma_desc, kGamma_desc, cos_desc, sin_desc, index_desc, qOut_desc, kCache_desc, vCache_desc,
              kScale_desc, vScale_desc, nullptr, nullptr, qkv_size, head_nums, epsilon, cache_mode),
        OUTPUT(nullptr, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_qkv_rms_norm_rope_cache_test, qkv_rms_norm_rope_cache_case_01A)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    TensorDesc qkv_desc = TensorDesc({batch_size * seq_len, Nqkv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc qGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({batch_size * seq_len}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc qOut_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_desc = TensorDesc({block_num, Nk * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc vCache_desc = TensorDesc({block_num, Nv * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc kScale_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vScale_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);

    char* cache_mode = "PA_NZ";
    auto qkv_size = IntArrayDesc(vector<int64_t>{batch_size, seq_len, Nqkv, dim});
    auto head_nums = IntArrayDesc(vector<int64_t>{Nq, Nk, Nv});
    double epsilon = 1e-6;

    TensorDesc qOut_proto_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_proto_desc = TensorDesc({batch_size * seq_len, Nk * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc vCache_proto_desc = TensorDesc({batch_size * seq_len, Nv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCache,
        INPUT(qkv_desc, qGamma_desc, kGamma_desc, cos_desc, sin_desc, index_desc, qOut_desc, kCache_desc, vCache_desc,
              kScale_desc, vScale_desc, nullptr, nullptr, qkv_size, head_nums, epsilon, cache_mode),
        OUTPUT(qOut_proto_desc, kCache_proto_desc, vCache_proto_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_qkv_rms_norm_rope_cache_test, qkv_rms_norm_rope_cache_case_02)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    TensorDesc qkv_desc = TensorDesc({batch_size * seq_len, Nqkv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc qGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({batch_size * seq_len}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc qOut_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_desc = TensorDesc({block_num, Nk * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc vCache_desc = TensorDesc({block_num, Nv * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc kScale_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vScale_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc kOffset_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vOffset_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);

    char* cache_mode = "PA_NZ";
    auto qkv_size = IntArrayDesc(vector<int64_t>{batch_size, seq_len, Nqkv, dim});
    auto head_nums = IntArrayDesc(vector<int64_t>{Nq, Nk, Nv});
    double epsilon = 1e-6;

    TensorDesc qOut_proto_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_proto_desc = TensorDesc({batch_size * seq_len, Nk * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc vCache_proto_desc = TensorDesc({batch_size * seq_len, Nv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCache,
        INPUT(qkv_desc, qGamma_desc, kGamma_desc, cos_desc, sin_desc, index_desc, qOut_desc, kCache_desc, vCache_desc,
              kScale_desc, vScale_desc, kOffset_desc, vOffset_desc, qkv_size, head_nums, epsilon, cache_mode),
        OUTPUT(qOut_proto_desc, kCache_proto_desc, vCache_proto_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_qkv_rms_norm_rope_cache_test, qkv_rms_norm_rope_cache_case_02A)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    TensorDesc qkv_desc = TensorDesc({batch_size * seq_len, Nqkv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc qGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({batch_size * seq_len}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc qOut_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_desc = TensorDesc({block_num, Nk * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc vCache_desc = TensorDesc({block_num, Nv * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc kScale_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vScale_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc kOffset_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vOffset_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);

    char* cache_mode = "PA_NZ";
    auto qkv_size = IntArrayDesc(vector<int64_t>{batch_size, seq_len, Nqkv, dim});
    auto head_nums = IntArrayDesc(vector<int64_t>{Nq, Nk, Nv});
    double epsilon = 1e-6;

    TensorDesc qOut_proto_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_proto_desc = TensorDesc({batch_size * seq_len, Nk * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc vCache_proto_desc = TensorDesc({batch_size * seq_len, Nv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQkvRmsNormRopeCache,
        INPUT(qkv_desc, qGamma_desc, kGamma_desc, cos_desc, sin_desc, index_desc, qOut_desc, kCache_desc, vCache_desc,
              kScale_desc, vScale_desc, kOffset_desc, vOffset_desc, qkv_size, head_nums, epsilon, cache_mode),
        OUTPUT(nullptr, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_qkv_rms_norm_rope_cache_test, qkv_rms_norm_rope_cache_case_03)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    TensorDesc qkv_desc = TensorDesc({batch_size * seq_len, Nqkv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc qGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({batch_size * seq_len}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc qOut_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_desc = TensorDesc({block_num, Nk * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc vCache_desc = TensorDesc({block_num, Nv * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc kScale_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vScale_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);

    char* cache_mode = "PA_NZ";
    auto qkv_size = IntArrayDesc(vector<int64_t>{batch_size, seq_len, Nqkv, dim});
    auto head_nums = IntArrayDesc(vector<int64_t>{Nq, Nk, Nv});
    double epsilon = 1e-6;

    TensorDesc qOut_proto_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_proto_desc = TensorDesc({batch_size * seq_len, Nk * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc vCache_proto_desc = TensorDesc({batch_size * seq_len, Nv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnQkvRmsNormRopeCache,
                  INPUT(qkv_desc, qGamma_desc, kGamma_desc, cos_desc, sin_desc, index_desc, qOut_desc, kCache_desc,
                        vCache_desc, nullptr, nullptr, nullptr, nullptr, qkv_size, head_nums, epsilon, cache_mode),
                  OUTPUT(qOut_proto_desc, kCache_proto_desc, vCache_proto_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_qkv_rms_norm_rope_cache_test, qkv_rms_norm_rope_cache_case_03A)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    TensorDesc qkv_desc = TensorDesc({batch_size * seq_len, Nqkv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc qGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kGamma_desc = TensorDesc({dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc cos_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sin_desc = TensorDesc({batch_size * seq_len, dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({batch_size * seq_len}, ACL_INT64, ACL_FORMAT_ND);
    TensorDesc qOut_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_desc = TensorDesc({block_num, Nk * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc vCache_desc = TensorDesc({block_num, Nv * dim / 32, block_size, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc kScale_desc = TensorDesc({Nk, dim}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc vScale_desc = TensorDesc({Nv, dim}, ACL_FLOAT, ACL_FORMAT_ND);

    char* cache_mode = "PA_NZ";
    auto qkv_size = IntArrayDesc(vector<int64_t>{batch_size, seq_len, Nqkv, dim});
    auto head_nums = IntArrayDesc(vector<int64_t>{Nq, Nk, Nv});
    double epsilon = 1e-6;

    TensorDesc qOut_proto_desc = TensorDesc({batch_size * seq_len, Nq * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc kCache_proto_desc = TensorDesc({batch_size * seq_len, Nk * dim}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc vCache_proto_desc = TensorDesc({batch_size * seq_len, Nv * dim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnQkvRmsNormRopeCache,
                  INPUT(qkv_desc, qGamma_desc, kGamma_desc, cos_desc, sin_desc, index_desc, qOut_desc, kCache_desc,
                        vCache_desc, nullptr, nullptr, nullptr, nullptr, qkv_size, head_nums, epsilon, cache_mode),
                  OUTPUT(nullptr, nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
