/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_scatter_pa_kv_cache_with_k_scale.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class scatter_pa_kv_cache_with_k_scale_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_pa_kv_cache_with_k_scale_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "scatter_pa_kv_cache_with_k_scale_opapi_ut TearDown" << endl;
    }
};

// nullptr 测试
TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_key_nullptr)
{
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(aclnnScatterPaKvCacheWithKScale,
                        INPUT((aclTensor*)nullptr, value, keyCacheRef, valueCacheRef, slotMapping, keyScale,
                              keyScaleCacheRef, cacheLayout),
                        OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_value_nullptr)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(aclnnScatterPaKvCacheWithKScale,
                        INPUT(key, (aclTensor*)nullptr, keyCacheRef, valueCacheRef, slotMapping, keyScale,
                              keyScaleCacheRef, cacheLayout),
                        OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_key_cache_nullptr)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, (aclTensor*)nullptr, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout),
        OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_slot_mapping_nullptr)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, (aclTensor*)nullptr, keyScale, keyScaleCacheRef, cacheLayout),
        OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_key_scale_nullptr)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, (aclTensor*)nullptr, keyScaleCacheRef, cacheLayout),
        OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype 不匹配测试
TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_key_dtype)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_key_keycache_dtype_mismatch)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_key_value_dtype_mismatch)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_slot_mapping_dtype)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_key_scale_dtype)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape 不匹配测试
TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_key_dim)
{
    auto key = TensorDesc({2, 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_value_dim)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_slot_mapping_dim)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_key_cache_dim)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_invalid_key_scale_dim)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常功能测试 - 4 种 dtype 组合
TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut,
       scatter_pa_kv_cache_with_k_scale_aclnn_dtype_combo_1_float8_e5m2_int64)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut,
       scatter_pa_kv_cache_with_k_scale_aclnn_dtype_combo_2_float8_e4m3fn_int64)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut,
       scatter_pa_kv_cache_with_k_scale_aclnn_dtype_combo_3_float8_e5m2_int32)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut,
       scatter_pa_kv_cache_with_k_scale_aclnn_dtype_combo_4_float8_e4m3fn_int32)
{
    auto key = TensorDesc({2, 2, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto value = TensorDesc({2, 2, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 边界条件测试 - 不同 shape 组合
TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_shape_large_batch)
{
    auto key = TensorDesc({8, 2, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({8, 2, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({16, 2, 32, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({16, 2, 32, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({16, 2, 32, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_shape_small_batch)
{
    auto key = TensorDesc({1, 1, 32}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({1, 1, 32}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({2, 1, 8, 32}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({2, 1, 8, 32}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({2, 1, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty tensor 测试
TEST_F(scatter_pa_kv_cache_with_k_scale_opapi_ut, scatter_pa_kv_cache_with_k_scale_aclnn_empty_slot_mapping)
{
    auto key = TensorDesc({0, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto value = TensorDesc({0, 2, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto keyCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto valueCacheRef = TensorDesc({4, 2, 16, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto slotMapping = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto keyScale = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto keyScaleCacheRef = TensorDesc({4, 2, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    char cacheLayout[] = "BNBD";

    auto ut = OP_API_UT(
        aclnnScatterPaKvCacheWithKScale,
        INPUT(key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, keyScaleCacheRef, cacheLayout), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
