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

// compressStride=0 非法：第一段应失败，避免进入窗口取模除零（issue 7234）
TEST_F(NsaCompressWithCacheApiTest, aclnn_nsa_compress_with_cache_reject_stride_0)
{
    constexpr int64_t batchSize = 4;
    constexpr int64_t pageBlockSize = 128;
    constexpr int64_t maxSeqLen = 512;
    constexpr int64_t blockNumPerBatch = maxSeqLen / pageBlockSize;
    constexpr int64_t blocksNum = blockNumPerBatch * batchSize;
    constexpr int64_t headsNum = 24;
    constexpr int64_t headsDim = 192;
    constexpr int64_t compressBlockSize = 32;

    auto input =
        TensorDesc({blocksNum, pageBlockSize, headsNum, headsDim}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight = TensorDesc({compressBlockSize, headsNum}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto slotMapping = TensorDesc({batchSize}, ACL_INT32, ACL_FORMAT_ND);
    IntArrayDesc actSeqLen(std::vector<int64_t>{32, 0, 0, 0});
    auto blockTable = TensorDesc({batchSize, blockNumPerBatch}, ACL_INT32, ACL_FORMAT_ND);
    char layout[] = "TND";
    auto outputCache = TensorDesc({maxSeqLen, headsNum, headsDim}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnNsaCompressWithCache,
        INPUT(input, weight, slotMapping, actSeqLen, blockTable, layout, compressBlockSize, 0, 1, pageBlockSize),
        OUTPUT(outputCache));
    uint64_t workspaceSize = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
