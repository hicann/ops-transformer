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
#include <iostream>
#include <vector>
#include <array>
#include "opdev/op_errno.h"
#include "../../../op_host/op_api/aclnn_nsa_compress.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class NsaCompressApiTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressApiTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressApiTest TearDown" << std::endl;
    }
};

TEST_F(NsaCompressApiTest, aclnn_nsa_compress_A1_small_fp16)
{
    auto input = TensorDesc({48, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    IntArrayDesc actSeqLen(std::vector<int64_t>{16, 32, 48});
    char layoutOptional[] = "TND";
    auto output = TensorDesc({3, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layoutOptional, 16, 16, 0), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(NsaCompressApiTest, aclnn_nsa_compress_A2_large_fp16)
{
    std::vector<int64_t> actSeqLenData;
    actSeqLenData.reserve(48);
    for (int64_t i = 1; i <= 48; ++i) {
        actSeqLenData.push_back(i * 100);
    }
    auto input = TensorDesc({4800, 4, 192}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight = TensorDesc({16, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    IntArrayDesc actSeqLen(actSeqLenData);
    char layoutOptional[] = "TND";
    auto output = TensorDesc({288, 4, 192}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layoutOptional, 16, 16, 0), OUTPUT(output));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(NsaCompressApiTest, invalid_input_and_weight_ranks)
{
    const std::vector<std::array<std::vector<int64_t>, 2>> shapes = {
        {{{48, 32}, {16, 32}}}, {{{48, 32, 16}, {16}}}, {{{48, 32, 16, 1}, {16, 32}}}, {{{48, 32, 16}, {16, 32, 1}}}};
    for (size_t i = 0; i < shapes.size(); ++i) {
        SCOPED_TRACE(i);
        auto input = TensorDesc(shapes[i][0], ACL_FLOAT16, ACL_FORMAT_ND);
        auto weight = TensorDesc(shapes[i][1], ACL_FLOAT16, ACL_FORMAT_ND);
        IntArrayDesc actSeqLen(std::vector<int64_t>{16, 32, 48});
        char layout[] = "TND";
        auto output = TensorDesc({3, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layout, 16, 16, 0), OUTPUT(output));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(NsaCompressApiTest, null_layout)
{
    auto input = TensorDesc({48, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weight = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    IntArrayDesc actSeqLen(std::vector<int64_t>{16, 32, 48});
    auto output = TensorDesc({3, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, static_cast<char*>(nullptr), 16, 16, 0),
                        OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NsaCompressApiTest, null_actual_sequence_lengths)
{
    auto input = TensorDesc({64, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({32, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    char layout[] = "TND";
    auto output = TensorDesc({2, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, nullptr, layout, 32, 32, 0), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(NsaCompressApiTest, matching_unsupported_dtypes)
{
    for (auto dtype : {ACL_FLOAT, ACL_INT32, ACL_DOUBLE}) {
        SCOPED_TRACE(dtype);
        auto input = TensorDesc({64, 4, 32}, dtype, ACL_FORMAT_ND);
        auto weight = TensorDesc({32, 4}, dtype, ACL_FORMAT_ND);
        IntArrayDesc actSeqLen(std::vector<int64_t>{64});
        char layout[] = "TND";
        auto output = TensorDesc({2, 4, 32}, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layout, 32, 32, 0), OUTPUT(output));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(NsaCompressApiTest, mismatched_supported_dtypes)
{
    for (size_t i = 0; i < 2; ++i) {
        SCOPED_TRACE(i);
        auto input = TensorDesc({64, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto weight = TensorDesc({32, 4}, i == 0 ? ACL_BF16 : ACL_FLOAT16, ACL_FORMAT_ND);
        IntArrayDesc actSeqLen(std::vector<int64_t>{64});
        char layout[] = "TND";
        auto output = TensorDesc({2, 4, 32}, i == 1 ? ACL_BF16 : ACL_FLOAT16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layout, 32, 32, 0), OUTPUT(output));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(NsaCompressApiTest, empty_actual_sequence_lengths)
{
    auto input = TensorDesc({64, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({32, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    IntArrayDesc actSeqLen(std::vector<int64_t>{});
    char layout[] = "TND";
    auto output = TensorDesc({2, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layout, 32, 32, 0), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NsaCompressApiTest, valid_bf16)
{
    auto input = TensorDesc({64, 4, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto weight = TensorDesc({32, 4}, ACL_BF16, ACL_FORMAT_ND);
    IntArrayDesc actSeqLen(std::vector<int64_t>{64});
    char layout[] = "TND";
    auto output = TensorDesc({2, 4, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnNsaCompress, INPUT(input, weight, actSeqLen, layout, 32, 32, 0), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
