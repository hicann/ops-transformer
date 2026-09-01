/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fetch_grad_tiling.cpp
 * \brief EngramFetchGrad 算子 host 侧 tiling UT
 *
 * 测试用例覆盖:
 * - 正常场景: bf16/fp16/fp32 gradFetched, 各种 token/hidden 组合
 * - 异常场景: dtype 不匹配、维度错误、attr 非法值
 *
 * EngramFetchGrad 输入输出规格:
 *   input:  commContext(0), gradFetched(1), perm(2), sendCounts(3), recvCounts(4), recvLocalEntry(5), numRecv(6)
 *   output: gradUniqueOut(0), uniqueLocalEntryOut(1), numUniqueOut(2)
 *   attr:   num_entries_per_rank, comm_buffer_size
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

namespace EngramFetchGradUT {

static const std::string OP_NAME = "EngramFetchGrad";

struct EngramFetchGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> commContextShape;
    ge::DataType commContextDtype;
    ge::Format commContextFormat;
    std::initializer_list<int64_t> gradFetchedShape;
    ge::DataType gradFetchedDtype;
    ge::Format gradFetchedFormat;
    std::initializer_list<int64_t> permShape;
    ge::DataType permDtype;
    std::initializer_list<int64_t> sendCountsShape;
    std::initializer_list<int64_t> recvCountsShape;
    std::initializer_list<int64_t> recvLocalEntryShape;
    std::initializer_list<int64_t> gradUniqueShape;
    ge::DataType gradUniqueDtype;
    std::initializer_list<int64_t> uniqueLocalEntryShape;
    std::initializer_list<int64_t> numUniqueShape;
    int64_t numEntriesPerRank;
    int64_t commBufferSize;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
};

inline std::ostream &operator<<(std::ostream &os, const EngramFetchGradTestParam &param)
{
    return os << param.caseName;
}

static EngramFetchGradTestParam g_testCases[] = {
    {"success_bf16_basic",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {}},

    {"success_fp16_basic",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_FLOAT16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {}},

    {"success_fp32_basic",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_FLOAT,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {}},

    {"success_large_tokens",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {128, 256},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {128},
     ge::DT_INT32,
     {16},
     {2},
     {256},
     {256, 256},
     ge::DT_BF16,
     {256},
     {1},
     100,
     4194304,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {}},

    {"success_single_token",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT32,
     {16},
     {2},
     {2},
     {2, 512},
     ge::DT_BF16,
     {2},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {}},

    {"fail_gradFetched_dtype_int32",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_INT32,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_gradUnique_dtype_int32",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_INT32,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_perm_dtype_float",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_FLOAT,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_gradFetched_1d",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {4096},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_perm_dim0_mismatch",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {16},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_numUnique_dim0_not_1",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {2},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_hidden_size_zero",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 0},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 0},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_hidden_size_not_aligned",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 100},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 100},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_num_entries_negative",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {1},
     -1,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_commContext_dtype_bf16",
     {6146},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     {16},
     {2},
     {16},
     {16, 512},
     ge::DT_BF16,
     {16},
     {1},
     4,
     4194304,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},
};

class EngramFetchGradArch35TilingTest : public testing::TestWithParam<EngramFetchGradTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramFetchGradArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramFetchGradArch35TilingTest TearDown." << std::endl;
    }
};

static struct EngramFetchGradCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const EngramFetchGradTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape commContextShape = {param.commContextShape, param.commContextShape};
    gert::StorageShape gradFetchedShape = {param.gradFetchedShape, param.gradFetchedShape};
    gert::StorageShape permShape = {param.permShape, param.permShape};
    gert::StorageShape sendCountsShape = {param.sendCountsShape, param.sendCountsShape};
    gert::StorageShape recvCountsShape = {param.recvCountsShape, param.recvCountsShape};
    gert::StorageShape recvLocalEntryShape = {param.recvLocalEntryShape, param.recvLocalEntryShape};
    gert::StorageShape gradUniqueShape = {param.gradUniqueShape, param.gradUniqueShape};
    gert::StorageShape uniqueLocalEntryShape = {param.uniqueLocalEntryShape, param.uniqueLocalEntryShape};
    gert::StorageShape numUniqueShape = {param.numUniqueShape, param.numUniqueShape};

    // 7 inputs: commContext, gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{commContextShape, param.commContextDtype, param.commContextFormat},
         {gradFetchedShape, param.gradFetchedDtype, param.gradFetchedFormat},
         {permShape, param.permDtype, ge::FORMAT_ND},
         {sendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
         {recvCountsShape, ge::DT_INT32, ge::FORMAT_ND},
         {recvLocalEntryShape, ge::DT_INT32, ge::FORMAT_ND},
         {numUniqueShape, ge::DT_INT32, ge::FORMAT_ND}});

    // 3 outputs: gradUniqueOut (same dtype as gradFetched), uniqueLocalEntryOut (int32), numUniqueOut (int32)
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{gradUniqueShape, param.gradUniqueDtype, ge::FORMAT_ND},
         {uniqueLocalEntryShape, ge::DT_INT32, ge::FORMAT_ND},
         {numUniqueShape, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"num_entries_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numEntriesPerRank)},
         {"comm_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commBufferSize)}});

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

TEST_P(EngramFetchGradArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

INSTANTIATE_TEST_CASE_P(EngramFetchGradTilingUT, EngramFetchGradArch35TilingTest, testing::ValuesIn(g_testCases));

} // namespace EngramFetchGradUT
