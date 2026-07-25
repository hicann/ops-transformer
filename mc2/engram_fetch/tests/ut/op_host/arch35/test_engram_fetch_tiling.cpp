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
 * \file test_engram_fetch_tiling.cpp
 * \brief engram_fetch 算子 host 侧 tiling UT
 *
 * 测试用例覆盖:
 * - 推理场景: 2 inputs + 1 output + 2 attrs (with_grad=0)
 * - 训练场景: 3 inputs + 6 outputs + 5 attrs (with_grad=1)
 * - 异常场景: dtype 不匹配、维度错误、attr 非法值
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

namespace EngramFetchUT {

static const std::string OP_NAME = "EngramFetch";

struct EngramFetchTestParam {
    std::string caseName;
    std::initializer_list<int64_t> commContextShape;
    ge::DataType commContextDtype;
    ge::Format commContextFormat;
    std::initializer_list<int64_t> indicesShape;
    ge::DataType indicesDtype;
    ge::Format indicesFormat;
    std::initializer_list<int64_t> localStorageAddrShape;
    ge::DataType localStorageAddrDtype;
    std::initializer_list<int64_t> fetchedShape;
    ge::DataType fetchedDtype;
    ge::Format fetchedFormat;
    std::initializer_list<int64_t> permOutShape;
    std::initializer_list<int64_t> sendCountsOutShape;
    std::initializer_list<int64_t> recvCountsOutShape;
    std::initializer_list<int64_t> recvLocalEntryOutShape;
    std::initializer_list<int64_t> numRecvOutShape;
    int64_t hiddenSize;
    int64_t numEntriesPerRank;
    int64_t numMaxTokensPerRank;
    int64_t commBufferSize;
    int64_t withGrad;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
};

inline std::ostream &operator<<(std::ostream &os, const EngramFetchTestParam &param)
{
    return os << param.caseName;
}

static EngramFetchTestParam g_testCases[] = {
    {"infer_success_bf16_basic",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {16777216}},

    {"infer_success_fp16_basic",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {16777216}},

    {"infer_success_fp32_basic",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {16777216}},

    {"infer_success_large_tokens",
     {100 * 256},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {128},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {128, 256},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     256,
     100,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {16777216}},

    {"infer_success_single_token",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {1, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {16777216}},

    {"infer_success_empty_indices",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {0},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {0, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_SUCCESS,
     0UL,
     "",
     {16777216}},

    {"train_success_bf16_basic",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     {2},
     {2},
     {16},
     {1},
     512,
     4,
     8,
     4194304,
     1,
     "3510",
     ge::GRAPH_SUCCESS,
     1UL,
     "",
     {}},

    {"train_success_fp16_basic",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT64,
     {8, 512},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {8},
     {2},
     {2},
     {16},
     {1},
     512,
     4,
     8,
     4194304,
     1,
     "3510",
     ge::GRAPH_SUCCESS,
     1UL,
     "",
     {}},

    {"train_success_fp32_basic",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT64,
     {8, 512},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8},
     {2},
     {2},
     {16},
     {1},
     512,
     4,
     8,
     4194304,
     1,
     "3510",
     ge::GRAPH_SUCCESS,
     1UL,
     "",
     {}},

    {"train_success_large_tokens",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {128},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT64,
     {128, 256},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {128},
     {2},
     {2},
     {256},
     {1},
     256,
     100,
     128,
     4194304,
     1,
     "3510",
     ge::GRAPH_SUCCESS,
     1UL,
     "",
     {}},

    {"train_success_single_token",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT64,
     {1, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     {2},
     {2},
     {2},
     {1},
     512,
     4,
     1,
     4194304,
     1,
     "3510",
     ge::GRAPH_SUCCESS,
     1UL,
     "",
     {}},

    {"fail_commContext_dtype_bf16",
     {4 * 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_indices_dtype_float16",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_fetched_dtype_int32",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_indices_2d",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {2, 4},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_fetched_1d",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {4096},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_fetched_dim0_mismatch",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {16, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_hidden_size_zero",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     0,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_hidden_size_not_aligned",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 100},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     100,
     4,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_num_entries_negative",
     {4 * 512},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {},
     {},
     {},
     {},
     {},
     512,
     -1,
     0,
     0,
     0,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_train_localStorageAddr_dtype_int32",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT32,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     {2},
     {2},
     {16},
     {1},
     512,
     4,
     8,
     4194304,
     1,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},

    {"fail_train_localStorageAddr_wrong_shape",
     {6146},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {2},
     ge::DT_INT64,
     {8, 512},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {8},
     {2},
     {2},
     {16},
     {1},
     512,
     4,
     8,
     4194304,
     1,
     "3510",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {}},
};

class EngramFetchArch35TilingTest : public testing::TestWithParam<EngramFetchTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramFetchArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramFetchArch35TilingTest TearDown." << std::endl;
    }
};

static struct EngramFetchCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const EngramFetchTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape commContextShape = {param.commContextShape, param.commContextShape};
    gert::StorageShape indicesShape = {param.indicesShape, param.indicesShape};
    gert::StorageShape fetchedShape = {param.fetchedShape, param.fetchedShape};

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_;
    inputTensorDesc_.push_back({commContextShape, param.commContextDtype, param.commContextFormat});
    inputTensorDesc_.push_back({indicesShape, param.indicesDtype, param.indicesFormat});
    if (param.localStorageAddrShape.size() > 0) {
        gert::StorageShape localStorageAddrShape = {param.localStorageAddrShape, param.localStorageAddrShape};
        inputTensorDesc_.push_back({localStorageAddrShape, param.localStorageAddrDtype, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_;
    outputTensorDesc_.push_back({fetchedShape, param.fetchedDtype, param.fetchedFormat});
    if (param.permOutShape.size() > 0) {
        gert::StorageShape permOutShape = {param.permOutShape, param.permOutShape};
        outputTensorDesc_.push_back({permOutShape, ge::DT_INT32, ge::FORMAT_ND});
    }
    if (param.sendCountsOutShape.size() > 0) {
        gert::StorageShape sendCountsOutShape = {param.sendCountsOutShape, param.sendCountsOutShape};
        outputTensorDesc_.push_back({sendCountsOutShape, ge::DT_INT32, ge::FORMAT_ND});
    }
    if (param.recvCountsOutShape.size() > 0) {
        gert::StorageShape recvCountsOutShape = {param.recvCountsOutShape, param.recvCountsOutShape};
        outputTensorDesc_.push_back({recvCountsOutShape, ge::DT_INT32, ge::FORMAT_ND});
    }
    if (param.recvLocalEntryOutShape.size() > 0) {
        gert::StorageShape recvLocalEntryOutShape = {param.recvLocalEntryOutShape, param.recvLocalEntryOutShape};
        outputTensorDesc_.push_back({recvLocalEntryOutShape, ge::DT_INT32, ge::FORMAT_ND});
    }
    if (param.numRecvOutShape.size() > 0) {
        gert::StorageShape numRecvOutShape = {param.numRecvOutShape, param.numRecvOutShape};
        outputTensorDesc_.push_back({numRecvOutShape, ge::DT_INT32, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back({"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hiddenSize)});
    attrs_.push_back(
        {"num_entries_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numEntriesPerRank)});
    if (param.withGrad != 0) {
        attrs_.push_back(
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.numMaxTokensPerRank)});
        attrs_.push_back({"comm_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commBufferSize)});
        attrs_.push_back({"with_grad", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.withGrad)});
    }

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

TEST_P(EngramFetchArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

INSTANTIATE_TEST_CASE_P(EngramFetchTilingUT, EngramFetchArch35TilingTest, testing::ValuesIn(g_testCases));

} // namespace EngramFetchUT
