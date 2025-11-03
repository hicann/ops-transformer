/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/grouped_mat_mul_allto_allv_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace GroupedMatMulAlltoAllvUT {
struct TestParam {
    string test_name{};
    std::vector<std::pair<string, string>> tiling_params_str_pair{};
    std::vector<std::pair<string, std::vector<int64_t>>> tiling_params_vec_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_dTypes_pair{};
    ge::graphStatus status;
};

struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{64};
    uint64_t ep_world_size{8};
    uint64_t e{4};
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t gmm_weight_dim1{7168};
    uint64_t y_dim1{4096};
    uint64_t mm_weight_dim0{7168};
    bool trans_gmm_weight{false};
    bool trans_mm_weight{false};
    std::string group{"group"};
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>> 
        tiling_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BSK = std::stoi(value_str); }},
        {"BS", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BS = std::stoi(value_str); }},
        {"K", [](TilingParams& tiling_params, const string& value_str) { tiling_params.K = std::stoi(value_str); }},
        {"H1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H1 = std::stoi(value_str); }},
        {"H2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H2 = std::stoi(value_str); }},
        {"A", [](TilingParams& tiling_params, const string& value_str) { tiling_params.A = std::stoi(value_str); }},
        {"N1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N1 = std::stoi(value_str); }},
        {"N2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N2 = std::stoi(value_str); }},
        {"ep_world_size", [](TilingParams& tiling_params,
                             const string& value_str) { tiling_params.ep_world_size = std::stoi(value_str); }},
        {"e", [](TilingParams& tiling_params, const string& value_str) { tiling_params.e = std::stoi(value_str); }},
        {"gmm_weight_dim1", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.gmm_weight_dim1 = std::stoi(value_str); }},
        {"y_dim1",
         [](TilingParams& tiling_params, const string& value_str) { tiling_params.y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParams& tiling_params,
                              const string& value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight", [](TilingParams& tiling_params,
                                const string& value_str) { tiling_params.trans_gmm_weight = value_str == "true"; }},
        {"trans_mm_weight", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.trans_mm_weight = value_str == "true";
         }}};

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
        tiling_params_vec_handlers = {
        {"send_counts", [](TilingParams& tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParams& tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

class GroupedMatMulAlltoAllvTiling : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvTiling Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvTiling Test TearDown" << std::endl;
    }
};

TEST_P(GroupedMatMulAlltoAllvTiling, shape_size)
{
    auto test_param = GetParam();
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    auto tiling_params = TilingParams{};
    for (auto& kv : test_param.tiling_params_str_pair) {
        if (tiling_params_str_handlers.count(kv.first) != 0) {
            tiling_params_str_handlers[kv.first](tiling_params, kv.second);
        }
    }
    for (auto& kv : test_param.tiling_params_vec_pair) {
        if (tiling_params_vec_handlers.count(kv.first) != 0) {
            tiling_params_vec_handlers[kv.first](tiling_params, kv.second);
        }
    }

    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {   
            {{{tiling_params.A, tiling_params.H1}, {tiling_params.A, tiling_params.H1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.BS, tiling_params.H2}, {tiling_params.BS, tiling_params.H2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.mm_weight_dim0, tiling_params.N2}, {tiling_params.mm_weight_dim0, tiling_params.N2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
        },
        {
            {{{tiling_params.BSK, tiling_params.y_dim1}, {tiling_params.BSK, tiling_params.y_dim1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.BS, tiling_params.N2}, {tiling_params.BS, tiling_params.N2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    if (test_param.status == ge::GRAPH_FAILED){
        ExecuteTestCase(tilingContextPara);
    } else {
        uint64_t expectTilingKey = 1UL;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey); 
    }
}

static TestParam test_params[] = {
    {"Test_sample", {}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_BSK_1", {{"BSK", "52428800"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_BS_1", {{"BS", "52428800"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H1", {{"H1", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H2", {{"H2", "12889"}, {"mm_weight_dim0", "12889"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N1", {{"N1", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N2", {{"N2", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_ep_world_size", {{"ep_world_size", "4"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_size", {{"ep_world_size", "16"}, {"e", "32"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_1", {{"H1", "7168"}, {"gmm_weight_dim1", "7169"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_3", {{"H2", "7168"}, {"mm_weight_dim0", "7169"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_0",
     {{"A", "16386"}},
     {{"send_counts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_recv_counts_0",
     {{"BSK", "16386"}, {"BS", "8193"}},
     {{"recv_counts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_recv_counts_1",
     {{"BSK", "16386"}},
     {{"recv_counts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(GroupedMatMulAlltoAllv, GroupedMatMulAlltoAllvTiling, testing::ValuesIn(test_params), [](const testing::TestParamInfo<GroupedMatMulAlltoAllvTiling::ParamType>& info) {
                             return info.param.test_name;
                         });

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_dim_1)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_dim_2)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 4096}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_dim_3)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_dim_4)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2048, 7168}, {2, 2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 6464}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_dim_5)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 7168, 6464}, {2, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_dim_6)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 6464}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2048, 64}, {2, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(GroupedMatMulAlltoAllvTiling, grouped_matmul_allto_allv_tiling_test_transMmWeight_invalid)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    ExecuteTestCase(tilingContextPara);
}
} // grouped_mat_mul_allto_allv_ut