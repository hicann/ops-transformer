/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.h"

#include <iostream>
#include <gtest/gtest.h>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace AlltoAllvGroupedMatMulUT {
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
    uint64_t commOut;
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t totalUbSize{196608};
    uint64_t gmm_weight_dim1{7168};
    uint64_t gmm_y_dim1{4096};
    uint64_t mm_weight_dim0{7168};
    bool trans_gmm_weight{false};
    bool trans_mm_weight{false};
    bool permute_out_flag{false};
    bool is_Need_MM{true};
    std::string group{"group"};
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

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
        {"gmm_y_dim1",
         [](TilingParams& tiling_params, const string& value_str) { tiling_params.gmm_y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParams& tiling_params,
                              const string& value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight", [](TilingParams& tiling_params,
                                const string& value_str) { tiling_params.trans_gmm_weight = value_str == "true"; }},
        {"trans_mm_weight", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.trans_mm_weight = value_str == "true"; }},
        {"permute_out_flag", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.permute_out_flag = value_str == "true";
         }},
        {"is_Need_MM", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.is_Need_MM = value_str == "true";
         }}
        };

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
    tiling_params_vec_handlers = {
        {"send_counts", [](TilingParams& tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParams& tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

bool has_any_target_key(
    const std::vector<std::pair<std::string, std::string>>& params,
    const std::vector<std::string>& targets
) {
    return std::any_of(
        params.begin(),
        params.end(),
        [&targets](const auto& p) {
            return std::find(targets.begin(), targets.end(), p.first) != targets.end();
        }
    );
}

// 提取：初始化 tiling_params
void InitializeTilingParams(
    const TestParam& test_param,
    TilingParams& tiling_params
) {
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
}

std::unique_ptr<gert::TilingContextPara::TensorDescription> CreateTensorShape(
    gert::StorageShape shape,
    ge::DataType dtype,
    ge::Format format
) {
    return std::unique_ptr<gert::TilingContextPara::TensorDescription>(
        new gert::TilingContextPara::TensorDescription(
            shape,  // 这里用大括号构造
            dtype,
            format
        )
    );
}

std::vector<gert::TilingContextPara::TensorDescription> CreateInputTensors(
    const TilingParams& tiling_params,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_x_shape,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_weight_shape
) {
    return {
        {{{tiling_params.BSK, tiling_params.H1}, {tiling_params.BSK, tiling_params.H1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, // placeholder
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, // placeholder
        *mm_x_shape,
        *mm_weight_shape, 
    };
}

std::vector<gert::TilingContextPara::TensorDescription> CreateOutputTensors(
    const TilingParams& tiling_params,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_y_shape
) {
    return {
        {{{tiling_params.A, tiling_params.gmm_y_dim1}, {tiling_params.A, tiling_params.gmm_y_dim1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
        *mm_y_shape,
        {{{tiling_params.A, tiling_params.H1}, {tiling_params.A, tiling_params.H1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
    };
}

std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> CreateAttrs(
    const TestParam& test_param,
    const TilingParams& tiling_params
) {
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)}
    };
}

class AlltoAllvGroupedMatMulTiling : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTiling Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTiling Test TearDown" << std::endl;
    }

public:
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

TEST_P(AlltoAllvGroupedMatMulTiling, shape_size)
{
    auto test_param = GetParam();

    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;

    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    TilingParams tiling_params;
    InitializeTilingParams(test_param, tiling_params);

    std::vector<std::string> targets = {"BS", "H2", "mm_weight_dim0", "N2"};

    auto mm_x_shape = CreateTensorShape({{tiling_params.BS, tiling_params.H2}, {tiling_params.BS, tiling_params.H2}}, 
                                        ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mm_weight_shape = CreateTensorShape({{tiling_params.mm_weight_dim0, tiling_params.N2}, 
                                              {tiling_params.mm_weight_dim0, tiling_params.N2}}, 
                                              ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mm_y_shape = CreateTensorShape({{tiling_params.BS, tiling_params.N2}, {tiling_params.BS, tiling_params.N2}},
                                         ge::DT_FLOAT16, ge::FORMAT_ND);

    if (!(has_any_target_key(test_param.tiling_params_str_pair, targets) || tiling_params.is_Need_MM == false)) {
        mm_x_shape->shape_ = {};
        mm_weight_shape->shape_ = {};
        mm_y_shape->shape_ = {};
    }

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        CreateInputTensors(tiling_params, mm_x_shape, mm_weight_shape),
        CreateOutputTensors(tiling_params, mm_y_shape),
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)}
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize
    );

    if (test_param.status == ge::GRAPH_FAILED) {
        ExecuteTestCase(tilingContextPara);
    } else {
        uint64_t expectTilingKey = 1000UL;
        if (test_param.test_name == "Test_no_MM") {
            expectTilingKey = 1100UL;
        }
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
    }
}

static TestParam test_params[] = {
    {"Test_gmmWeight_size", {{"e", "64"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_ep_world_size", {{"ep_world_size", "4"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_e", {{"e", "64"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_e_multi_ep",
     {{"ep_world_size", "16"}, {"e", "32"}, {"permute_out_flag", "true"}},
     {{"send_counts", std::vector<int64_t>(512, 128)}, {"recv_counts", std::vector<int64_t>(512, 128)}},
     {},
     ge::GRAPH_FAILED},
    {"Test_send_counts_size",
     {{"ep_world_size", "16"}, {"e", "32"}, {"permute_out_flag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    {"Test_BSK_1", {{"BSK", "52428800"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_BS_1", {{"BS", "52428800"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H1", {{"H1", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H2", {{"H2", "12289"}, {"mm_weight_dim0", "12289"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N1", {{"N1", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N2", {{"N2", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_1", {{"H1", "7168"}, {"gmm_weight_dim1", "7169"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_3", {{"H2", "7168"}, {"mm_weight_dim0", "7169"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_4", {{"H1", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_0",
     {{"BSK", "16386"}, {"permute_out_flag", "true"}},
     {{"send_counts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_FAILED},
    {"Test_recv_counts_0",
     {{"A", "16386"}, {"BS", "8193"}, {"permute_out_flag", "true"}},
     {{"recv_counts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_FAILED},
    {"Test_recv_counts_1",
     {{"A", "16386"}, {"BS", "8193"}, {"permute_out_flag", "true"}},
     {{"recv_counts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
    {"Test_no_MM", {{"permute_out_flag", "true"}, {"is_Need_MM", "false"}}, {}, {}, ge::GRAPH_SUCCESS}
};


INSTANTIATE_TEST_SUITE_P(AlltoAllvGroupedMatMul, AlltoAllvGroupedMatMulTiling,
                         testing::ValuesIn(test_params),
                         [](const testing::TestParamInfo<AlltoAllvGroupedMatMulTiling::ParamType>& info) {
                             return info.param.test_name;
                         });

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_H_4)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4096, 7169}, {4096, 7169}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_A_1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4097, 7168}, {4097, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_BS_1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },    
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },    
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_2)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{7168, 4096}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },    
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_3)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_5)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2, 2048, 7168}, {2, 2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},  
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_6)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 7168, 64}, {2, 7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_7)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 7168, 64}, {2, 7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},     
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_dim_10)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}

TEST_F(AlltoAllvGroupedMatMulTiling, allto_allv_grouped_matmul_tiling_test_transMmWeight_1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul",
    {
        {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        
    },
    {
        {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
    },
    &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    ExecuteTestCase(tilingContextPara);
}
} // allto_allv_grouped_mat_mul_ut