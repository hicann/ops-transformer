/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_datatype_context_faker.h"
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace GroupedMatMulAlltoAllvUT {
struct TestParams {
    std::string test_name{};
    std::vector<std::pair<std::string, std::string>> tiling_params_str_pair{};
    std::vector<std::pair<std::string, std::vector<int64_t>>> tiling_params_vec_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_input_dtypes_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_output_dtypes_pair{};
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
    uint64_t N2{4096};
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

struct TilingShapes {
    gert::StorageShape gmm_x_shape;
    gert::StorageShape gmm_weight_shape;
    gert::StorageShape send_counts_shape;
    gert::StorageShape recv_counts_shape;
    gert::StorageShape mm_x_shape;
    gert::StorageShape mm_weight_shape;

    gert::StorageShape y_shape;
    gert::StorageShape mm_y_shape;
};

struct TilingDTypes {
    std::vector<ge::DataType> input_dtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT64,
                                           ge::DT_INT64,   ge::DT_FLOAT16, ge::DT_FLOAT16};
    std::vector<ge::DataType> output_dtypes{ge::DT_FLOAT16, ge::DT_FLOAT16};
};

class GroupedMatMulAlltoAllvInfershape : public testing::TestWithParam<TestParams>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvRuntimeProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvRuntimeProtoTest TearDown" << std::endl;
    }
};

std::unordered_map<std::string, std::function<void(TilingParams& tiling_params, const std::string& value_str)>>
    infershape_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.BSK = std::stoi(value_str); }},
        {"BS", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.BS = std::stoi(value_str); }},
        {"K", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.K = std::stoi(value_str); }},
        {"H1", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.H1 = std::stoi(value_str); }},
        {"H2", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.H2 = std::stoi(value_str); }},
        {"A", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.A = std::stoi(value_str); }},
        {"N1", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.N1 = std::stoi(value_str); }},
        {"N2", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.N2 = std::stoi(value_str); }},
        {"ep_world_size", [](TilingParams& tiling_params,
                             const std::string& value_str) { tiling_params.ep_world_size = std::stoi(value_str); }},
        {"e", [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.e = std::stoi(value_str); }},
        {"gmm_weight_dim1", [](TilingParams& tiling_params,
                               const std::string& value_str) { tiling_params.gmm_weight_dim1 = std::stoi(value_str); }},
        {"y_dim1",
         [](TilingParams& tiling_params, const std::string& value_str) { tiling_params.y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParams& tiling_params,
                              const std::string& value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight", [](TilingParams& tiling_params,
                                const std::string& value_str) { tiling_params.trans_gmm_weight = value_str == "true"; }},
        {"trans_mm_weight", [](TilingParams& tiling_params, const std::string& value_str) {
             tiling_params.trans_mm_weight = value_str == "true";
         }}};

std::unordered_map<std::string, std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
    infershape_params_vec_handlers = {
        {"send_counts", [](TilingParams& tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParams& tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

TEST_P(GroupedMatMulAlltoAllvInfershape, inferdatatype_test)
{
    auto test_param = GetParam();
    int64_t input_num{6};
    int64_t output_num{2};
    auto tiling_param = TilingParams{};
    auto tiling_dtypes = TilingDTypes{};
    
    for (auto& kv : test_param.tiling_input_dtypes_pair) {
        if (kv.first >= 0 && kv.first < tiling_dtypes.input_dtypes.size()) {
            tiling_dtypes.input_dtypes[kv.first] = kv.second;
        }
    }
    for (auto& kv : test_param.tiling_output_dtypes_pair) {
            if (kv.first >= 0 && kv.first < tiling_dtypes.output_dtypes.size()) {
                tiling_dtypes.output_dtypes[kv.first] = kv.second;
        }
    }
    
    std::vector<void*> input_dtypes_ptrs(input_num);
    for (int64_t i = 0; i < input_num; i++) {
        input_dtypes_ptrs[i] = &tiling_dtypes.input_dtypes[i];
    }
    std::vector<void*> output_dtypes_ptrs(output_num);

    auto contextHolder = gert::InferDataTypeContextFaker()
                .NodeIoNum(input_num, output_num)
                .InputDataTypes(input_dtypes_ptrs)
                .OutputDataTypes(output_dtypes_ptrs)
                .NodeAttrs({
                    {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_param.group)},
                    {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_param.ep_world_size)},
                    {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_param.send_counts)},
                    {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_param.recv_counts)},
                    {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_param.trans_gmm_weight)},
                    {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_param.trans_mm_weight)}})
                .Build();
    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAlltoAllv")->infer_datatype;

    /* do infershape */
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_P(GroupedMatMulAlltoAllvInfershape, infershape_test)
{
    auto test_param = GetParam();
    auto tiling_params = TilingParams{};
    for (auto& kv : test_param.tiling_params_str_pair) {
        if (infershape_params_str_handlers.count(kv.first) != 0) {
            infershape_params_str_handlers[kv.first](tiling_params, kv.second);
        }
    }
    for (auto& kv : test_param.tiling_params_vec_pair) {
        if (infershape_params_vec_handlers.count(kv.first) != 0) {
            infershape_params_vec_handlers[kv.first](tiling_params, kv.second);
        }
    }

    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatMulAlltoAllv",
        {   
            {{{tiling_params.A, tiling_params.H1}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.BS, tiling_params.H2}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tiling_params.mm_weight_dim0, tiling_params.N2}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.send_counts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.recv_counts)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_gmm_weight)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_mm_weight)}
        });
 
    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 4096}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); 
}

static TestParams test_paramss[] = {{"Test_sample", {}, {}, {}, {}, ge::GRAPH_SUCCESS},};

INSTANTIATE_TEST_SUITE_P(GroupedMatMulAlltoAllv, GroupedMatMulAlltoAllvInfershape, testing::ValuesIn(test_paramss), [](const testing::TestParamInfo<GroupedMatMulAlltoAllvInfershape::ParamType>& info) {
                             return info.param.test_name;
                         });
} // grouped_mat_mul_allto_allv_ut