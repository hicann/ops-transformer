/* *
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
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AlltoAllvQuantGroupedMatMulUT {
struct TestParams {
    std::string test_name{};
    std::vector<std::pair<std::string, std::string>> tiling_params_str_pair{};
    std::vector<std::pair<std::string, std::vector<int64_t>>> tiling_params_vec_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_input_dtypes_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_output_dtypes_pair{};
    ge::graphStatus status;
};

struct TilingParamss {
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
    bool permute_out_flag{false};
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
    gert::StorageShape permute_out_shape;
};

struct TilingDTypes {
    std::vector<ge::DataType> input_dtypes{ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT,    ge::DT_FLOAT,
                                           ge::DT_INT64,    ge::DT_INT64,    ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                                           ge::DT_FLOAT,    ge::DT_FLOAT};
    std::vector<ge::DataType> output_dtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_HIFLOAT8};
};

class AlltoAllvQuantGroupedMatMulInfershape : public testing::TestWithParam<TestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMulInfershape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvQuantGroupedMatMulInfershape Test TearDown" << std::endl;
    }
};

std::unordered_map<std::string, std::function<void(TilingParamss &tiling_params, const std::string &value_str)>>
    infershape_params_str_handlers = {
        {"BSK",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.BSK = std::stoi(value_str); }},
        {"BS",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.BS = std::stoi(value_str); }},
        {"K",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.K = std::stoi(value_str); }},
        {"H1",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.H1 = std::stoi(value_str); }},
        {"H2",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.H2 = std::stoi(value_str); }},
        {"A",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.A = std::stoi(value_str); }},
        {"N1",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.N1 = std::stoi(value_str); }},
        {"N2",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.N2 = std::stoi(value_str); }},
        {"ep_world_size", [](TilingParamss &tiling_params,
                             const std::string &value_str) { tiling_params.ep_world_size = std::stoi(value_str); }},
        {"e",
         [](TilingParamss &tiling_params, const std::string &value_str) { tiling_params.e = std::stoi(value_str); }},
        {"gmm_weight_dim1", [](TilingParamss &tiling_params,
                               const std::string &value_str) { tiling_params.gmm_weight_dim1 = std::stoi(value_str); }},
        {"y_dim1", [](TilingParamss &tiling_params,
                      const std::string &value_str) { tiling_params.y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParamss &tiling_params,
                              const std::string &value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight",
         [](TilingParamss &tiling_params, const std::string &value_str) {
             tiling_params.trans_gmm_weight = value_str == "true";
         }},
        {"trans_mm_weight", [](TilingParamss &tiling_params,
                               const std::string &value_str) { tiling_params.trans_mm_weight = value_str == "true"; }},
        {"permute_out_flag", [](TilingParamss &tiling_params, const std::string &value_str) {
             tiling_params.permute_out_flag = value_str == "true";
         }}};

std::unordered_map<std::string, std::function<void(TilingParamss &tiling_params, const std::vector<int64_t> value_vec)>>
    infershape_params_vec_handlers = {
        {"send_counts", [](TilingParamss &tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParamss &tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

TEST_P(AlltoAllvQuantGroupedMatMulInfershape, inferdatatype_test)
{
    auto test_param = GetParam();
    int64_t input_num{10};
    int64_t output_num{3};
    auto tiling_param = TilingParamss{};
    auto tiling_dtypes = TilingDTypes{};

    for (auto &kv : test_param.tiling_input_dtypes_pair) {
        if (kv.first >= 0 && kv.first < tiling_dtypes.input_dtypes.size()) {
            tiling_dtypes.input_dtypes[kv.first] = kv.second;
        }
    }
    for (auto &kv : test_param.tiling_output_dtypes_pair) {
        if (kv.first >= 0 && kv.first < tiling_dtypes.output_dtypes.size()) {
            tiling_dtypes.output_dtypes[kv.first] = kv.second;
        }
    }

    std::vector<void *> input_dtypes_ptrs(input_num);
    for (int64_t i = 0; i < input_num; i++) {
        input_dtypes_ptrs[i] = &tiling_dtypes.input_dtypes[i];
    }
    std::vector<void *> output_dtypes_ptrs(output_num);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(input_num, output_num)
            .InputDataTypes(input_dtypes_ptrs)
            .OutputDataTypes(output_dtypes_ptrs)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_param.group)},
                 {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_param.ep_world_size)},
                 {"send_counts",
                  Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_param.send_counts)},
                 {"recv_counts",
                  Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_param.recv_counts)},
                 {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_param.trans_gmm_weight)},
                 {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_param.trans_mm_weight)},
                 {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_param.permute_out_flag)},
                 {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
            .Build();
    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("AlltoAllvQuantGroupedMatMul")->infer_datatype;

    /* do infershape */
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_P(AlltoAllvQuantGroupedMatMulInfershape, infershape_test)
{
    auto test_param = GetParam();
    auto tiling_params = TilingParamss{};
    for (auto &kv : test_param.tiling_params_str_pair) {
        if (infershape_params_str_handlers.count(kv.first) != 0) {
            infershape_params_str_handlers[kv.first](tiling_params, kv.second);
        }
    }
    for (auto &kv : test_param.tiling_params_vec_pair) {
        if (infershape_params_vec_handlers.count(kv.first) != 0) {
            infershape_params_vec_handlers[kv.first](tiling_params, kv.second);
        }
    }
    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllvQuantGroupedMatMul",
        {{{{tiling_params.BSK, tiling_params.H1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tiling_params.BS, tiling_params.H2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.mm_weight_dim0, tiling_params.N2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.send_counts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.recv_counts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_gmm_weight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_mm_weight)},
         {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

static TestParams test_params[] = {{"Test_sample", {{"permute_out_flag", "true"}}, {}, {}, {}, ge::GRAPH_SUCCESS}};

INSTANTIATE_TEST_SUITE_P(AlltoAllvQuantGroupedMatMul, AlltoAllvQuantGroupedMatMulInfershape,
                         testing::ValuesIn(test_params),
                         [](const testing::TestParamInfo<AlltoAllvQuantGroupedMatMulInfershape::ParamType> &info) {
                             return info.param.test_name;
                         });

// Negative infershape tests for wrong dims
TEST_F(AlltoAllvQuantGroupedMatMulInfershape, gmmx_wrong_dim_3d)
{
    auto tiling_params = TilingParamss{};
    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllvQuantGroupedMatMul",
        {{{{tiling_params.BSK, tiling_params.H1, 1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tiling_params.BS, tiling_params.H2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.mm_weight_dim0, tiling_params.N2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.send_counts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.recv_counts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_gmm_weight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_mm_weight)},
         {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
}

TEST_F(AlltoAllvQuantGroupedMatMulInfershape, gmmweight_wrong_dim_2d)
{
    auto tiling_params = TilingParamss{};
    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllvQuantGroupedMatMul",
        {{{{tiling_params.BSK, tiling_params.H1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.e, tiling_params.N1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tiling_params.BS, tiling_params.H2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.mm_weight_dim0, tiling_params.N2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.send_counts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.recv_counts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_gmm_weight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_mm_weight)},
         {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
}

TEST_F(AlltoAllvQuantGroupedMatMulInfershape, mmx_wrong_dim_3d)
{
    auto tiling_params = TilingParamss{};
    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllvQuantGroupedMatMul",
        {{{{tiling_params.BSK, tiling_params.H1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tiling_params.BS, tiling_params.H2, 1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.mm_weight_dim0, tiling_params.N2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.send_counts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.recv_counts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_gmm_weight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_mm_weight)},
         {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
}

TEST_F(AlltoAllvQuantGroupedMatMulInfershape, mmweight_wrong_dim_3d)
{
    auto tiling_params = TilingParamss{};
    gert::InfershapeContextPara infershapeContextPara(
        "AlltoAllvQuantGroupedMatMul",
        {{{{tiling_params.BSK, tiling_params.H1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{{tiling_params.BS, tiling_params.H2}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{tiling_params.mm_weight_dim0, tiling_params.N2, 1}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
         {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.send_counts)},
         {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tiling_params.recv_counts)},
         {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_gmm_weight)},
         {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.trans_mm_weight)},
         {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)},
         {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, {});
}
} // namespace AlltoAllvQuantGroupedMatMulUT
