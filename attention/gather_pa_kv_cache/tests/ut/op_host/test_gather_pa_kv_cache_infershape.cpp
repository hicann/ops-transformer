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
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GatherPaKvCacheProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GatherPaKvCacheProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GatherPaKvCacheProto TearDown" << std::endl;
    }
};

TEST_F(GatherPaKvCacheProto, gather_pa_kv_cache_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GatherPaKvCache",
        {
            {{{30, 101, 128, 32}, {30, 101, 128, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // keyCache
            {{{30, 101, 128, 32}, {30, 101, 128, 32}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // valueCache
            {{{16, 12}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND},                            // blocktables
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                                    // seqLens
            {{{18933, 3232}, {18933, 3232}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},           // key
            {{{18933, 3232}, {18933, 3232}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},           // value
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                                    // seqOffset

        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // key
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_NZ")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{18933, 3232}, {18933, 3232}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GatherPaKvCacheProto, gather_pa_kv_cache_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GatherPaKvCache",
        {
            {{{128, 128, 16, 144}, {128, 128, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // keyCache
            {{{128, 128, 16, 128}, {128, 128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // valueCache
            {{{16, 12}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND},                      // blocktables
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                              // seqLens
            {{{9547, 16, 144}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND},         // key
            {{{9547, 16, 128}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},         // value
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                              // seqOffset

        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{9547, 16, 144}, {9547, 16, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GatherPaKvCacheProto, gather_pa_kv_cache_infershape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GatherPaKvCache",
        {
            {{{-1, -1, -1, -1}, {128, 128, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // keyCache
            {{{-1, -1, -1, -1}, {128, 128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // valueCache
            {{{-1, -1}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND},                   // blocktables
            {{{1 - 1}, {16}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqLens
            {{{-1, -1, -1}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND},         // key
            {{{-1, -1, -1}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},         // value
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},                           // seqOffset

        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1}, {-1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GatherPaKvCacheProto, gather_pa_kv_cache_infershape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GatherPaKvCache",
        {
            {{{-2}, {128, 128, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND}, // keyCache
            {{{-2}, {128, 128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // valueCache
            {{{-2}, {16, 12}}, ge::DT_INT32, ge::FORMAT_ND},           // blocktables
            {{{-2}, {16}}, ge::DT_INT32, ge::FORMAT_ND},               // seqLens
            {{{-2}, {9547, 16, 144}}, ge::DT_INT8, ge::FORMAT_ND},     // key
            {{{-2}, {9547, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},     // value
            {{{-2}, {16}}, ge::DT_INT32, ge::FORMAT_ND},               // seqOffset

        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // key
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}, // value
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
            {"is_seq_lens_cumsum", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(GatherPaKvCacheProto, gather_pa_kv_cache_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("GatherPaKvCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputDtype = ge::DT_INT8;
    ge::DataType inputReDtype = ge::DT_INT32;
    ge::DataType outDtype = ge::DT_INT8;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(7)
                              .NodeIoNum(7, 2)
                              .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                              .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputDtype, &inputDtype, &inputReDtype, &inputReDtype, &inputDtype,
                                               &inputDtype, &inputReDtype})
                              .OutputDataTypes({&outDtype, &outDtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), outDtype);
    EXPECT_EQ(context->GetOutputDataType(1), outDtype);
}
