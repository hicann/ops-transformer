/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class IndexerQuantCacheProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IndexerQuantCacheProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IndexerQuantCacheProto TearDown" << std::endl;
    }
};

TEST_F(IndexerQuantCacheProto, indexer_quant_cache_infershape_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IndexerQuantCache",
        {
            {{{2048, 1, 1, 128}, {2048, 1, 1, 128}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_ND}, // cache 4D [blockNum,blockSize,1,headDim]
            {{{2048, 1, 1, 1}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cacheScale 4D
            {{{1024, 128}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},                   // slotMapping
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // cacheOut
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},         // cacheScaleOut
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(IndexerQuantCacheProto, indexer_quant_cache_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IndexerQuantCache",
        {
            {{{-2}, {2048, 1, 1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // cache 4D
            {{{-2}, {2048, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},           // cacheScale 4D
            {{{-2}, {1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // x
            {{{-2}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},                    // slotMapping
        },
        {
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(IndexerQuantCacheProto, indexer_quant_cache_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("IndexerQuantCache")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputCacheDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType inputScaleDtype = ge::DT_FLOAT;
    ge::DataType inputXDtype = ge::DT_FLOAT16;
    ge::DataType inputSlotDtype = ge::DT_INT32;
    ge::DataType outCacheDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outScaleDtype = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 2)
                              .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputCacheDtype, &inputScaleDtype, &inputXDtype, &inputSlotDtype})
                              .OutputDataTypes({&outCacheDtype, &outScaleDtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
}
