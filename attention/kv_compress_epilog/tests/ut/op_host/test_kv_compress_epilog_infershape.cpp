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

class KvCompressEpilogProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvCompressEpilogProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvCompressEpilogProto TearDown" << std::endl;
    }
};

TEST_F(KvCompressEpilogProto, kv_compress_epilog_infershape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara("KvCompressEpilog",
                                                      {
                                                          {{{128, 16, 1, 384}, {128, 16, 1, 384}}, ge::DT_UINT8, ge::FORMAT_ND}, // cache 4D [blockNum,blockSize,1,headDim]
                                                          {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},              // x
                                                          {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},                       // slotMapping
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}, // cacheOut (in-place)
                                                      },
                                                      {
                                                          {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                                          {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                          {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(KvCompressEpilogProto, kv_compress_epilog_infershape_hif8_fp4_mode2)
{
    // mode2(rope hifloat8 + nope FLOAT4_E2M1): 输出 cache 仍与输入 cache 同 shape/dtype, 与 quant_mode 无关。
    gert::InfershapeContextPara infershapeContextPara("KvCompressEpilog",
                                                      {
                                                          {{{128, 16, 1, 256}, {128, 16, 1, 256}}, ge::DT_UINT8, ge::FORMAT_ND}, // cache 4D
                                                          {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},              // x (d=256 -> nope=192)
                                                          {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},                       // slotMapping
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}, // cacheOut (in-place)
                                                      },
                                                      {
                                                          {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                                          {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                          {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                          {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(KvCompressEpilogProto, kv_compress_epilog_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("KvCompressEpilog",
                                                      {
                                                          {{{-2}, {128, 16, 1, 384}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                          {{{-2}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{-2}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"quant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                                          {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"round_scale", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                                          {"x_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(KvCompressEpilogProto, kv_compress_epilog_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("KvCompressEpilog")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType inputCacheDtype = ge::DT_UINT8;
    ge::DataType inputXDtype = ge::DT_BF16;
    ge::DataType inputSlotDtype = ge::DT_INT32;
    ge::DataType outCacheDtype = ge::DT_UINT8;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .NodeInputTd(0, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputCacheDtype, &inputXDtype, &inputSlotDtype})
                              .OutputDataTypes({&outCacheDtype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
}
