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

class QuantCompressorProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QuantCompressorProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantCompressorProto TearDown" << std::endl; }
};

// BSH layout: x=[B,S,H]=[1,128,4096], wkv=[coff*D,H]=[256,4096], cmp_ratio=4, coff=2.
// Expected cmp_kv output shape: [B, ceil(S/cmp_ratio), D] = [1, 32, 128].
TEST_F(QuantCompressorProto, quant_compressor_infershape_bsh_c4li)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantCompressor",
        {
            {{{1, 128, 4096}, {1, 128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // x
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},       // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},      // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},                // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // state_block_table
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // cu_seqlens
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // seqused
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                              // start_pos
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // cmp_kv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TH layout: x=[T,H]=[128,4096], cu_seqlens=[B+1]=[2], cmp_ratio=4, coff=2.
// Expected cmp_kv output shape: [Sr, D] where Sr=min(T, T//cmp_ratio+B) = min(128, 32+1)=33, D=128.
TEST_F(QuantCompressorProto, quant_compressor_infershape_th)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantCompressor",
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // x [T,H]
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // wkv
            {{{256, 4096}, {256, 4096}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},  // wgate
            {{{1, 128, 512}, {1, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
            {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},           // ape
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // x_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // wkv_descale
            {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // wgate_descale
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},               // state_block_table
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                     // cu_seqlens [B+1]
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                         // seqused
            {{}, ge::DT_UNDEFINED, ge::FORMAT_ND},                         // start_pos
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // cmp_kv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache out
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{33, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantCompressorProto, quant_compressor_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QuantCompressor")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType dtHif8 = ge::DT_HIFLOAT8;
    ge::DataType dtFp32 = ge::DT_FLOAT;
    ge::DataType dtInt32 = ge::DT_INT32;
    ge::DataType dtBf16 = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(12)
                              .NodeIoNum(12, 2)
                              .NodeInputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(9, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(10, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(11, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&dtHif8, &dtHif8, &dtHif8, &dtFp32, &dtFp32, &dtFp32, &dtFp32, &dtFp32,
                                               &dtInt32, &dtInt32, &dtInt32, &dtInt32})
                              .OutputDataTypes({&dtBf16, &dtFp32})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
