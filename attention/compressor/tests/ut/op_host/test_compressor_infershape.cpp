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
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class CompressorInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CompressorInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CompressorInfershape TearDown" << std::endl;
    }
};

// ====================================================================
// BSH Layout Tests
// ====================================================================

// C4A: B=2, S=4, H=4096, D=512, coff=2, cmp_ratio=4
// Sr = ceil(4/4) = 1, output: [2, 1, 512]
TEST_F(CompressorInfershape, bsh_c4a_bf16)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         // input 0: x [B, S, H]
                                         {{{2, 4, 4096}, {2, 4, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         // input 1: wkv [coff*D, H] = [1024, 4096]
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         // input 2: wgate [coff*D, H] = [1024, 4096]
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         // input 3: state_cache [block_num, block_size, 2*coff*D]
                                         {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         // input 4: ape [cmp_ratio, coff*D] = [4, 1024]
                                         {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         // optional inputs (not used in BSH layout)
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // state_block_table
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // cu_seqlens
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // seqused
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // start_pos
                                     },
                                     {
                                         // output 0: cmp_kv (placeholder)
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         // attrs
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 512}, {}, {2, 1, 8, 512}, {2, 1, 8, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// C4A fp16: B=2, S=4, H=4096, D=512, coff=2, cmp_ratio=4
TEST_F(CompressorInfershape, bsh_c4a_fp16)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         {{{2, 4, 4096}, {2, 4, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 512}, {}, {2, 1, 8, 512}, {2, 1, 8, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// C4Li: B=2, S=4, H=2048, D=128, coff=2, cmp_ratio=4
// Sr = ceil(4/4) = 1, output: [2, 1, 128]
TEST_F(CompressorInfershape, bsh_c4li_bf16)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         {{{2, 4, 2048}, {2, 4, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{4, 128, 512}, {4, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 128}, {}, {2, 1, 8, 128}, {2, 1, 8, 128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// C128A: B=2, S=8, H=4096, D=512, coff=1, cmp_ratio=128
// Sr = ceil(8/128) = 1, output: [2, 1, 512]
TEST_F(CompressorInfershape, bsh_c128a_bf16)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{4, 128, 1024}, {4, 128, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{128, 512}, {128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 512}, {}, {2, 1, 128, 512}, {2, 1, 128, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BSH non-divisible: B=1, S=6, H=2048, D=128, coff=2, cmp_ratio=4
// Sr = ceil(6/4) = 2, output: [1, 2, 128]
TEST_F(CompressorInfershape, bsh_non_divisible)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         {{{1, 6, 2048}, {1, 6, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{2, 128, 512}, {2, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1, 6}, {1, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 128}, {}, {1, 2, 8, 128}, {1, 2, 8, 128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ====================================================================
// TH Layout Tests
// ====================================================================

// TH: T=8, H=128, B=2, cuSeqlens=[0,4,8]
// Sr = min(8, 8/4 + 3 - 1) = min(8, 4) = 4, output: [4, 512]
TEST_F(CompressorInfershape, th_c4a_bf16)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         // x [T, H] = [8, 4096]
                                         {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         // wkv [coff*D, H] = [1024, 4096]
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         // wgate [coff*D, H] = [1024, 4096]
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         // state_cache [block_num, block_size, 2*coff*D]
                                         {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         // ape [cmp_ratio, coff*D] = [4, 1024]
                                         {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         // optional inputs
                                         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND}, // state_block_table
                                                                                  // cu_seqlens [B+1] = [3]
                                         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // seqused
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // start_pos
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 512}, {}, {4, 8, 512}, {4, 8, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TH: T=16, H=4096, B=3, cuSeqlens=[0,4,10,16]
// Sr = min(16, 16/4 + 4 - 1) = min(16, 7) = 7, output: [7, 512]
TEST_F(CompressorInfershape, th_c4a_multi_batch)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         {{{16, 4096}, {16, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3, 16}, {3, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{7, 512}, {}, {7, 8, 512}, {7, 8, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ====================================================================
// Empty tensor (B=0 in BSH) — infershape should still succeed
// ====================================================================

TEST_F(CompressorInfershape, bsh_empty_batch)
{
    gert::InfershapeContextPara para("Compressor",
                                     {
                                         {{{0, 4, 4096}, {0, 4, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{4, 128, 2048}, {4, 128, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{0, 4}, {0, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // state_cache
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv

                                     },
                                     {
                                         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                         {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     });
    // B=0, Sr=ceil(4/4)=1, output: [0, 1, 512]
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 1, 512}, {}, {0, 1, 8, 512}, {0, 1, 8, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ====================================================================
// InferDataType — output dtype follows x
// ====================================================================

TEST_F(CompressorInfershape, inferdatatype_bf16)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("Compressor")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType dtBf16 = ge::DT_BF16;
    ge::DataType dtFp32 = ge::DT_FLOAT;
    ge::DataType dtInt32 = ge::DT_INT32;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(9)
                              .NodeIoNum(9, 2)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&dtBf16, &dtBf16, &dtBf16, &dtFp32, &dtFp32,
                                               &dtInt32, &dtInt32, &dtInt32, &dtInt32})
                              .OutputDataTypes({&dtBf16, &dtFp32})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    // cmp_kv output dtype should follow x (DT_BF16)
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
