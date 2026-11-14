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
#include "infer_shape_case_executor.h"

namespace {
// input, weight, slot_mapping, output_cache, act_seq_len(opt), block_table(opt)
const std::vector<uint32_t> kInputIrInstance{1, 1, 1, 1, 0, 0};
const std::vector<uint32_t> kOutputIrInstance{1};
} // namespace

class NsaCompressWithCacheProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressWithCacheProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressWithCacheProto TearDown" << std::endl;
    }
};

TEST_F(NsaCompressWithCacheProto, NsaCompressWithCache_infershape_A1_fp16)
{
    gert::InfershapeContextPara para("NsaCompressWithCache",
                                     {
                                         {{{16, 128, 32, 128}, {16, 128, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{512, 32, 128}, {512, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"act_seq_len_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"page_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {{512, 32, 128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

TEST_F(NsaCompressWithCacheProto, NsaCompressWithCache_infershape_A2_bf16)
{
    gert::InfershapeContextPara para("NsaCompressWithCache",
                                     {
                                         {{{16, 128, 32, 128}, {16, 128, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{32, 32}, {32, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{256, 32, 128}, {256, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"act_seq_len_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"page_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {{256, 32, 128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}
