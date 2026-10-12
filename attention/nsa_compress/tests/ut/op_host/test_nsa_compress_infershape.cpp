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
// actSeqLenOptional is mandatory for infershape (index 2).
const std::vector<uint32_t> kInputIrInstance{1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1};

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
} // namespace

class NsaCompressProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressProto TearDown" << std::endl;
    }
};

// A1: fp16 TND, small prefix-sum actSeqLen
TEST_F(NsaCompressProto, NsaCompress_infershape_A1_fp16)
{
    int64_t actSeqLen[] = {16, 32, 48};
    gert::InfershapeContextPara para("NsaCompress",
                                     {
                                         {{{48, 32, 16}, {48, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{16, 32}, {16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {{3, 32, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A2: bf16
TEST_F(NsaCompressProto, NsaCompress_infershape_A2_bf16)
{
    int64_t actSeqLen[] = {16, 32, 48};
    gert::InfershapeContextPara para("NsaCompress",
                                     {
                                         {{{48, 32, 16}, {48, 32, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{16, 32}, {16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {{3, 32, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A3: larger batch prefix sum (48 batches, T=4800)
TEST_F(NsaCompressProto, NsaCompress_infershape_A3_large_batch)
{
    int64_t actSeqLen[] = {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
                           1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                           2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                           3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800};
    gert::InfershapeContextPara para("NsaCompress",
                                     {
                                         {{{4800, 1, 32}, {4800, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {{288, 1, 32}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A4: per-batch seq shorter than compressBlockSize -> zero output tokens
TEST_F(NsaCompressProto, NsaCompress_infershape_A4_zero_output_tokens)
{
    int64_t actSeqLen[] = {16, 32, 48};
    gert::InfershapeContextPara para("NsaCompress",
                                     {
                                         {{{48, 32, 16}, {48, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {{0, 32, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}
