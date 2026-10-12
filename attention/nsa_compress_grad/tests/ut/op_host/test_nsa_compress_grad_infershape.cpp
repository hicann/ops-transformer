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
const std::vector<uint32_t> kInputIrInstance{1, 1, 1, 0};
const std::vector<uint32_t> kInputIrInstanceWithActSeq{1, 1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1, 1};
} // namespace

class NsaCompressGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressGradProto TearDown" << std::endl;
    }
};

// A1: fp16 base case (legacy nsa_compress_grad_base)
TEST_F(NsaCompressGradProto, NsaCompressGrad_infershape_A1_fp16)
{
    gert::InfershapeContextPara para("NsaCompressGrad",
                                     {
                                         {{{13, 128, 192}, {13, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{272, 128, 192}, {272, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {272, 128, 192},
        {32, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A2: bf16
TEST_F(NsaCompressGradProto, NsaCompressGrad_infershape_A2_bf16)
{
    gert::InfershapeContextPara para("NsaCompressGrad",
                                     {
                                         {{{13, 128, 192}, {13, 128, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{272, 128, 192}, {272, 128, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{32, 128}, {32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {272, 128, 192},
        {32, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A3: smaller shape
TEST_F(NsaCompressGradProto, NsaCompressGrad_infershape_A3_small_fp16)
{
    gert::InfershapeContextPara para("NsaCompressGrad",
                                     {
                                         {{{4, 8, 32}, {4, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     },
                                     kInputIrInstance, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {64, 8, 32},
        {16, 8},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A4: with actSeqLenOptional present (infershape does not read it)
TEST_F(NsaCompressGradProto, NsaCompressGrad_infershape_A4_with_act_seq_len)
{
    int64_t actSeqLen[] = {0, 120, 140, 272};
    gert::InfershapeContextPara para("NsaCompressGrad",
                                     {
                                         {{{13, 128, 192}, {13, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{272, 128, 192}, {272, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                         {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                         {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     },
                                     kInputIrInstanceWithActSeq, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {272, 128, 192},
        {32, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}
