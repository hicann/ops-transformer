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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/nsa_compress_grad_tiling.h"

namespace {
constexpr uint64_t kUtCoreNum = 1U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;
constexpr uint64_t kTilingKey = 0UL;

const std::vector<uint32_t> kInputIrInstanceNoActSeq{1, 1, 1, 0};
const std::vector<uint32_t> kInputIrInstanceWithActSeq{1, 1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1, 1};

inline optiling::NsaCompileInfo MakeCompileInfo()
{
    optiling::NsaCompileInfo info{};
    info.aicNum = 1U;
    info.aivNum = 1U;
    info.ubSize = kUbSize;
    return info;
}
} // namespace

class NsaCompressGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressGradTiling TearDown" << std::endl;
    }
};

// A1: fp16 base case with actSeqLen prefix sum
TEST_F(NsaCompressGradTiling, NsaCompressGrad_910b_tiling_A1_fp16)
{
    int64_t actSeqLen[] = {0, 120, 140, 272};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressGrad",
                                 {
                                     {{{13, 128, 192}, {13, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{272, 128, 192}, {272, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{272, 128, 192}, {272, 128, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                 },
                                 kInputIrInstanceWithActSeq, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum,
                                 kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKey);
}

// A2: bf16
TEST_F(NsaCompressGradTiling, NsaCompressGrad_910b_tiling_A2_bf16)
{
    int64_t actSeqLen[] = {0, 120, 140, 272};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressGrad",
                                 {
                                     {{{13, 128, 192}, {13, 128, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{272, 128, 192}, {272, 128, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{32, 128}, {32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{272, 128, 192}, {272, 128, 192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{32, 128}, {32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                 },
                                 kInputIrInstanceWithActSeq, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum,
                                 kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKey);
}

// A3: without actSeqLenOptional (batchSize defaults to 1)
TEST_F(NsaCompressGradTiling, NsaCompressGrad_910b_tiling_A3_no_act_seq_len)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressGrad",
                                 {
                                     {{{4, 8, 32}, {4, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                 },
                                 kInputIrInstanceNoActSeq, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum,
                                 kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKey);
}

// E1: headDim not aligned to 16
TEST_F(NsaCompressGradTiling, NsaCompressGrad_910b_tiling_E1_head_dim_misaligned)
{
    int64_t actSeqLen[] = {0, 64};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressGrad",
                                 {
                                     {{{4, 8, 17}, {4, 8, 17}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 8, 17}, {64, 8, 17}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{64, 8, 17}, {64, 8, 17}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                 },
                                 kInputIrInstanceWithActSeq, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum,
                                 kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2: compressBlockSize not multiple of 16
TEST_F(NsaCompressGradTiling, NsaCompressGrad_910b_tiling_E2_block_size_misaligned)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressGrad",
                                 {
                                     {{{4, 8, 32}, {4, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{7, 8}, {7, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{7, 8}, {7, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                 },
                                 kInputIrInstanceNoActSeq, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum,
                                 kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E3: compressStride not multiple of 16
TEST_F(NsaCompressGradTiling, NsaCompressGrad_910b_tiling_E3_stride_misaligned)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressGrad",
                                 {
                                     {{{4, 8, 32}, {4, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 8, 32}, {64, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                 },
                                 kInputIrInstanceNoActSeq, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum,
                                 kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
