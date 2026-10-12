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
#include "../../../../op_host/nsa_compress_tiling_common.h"

namespace {
constexpr uint64_t kUtCoreNum = 1U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;
constexpr uint64_t kTilingKey = 0UL;

const std::vector<uint32_t> kInputIrInstance{1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1};

inline optiling::NsaCompressCompileInfo MakeCompileInfo()
{
    optiling::NsaCompressCompileInfo info{};
    info.aivNum = 1U;
    info.ubSize = kUbSize;
    return info;
}
} // namespace

class NsaCompressTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressTiling TearDown" << std::endl;
    }
};

// A1: fp16 Case_0 shape (legacy kernel Case_0)
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_A1_fp16)
{
    int64_t actSeqLen[] = {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
                           1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                           2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                           3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{4800, 1, 32}, {4800, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{288, 1, 32}, {288, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKey);
}

// A2: bf16 Case_12
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_A2_bf16)
{
    int64_t actSeqLen[] = {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
                           1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                           2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                           3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{4800, 9, 32}, {4800, 9, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{16, 9}, {16, 9}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{288, 9, 32}, {288, 9, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKey);
}

// A3: small shape (aclnn Case_0)
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_A3_small_fp16)
{
    int64_t actSeqLen[] = {16, 32, 48};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{48, 32, 16}, {48, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 32}, {16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{3, 32, 16}, {3, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKey);
}

// E1: layout not TND
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_E1_invalid_layout)
{
    int64_t actSeqLen[] = {16, 32, 48};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{48, 32, 16}, {48, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 32}, {16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{3, 32, 16}, {3, 32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2: compressBlockSize not multiple of 16
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_E2_invalid_block_size)
{
    int64_t actSeqLen[] = {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
                           1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                           2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                           3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{4800, 32, 32}, {4800, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{7, 32}, {7, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{0, 32, 32}, {0, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E3: compressStride > compressBlockSize
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_E3_stride_gt_block)
{
    int64_t actSeqLen[] = {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
                           1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                           2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                           3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{4800, 23, 32}, {4800, 23, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 23}, {16, 23}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{0, 23, 32}, {0, 23, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E4: actSeqLenType != 0
TEST_F(NsaCompressTiling, NsaCompress_910b_tiling_E4_invalid_act_seq_len_type)
{
    int64_t actSeqLen[] = {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
                           1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                           2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
                           3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompress",
                                 {
                                     {{{4800, 5, 32}, {4800, 5, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 5}, {32, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{48}, {48}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                 },
                                 {
                                     {{{0, 5, 32}, {0, 5, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layoutOptional", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compressBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"compressStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"actSeqLenType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
