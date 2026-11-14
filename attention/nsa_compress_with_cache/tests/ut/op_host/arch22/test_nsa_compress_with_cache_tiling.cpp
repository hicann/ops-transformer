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
#include "../../../../op_host/nsa_compress_with_cache_tiling_common.h"

namespace {
constexpr uint64_t kUtCoreNum = 48U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint32_t kTilingDataSize = 8192U;

// input, weight, slot_mapping, output_cache, act_seq_len(opt), block_table(opt)
const std::vector<uint32_t> kInputIrInstance{1, 1, 1, 1, 1, 1};
const std::vector<uint32_t> kOutputIrInstance{1};

inline optiling::NsaCompressWithCacheCompileInfo MakeCompileInfo()
{
    optiling::NsaCompressWithCacheCompileInfo info{};
    info.aivNum = static_cast<uint32_t>(kUtCoreNum);
    info.ubSize = kUbSize;
    return info;
}
} // namespace

class NsaCompressWithCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressWithCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressWithCacheTiling TearDown" << std::endl;
    }
};

// compressStride=0 可绕过 %16 校验，tiling 必须显式拒绝（issue 7234）
TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_910b_tiling_reject_stride_le_0)
{
    int64_t actSeqLen[] = {144, 192, 144, 144};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressWithCache",
                                 {
                                     {{{16, 128, 32, 128}, {16, 128, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{32, 32}, {32, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{4, 32, 128}, {4, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     {{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{4, 32, 128}, {4, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                     {"act_seq_len_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"page_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// compressStride = 16 - 2^32 = -4294967280，截断到uint32_t后伪装成合法值16，
// 必须在int64_t原始值上拦截（截断绕过问题）
TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_910b_tiling_reject_stride_truncated_to_valid)
{
    int64_t actSeqLen[] = {144, 192, 144, 144};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaCompressWithCache",
        {
            {{{16, 128, 32, 128}, {16, 128, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{32, 32}, {32, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 32, 128}, {4, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
            {{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 32, 128}, {4, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16LL - 4294967296LL)},
            {"act_seq_len_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"page_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        },
        kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize, kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(NsaCompressWithCacheTiling, NsaCompressWithCache_910b_tiling_A1_fp16)
{
    int64_t actSeqLen[] = {32, 0, 0, 0};
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para("NsaCompressWithCache",
                                 {
                                     {{{16, 128, 24, 192}, {16, 128, 24, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 24}, {32, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{512, 24, 192}, {512, 24, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actSeqLen},
                                     {{{4, 4}, {4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{512, 24, 192}, {512, 24, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                     {"compress_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                     {"compress_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                     {"act_seq_len_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                     {"page_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                 },
                                 kInputIrInstance, kOutputIrInstance, &compileInfo, "Ascend910B", kUtCoreNum, kUbSize,
                                 kTilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS);
}
