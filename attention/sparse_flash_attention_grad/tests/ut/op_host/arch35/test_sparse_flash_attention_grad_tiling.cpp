/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {
struct CompileInfo {
    uint32_t aivNum = 64;
    uint32_t aicNum = 32;
    uint64_t ubSize = 262144;
    uint64_t l1Size = 524288;
    uint64_t l0aSize = 65536;
    uint64_t l0bSize = 65536;
    uint64_t l0cSize = 262144;
    uint64_t l2CacheSize = 134217728;
    int64_t coreNum = 32;
};

const std::string kAscend950SocInfo = "{\n"
                                      "  \"hardware_info\": {\n"
                                      "    \"BT_SIZE\": 0,\n"
                                      "    \"load3d_constraints\": \"1\",\n"
                                      "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                      "    \"Intrinsic_data_move_l12ub\": true,\n"
                                      "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                      "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                      "    \"UB_SIZE\": 262144,\n"
                                      "    \"L2_SIZE\": 134217728,\n"
                                      "    \"L1_SIZE\": 524288,\n"
                                      "    \"L0A_SIZE\": 65536,\n"
                                      "    \"L0B_SIZE\": 65536,\n"
                                      "    \"L0C_SIZE\": 262144,\n"
                                      "    \"CORE_NUM\": 32,\n"
                                      "    \"socVersion\": \"Ascend950\"\n"
                                      "  }\n"
                                      "}";

// G/S2/D 模板按 ASCENDC_TPL_UINT_DECL 合法值列表的 index 编码（CANN FastEncodeTilingKeyDirect 行为）：
// 模板 128/128/512 在声明 [128]/[128]/[512,576] 中均位于 index 0 → G/S2/D 编码 0。
// kvMerge=true 置 bit34，BF16 置 bit0。最终 key = (1 << 34) + 1 = 17179869185。
constexpr uint64_t kKeyBsndBf16KvMerge = 17179869185UL;
// Deterministic 在 ASCENDC_TPL_ARGS_DECL 中紧邻 KvMerge 之前声明，故占 bit33。
constexpr uint64_t kKeyBsndBf16KvMergeDeter = kKeyBsndBf16KvMerge | (1UL << 33);

gert::TilingContextPara MakeKvMergePara(CompileInfo &compileInfo, int64_t sparseBlockSize, int64_t s2,
                                        int32_t deterministicInfo = 0)
{
    return gert::TilingContextPara(
        "SparseFlashAttentionGrad",
        {{{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, s2, 1, 512}, {1, s2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 16, 1, 1024}, {1, 16, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, s2, 1, 512}, {1, s2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0441941738f)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseBlockSize)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(deterministicInfo != 0)}},
        &compileInfo, "Ascend950", kAscend950SocInfo, 4096, deterministicInfo);
}
} // namespace

class SparseFlashAttentionGradArch35Tiling : public testing::Test {};

TEST_F(SparseFlashAttentionGradArch35Tiling, kvmerge)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("SparseFlashAttentionGrad",
                                 {{{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 16, 1, 1024}, {1, 16, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0441941738f)},
                                  {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                  {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                  {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                  {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                  {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                 &compileInfo, "Ascend950", kAscend950SocInfo, 4096);

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kKeyBsndBf16KvMerge);
}

TEST_F(SparseFlashAttentionGradArch35Tiling, kvmerge_nonempty_dv_should_fail)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("SparseFlashAttentionGrad",
                                 {{{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 16, 1, 1024}, {1, 16, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{1, 8, 16}, {1, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{{{1, 16, 8, 512}, {1, 16, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                  {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
                                 {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0441941738f)},
                                  {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                  {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                  {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                  {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                  {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                 &compileInfo, "Ascend950", kAscend950SocInfo, 4096);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseFlashAttentionGradArch35Tiling, sparse_block_size_support_list)
{
    CompileInfo compileInfo;
    for (int64_t sparseBlockSize : {1, 8, 16, 32, 64}) {
        auto para = MakeKvMergePara(compileInfo, sparseBlockSize, 128);
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, kKeyBsndBf16KvMerge);
    }
}

TEST_F(SparseFlashAttentionGradArch35Tiling, sparse_block_size_not_in_support_list_should_fail)
{
    CompileInfo compileInfo;
    for (int64_t sparseBlockSize : {0, 2, 4, 128}) {
        auto para = MakeKvMergePara(compileInfo, sparseBlockSize, 128);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
}

// 确定性计算每核常驻一整份 selected 区：S2=128 时只需 min(1024, ceil(128/64)) * 64 = 128 行。
TEST_F(SparseFlashAttentionGradArch35Tiling, deterministic_sparse_block_size_support_list)
{
    CompileInfo compileInfo;
    for (int64_t sparseBlockSize : {1, 8, 16, 32, 64}) {
        auto para = MakeKvMergePara(compileInfo, sparseBlockSize, 128, 1);
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, kKeyBsndBf16KvMergeDeter);
    }
}

TEST_F(SparseFlashAttentionGradArch35Tiling, deterministic_large_s2_block_size_64)
{
    CompileInfo compileInfo;
    auto para = MakeKvMergePara(compileInfo, 64, 16384, 1);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kKeyBsndBf16KvMergeDeter);
}

TEST_F(SparseFlashAttentionGradArch35Tiling, non_deterministic_large_s2_block_size_64)
{
    CompileInfo compileInfo;
    auto para = MakeKvMergePara(compileInfo, 64, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kKeyBsndBf16KvMerge);
}
