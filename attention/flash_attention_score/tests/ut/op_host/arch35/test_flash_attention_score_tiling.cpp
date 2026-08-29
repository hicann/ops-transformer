/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/flash_attention_score_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FlashAttentionScoreTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "FlashAttentionScoreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        // std::cout << "FlashAttentionScoreTiling TearDown" << std::endl;
    }
};

// FP32 SBH
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_0)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: S1=256, B=1, H1=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=1, N1=1, S1=256, 8
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=256, 8
         {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // attentionOut
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603268571936;
    std::string expectTilingData =
        "1 0 0 1 1 256 256 0 128 128 0 4446465452318654464 65536 65536 0 0 0 16908546 1099511627776 1 0 0 0 0 0 0 255 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP16 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_1)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=1, N1=1, S1=128, D=256
            {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=1, N2=1, S2=128, D=256
            {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=1, N2=1, S2=128, DV=256
            {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=1, N1=1, S1=128, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=128, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 256}, {1, 1, 128, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2526519393640251952;
    std::string expectTilingData =
        "1 0 0 1 1 128 128 0 256 256 0 4467570831413529805 2147483647 2147483647 128 128 1 281474993618947 "
        "549755813888 1 0 0 2147483647 2147483647 0 0 204 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 "
        "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {17170432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BF16 TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_2)
{
    int64_t actual_seq_qlist[] = {13, 14, 81};
    int64_t actual_seq_kvlist[] = {13, 14, 81};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=81, N1=2, D=192
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=81, N2=2, D=192
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=81, N2=2, DV=192
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{1, 2, 1024, 67}, {1, 2, 1024, 67}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{81, 2, 192}, {81, 2, 192}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=81, N1=2, 8
         {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=81, N1=2, 8
         {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{81, 2, 192}, {81, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2522015793744446016;
    std::string expectTilingData =
        "3 0 0 2 1 81 67 0 192 192 0 4437115660700903014 67 0 1024 67 1 74027927481745412 8796093026561 0 0 0 67 0 2 0 "
        "229 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 "
        "6 7 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 1179648 8 8 8 ";
    std::vector<size_t> expectWorkspaces = {17966592};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP16  TND D=126
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_3)
{
    int64_t actual_seq_qlist[] = {169, 333};
    int64_t actual_seq_kvlist[] = {169, 297};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=333, N1=4, D=126
            {{{333, 4, 126}, {333, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: T2=297, N2=4, D=126
            {{{297, 4, 126}, {297, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: T2=297, N2=4, DV=126
            {{{297, 4, 126}, {297, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=333, N1=4, 8
         {{{333, 4, 8}, {333, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=333, N1=4, 8
         {{{333, 4, 8}, {333, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{333, 4, 126}, {333, 4, 126}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.1767766f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2328361009230643776;
    std::string expectTilingData =
        "2 0 0 4 1 333 169 0 126 126 0 4482494421136310272 65536 65536 0 0 0 260 0 0 0 0 65536 65536 2 0 255 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 24 24 3 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 "
        "11 12 13 14 15 16 17 18 19 20 21 22 23 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 "
        "24 24 24 24 24 24 24 24 24 24 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP8 + scale BSND
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_4)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, S1=128, N1=9, D=16
            {{{2, 128, 9, 16}, {2, 128, 9, 16}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            // k: B=2, S2=256, N2=3, D=16
            {{{2, 256, 3, 16}, {2, 256, 3, 16}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            // v: B=2, S2=256, N2=3, DV=16
            {{{2, 256, 3, 16}, {2, 256, 3, 16}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            // real_shift
            //  {{{2, 9, 1, 256}, {2, 9, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{2, 9, 1, 1}, {2, 9, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{2, 3, 1, 1}, {2, 3, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{2, 3, 1, 1}, {2, 3, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=9, S1=128, 8
         {{{2, 9, 128, 8}, {2, 9, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=9, S1=128, 8
         {{{2, 9, 128, 8}, {2, 9, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 128, 9, 16}, {2, 128, 9, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // attentionOut
         {{{2, 128, 9, 16}, {2, 128, 9, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.25f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2904821760729023248;
    std::string expectTilingData =
        "2 0 0 3 3 128 256 0 16 16 0 4503599628435849216 65536 65536 0 0 0 257 0 0 0 0 0 0 0 0 255 0 0 0 0 0 0 0 0 0 0 "
        "0 0 18 18 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP8 e5m2 + band + attenmask=SS pse=BN BSH
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_5)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, S1=700, H1=128
            {{{2, 700, 128}, {2, 700, 128}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            // k: B=2, S2=856, H2=128
            {{{2, 856, 128}, {2, 856, 128}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            // v: B=2, S2=856, H2=128
            {{{2, 856, 128}, {2, 856, 128}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{2, 1, 6, 1}, {2, 1, 6, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{2, 1, 4, 1}, {2, 1, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{2, 1, 4, 1}, {2, 1, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=1, S1=700, 8
         {{{2, 1, 700, 8}, {2, 1, 700, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=1, S1=700, 8
         {{{2, 1, 700, 8}, {2, 1, 700, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 700, 128}, {2, 700, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 700, 128}, {2, 700, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 3481282513837753104;
    std::string expectTilingData =
        "2 0 0 1 1 700 856 0 128 128 0 4446465645592182784 65536 65536 0 0 0 257 0 0 0 0 0 0 2 0 255 0 0 0 0 0 0 0 0 0 "
        "0 0 0 12 12 6 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BF16 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_6)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=1, N1=1, S1=128, D=384
            {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: B=1, N2=1, S2=128, D=384
            {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: B=1, N2=1, S2=128, DV=384
            {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{128, 128}, {128, 128}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=1, N1=1, S1=128, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=128, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 384}, {1, 1, 128, 384}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(12)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(33)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2526519393908687664;
    std::string expectTilingData =
        "1 0 0 1 1 128 128 0 384 384 0 4467570831408496640 2147483647 2147483647 128 128 1 281474993618947 "
        "549755813888 1 0 0 2147483647 2147483647 12 33 127 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 "
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {18546688};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BF16 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_7)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=1, N1=1, S1=128, D=64
            {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: B=1, N2=1, S2=255, D=64
            {{{1, 1, 255, 64}, {1, 1, 255, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: B=1, N2=1, S2=255, DV=64
            {{{1, 1, 255, 64}, {1, 1, 255, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask bnss
            {{{1, 1, 128, 255}, {1, 1, 128, 255}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=1, N1=1, S1=128, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=1, N1=1, S1=128, 8
         {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.6f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(24)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2531022991657009712;
    std::string expectTilingData =
        "1 0 0 1 1 128 255 0 64 64 0 4467570831410174362 65409 127 0 0 1 73183506846581507 8796093022209 2 0 0 128 128 "
        "6 24 153 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 "
        "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {16809984};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP16 TND D=384
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_8)
{
    int64_t actual_seq_qlist[] = {64, 192};
    int64_t actual_seq_kvlist[] = {64, 192};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=192, N1=4, D=384
            {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: T2=192, N2=4, D=384
            {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: T2=192, N2=4, DV=384
            {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=192, N1=4, 8
         {{{192, 4, 8}, {192, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=192, N1=4, 8
         {{{192, 4, 8}, {192, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{192, 4, 384}, {192, 4, 384}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0510310f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2328361010304385856;
    std::string expectTilingData =
        "2 0 0 4 1 192 128 0 384 384 0 4418319178713268224 65536 65536 0 0 0 260 0 0 0 0 65536 65536 2 0 255 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 8 8 8 8 "
        "8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 ";
    std::vector<size_t> expectWorkspaces = {30932992};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// hif8 BSND
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_9)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, S1=128, N1=9, D=16
            {{{1, 7200, 40, 128}, {1, 7200, 40, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            // k: B=2, S2=256, N2=3, D=16
            {{{1, 512, 40, 128}, {1, 512, 40, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            // v: B=2, S2=256, N2=3, DV=16
            {{{1, 512, 40, 128}, {1, 512, 40, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            // real_shift
            //  {{{2, 9, 1, 256}, {2, 9, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{1, 40, 57, 1}, {1, 40, 57, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{1, 40, 2, 1}, {1, 40, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{1, 40, 1, 1}, {1, 40, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=9, S1=128, 8
         {{{1, 40, 7200, 1}, {1, 40, 7200, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=9, S1=128, 8
         {{{1, 40, 7200, 1}, {1, 40, 7200, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{1, 7200, 40, 128}, {1, 7200, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1, 7200, 40, 128}, {1, 7200, 40, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.25f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2904821761534591760;
    std::string expectTilingData =
        "1 0 0 40 1 7200 512 0 128 128 0 4503599628435849216 65536 65536 0 0 0 257 0 0 0 0 0 0 0 0 255 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 64 2280 57 36 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=5 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_10)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    int64_t prefix_list[] = {256, 256};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{2, 8, 1, 1024}, {2, 8, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2, 8, 1024, 1024}, {2, 8, 1024, 1024}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2382404204759089968;
    std::string expectTilingData =
        "2 0 0 8 1 1024 1024 0 128 128 0 4467570831416885248 65536 65536 1 1024 2 1407374900330755 4398046511104 1 0 0 "
        "65536 65536 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 128 8 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 4 6 8 12 14 16 20 22 24 28 30 32 36 38 40 44 46 48 52 54 56 60 62 64 68 70 72 76 78 80 84 86 88 92 "
        "94 96 100 102 104 108 110 112 116 118 120 124 126 128 128 128 128 128 128 128 128 128 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=2 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_11)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{2, 8, 1, 1024}, {2, 8, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2382404204759089968;
    std::string expectTilingData =
        "2 0 0 8 1 1024 1024 0 128 128 0 4467570831416885248 1024 0 1 1024 2 844429242007811 8796093022208 1 0 0 1024 "
        "0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 128 8 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "4 6 7 10 12 14 15 18 20 22 23 26 28 30 31 34 36 38 39 42 44 46 47 50 52 54 55 58 60 62 63 66 68 70 71 74 76 "
        "78 79 82 84 86 87 90 92 94 95 98 100 102 103 106 108 110 111 114 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=4 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_12)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{2, 8, 1, 1024}, {2, 8, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2382404204759089968;
    std::string expectTilingData =
        "2 0 0 8 1 1024 1024 0 128 128 0 4467570831416885248 65536 65536 1 1024 2 12901810435 8796093022208 1 0 0 0 0 "
        "0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 128 8 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=4 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_13)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{2, 8, 1, 1024}, {2, 8, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2382404204759089976;
    std::string expectTilingData =
        "2 0 0 8 1 1024 1024 0 128 128 0 4467570831416885248 -256 512 1 1024 2 1128111831908611 8796093022208 1 0 0 "
        "-256 512 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 128 8 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 2 5 7 8 10 13 15 16 18 21 23 24 26 29 31 32 34 37 39 40 42 45 47 48 50 53 55 56 58 61 63 64 66 69 71 72 "
        "74 77 79 80 82 85 87 88 90 93 95 96 98 101 103 104 106 109 111 112 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=3 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_14)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{2, 8, 1024, 2048}, {2, 8, 1024, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 2048, 8}, {2, 8, 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 2048, 8}, {2, 8, 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2382404204759089968;
    std::string expectTilingData =
        "2 0 0 8 1 2048 2048 0 128 128 0 4467570831416885248 2048 0 1024 2048 2 844433536974851 8796093026560 1 0 0 "
        "2048 0 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 256 16 4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 8 11 14 19 25 28 30 35 41 44 46 51 57 60 62 67 73 76 78 83 89 92 94 99 105 108 110 115 121 124 126 131 "
        "137 140 142 147 153 156 158 163 169 172 174 179 185 188 190 195 201 204 206 211 217 220 222 227 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=4 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_15)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{2, 8, 1, 1024}, {2, 8, 1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2382404204759089976;
    std::string expectTilingData =
        "2 0 0 8 1 1024 1024 0 128 128 0 4467570831416885248 512 -256 1 1024 2 1128111831908611 8796093022208 1 0 0 "
        "512 -256 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 128 8 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 3 6 8 9 11 14 16 17 19 22 24 25 27 30 32 33 35 38 40 41 43 46 48 49 51 54 56 57 59 62 64 65 67 70 72 "
        "73 75 78 80 81 83 86 88 89 91 94 96 97 99 102 104 105 107 110 112 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=3 BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_16)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: B=2, N1=8, S1=1024, D=128
            {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: B=2, N2=8, S2=1024, D=128
            {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: B=2, N2=8, S2=1024, DV=128
            {{{2, 8, 2048, 128}, {2, 8, 2048, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: B=2, N1=8, S1=1024, 8
         {{{2, 8, 1024, 8}, {2, 8, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{2, 8, 1024, 128}, {2, 8, 1024, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603268571952;
    std::string expectTilingData =
        "2 0 0 8 1 1024 2048 0 128 128 0 4467570831416885248 1024 1024 0 0 0 1125908513685763 8796093022208 1 0 0 1024 "
        "1024 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 128 8 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 "
        "77 79 81 83 85 87 89 91 93 95 97 99 101 103 105 107 109 111 113 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=1 TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_17)
{
    int64_t actual_seq_qlist[] = {512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {512, 1024, 1536};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1536, N2=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1536, N2=2, DV=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{1536, 1536}, {1536, 1536}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "3 0 0 2 1 1536 512 0 192 192 0 4437115660702580736 2147483647 2147483647 0 0 0 281474999976196 6597069766656 "
        "0 0 0 2147483647 2147483647 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 "
        "32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=2 TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_18)
{
    int64_t actual_seq_qlist[] = {512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {512, 1024, 1536};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1536, N2=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1536, N2=2, DV=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "3 0 0 2 1 1536 512 0 192 192 0 4437115660702580736 512 0 0 0 0 844429242007812 8796093022208 0 0 0 512 0 0 0 "
        "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 "
        "5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 "
        "43 44 45 46 47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=4 TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_19)
{
    int64_t actual_seq_qlist[] = {512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {512, 1024, 1536};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1536, N2=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1536, N2=2, DV=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "3 0 0 2 1 1536 512 0 192 192 0 4437115660702580736 512 0 0 0 0 1970337738785028 8796093022208 0 0 0 512 0 0 0 "
        "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 "
        "5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 "
        "43 44 45 46 47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=7 TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_20)
{
    int64_t actual_seq_qlist[] = {512, 768};
    int64_t actual_seq_kvlist[] = {512, 1024};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=768, N1=2, D=192
            {{{768, 2, 192}, {768, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1024, N2=2, D=192
            {{{1024, 2, 192}, {1024, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1024, N2=2, DV=192
            {{{1024, 2, 192}, {1024, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=768, N1=2, 8
         {{{768, 2, 8}, {768, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=768, N1=2, 8
         {{{768, 2, 8}, {768, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{768, 2, 192}, {768, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "2 0 0 2 1 768 512 0 192 192 0 4437115660702580736 65536 -256 0 0 4294967296 2251821305430276 8796093022208 0 "
        "0 0 65536 -256 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 24 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 24 24 24 24 24 24 24 24 24 24 24 "
        "24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 ";
    std::vector<size_t> expectWorkspaces = {20316160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=8 TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_21)
{
    int64_t actual_seq_qlist[] = {256, 768};
    int64_t actual_seq_kvlist[] = {512, 1024};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=768, N1=2, D=192
            {{{768, 2, 192}, {768, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1024, N2=2, D=192
            {{{1024, 2, 192}, {1024, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1024, N2=2, DV=192
            {{{1024, 2, 192}, {1024, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=768, N1=2, 8
         {{{768, 2, 8}, {768, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=768, N1=2, 8
         {{{768, 2, 8}, {768, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{768, 2, 192}, {768, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "2 0 0 2 1 768 512 0 192 192 0 4437115660702580736 65536 -256 0 0 0 2533300577108228 8796093022208 0 0 0 65536 "
        "-256 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 24 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 "
        "24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 ";
    std::vector<size_t> expectWorkspaces = {20316160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_22)
{
    int64_t actual_seq_qlist[] = {512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {512, 1024, 1536};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1536, N2=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1536, N2=2, DV=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "3 0 0 2 1 1536 512 0 192 192 0 4437115660702580736 512 0 0 0 0 844429242007812 8796093022208 0 0 0 512 0 0 0 "
        "255 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 "
        "11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 "
        "47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_23)
{
    int64_t actual_seq_qlist[] = {0, 512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {0, 512, 1024, 1536};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1536, N2=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1536, N2=2, DV=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "4 0 0 2 1 1536 512 0 192 192 0 4437115660702580736 256 256 0 0 0 1125899923751172 2199023255552 0 0 0 256 256 "
        "0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 "
        "3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 "
        "42 43 44 45 46 47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND D=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_24)
{
    int64_t actual_seq_qlist[] = {0, 512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {0, 1024, 2048, 3072};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=3072, N2=2, D=192
            {{{3072, 2, 192}, {3072, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=3072, N2=2, DV=192
            {{{3072, 2, 192}, {3072, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{512, 1024}, {512, 1024}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-128)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "4 0 0 2 1 1536 1024 0 192 192 0 4437115660702580736 -128 256 0 0 0 1125899923751172 4398046511104 0 0 0 -128 "
        "256 0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 "
        "40 41 42 43 44 45 46 47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=6
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_25)
{
    int64_t actual_seq_qlist[] = {512, 1024, 1536};
    int64_t actual_seq_kvlist[] = {512, 1024, 1536};
    int64_t prefix[3] = {10, 12, 60};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=1536, N1=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1536, N2=2, D=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1536, N2=2, DV=192
            {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=1536, N1=2, 8
         {{{1536, 2, 8}, {1536, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{1536, 2, 192}, {1536, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "3 0 0 2 1 1536 512 0 192 192 0 4437115660702580736 256 256 0 0 0 1407392080331012 8796093022208 0 0 0 256 256 "
        "0 0 255 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 48 48 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 "
        "3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 "
        "42 43 44 45 46 47 48 48 48 48 48 48 48 48 48 ";
    std::vector<size_t> expectWorkspaces = {23855104};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// sparse_mode=8 pse_type=3
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_950_tiling_26)
{
    int64_t actual_seq_qlist[] = {256, 768};
    int64_t actual_seq_kvlist[] = {512, 1024};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
            // q: T1=768, N1=2, D=192
            {{{768, 2, 192}, {768, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // k: T2=1024, N2=2, D=192
            {{{1024, 2, 192}, {1024, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // v: T2=1024, N2=2, D=192
            {{{1024, 2, 192}, {1024, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
            // real_shift
            {{{2, 2}, {2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 输出Tensor
         // softmaxMax: T1=768, N1=2, 8
         {{{768, 2, 8}, {768, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxSum: T1=768, N1=2, 8
         {{{768, 2, 8}, {768, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // softmaxOut
         {{{0, 0, 0}, {0, 0, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // attentionOut
         {{{768, 2, 192}, {768, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", 64, 262144, 8192);
    int64_t expectTilingKey = 2400418603805442624;
    std::string expectTilingData =
        "2 0 0 2 1 768 512 0 192 192 0 4437115660702580736 65536 -256 0 0 0 2533300577108228 8796093022208 3 0 0 65536 "
        "-256 0 0 255 0 0 0 0 0 0 0 0 0 0 0 24 24 8 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 "
        "7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 "
        "24 24 24 24 24 24 24 24 24 24 24 24 24 ";
    std::vector<size_t> expectWorkspaces = {20316160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
