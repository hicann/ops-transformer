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
        std::cout << "FlashAttentionScoreTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreTiling TearDown" << std::endl;
    }
};

// BSH
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_0)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        24, 48, 196608, 524288, 131072, 201326592, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_2201};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
        // q
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
        // atten_mask: S1=256, S2=256
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
        // sink
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        // 输出Tensor
        // softmaxMax: B=1, N1=1, S1=256, 8
        {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxSum: B=1, N1=1, S1=256, 8
        {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxOut
        {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // attentionOut
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
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
         {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
        },
                &compileInfo,"Ascend910B",64,262144,8192);
    int64_t expectTilingKey = 5438203184;
    std::string expectTilingData = "1 1 1 256 256 256 128 4446465452318654464 65536 65536 0 0 0 16908546 0 4294967552 0 0 0 0 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755814016 2 32 0 549755814016 2 549755814016 1 4294967297 1 4294967297 1 4294967297 1 1 128 0 0 0 0 70368744194048 0 4398046511104 4398046512128 70368744243712 0 0 0 0 0 0 0 0 0 1099511627777 549755814144 549755814016 549755814144 549755814016 8589934656 4294967300 2 0 2250700302057472 131072 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 1099511627904 549755814016 1099511627904 549755814016 17179869248 4294967300 1 0 2250700302057472 131072 4294967297 4294967297 17179869186 0 8589934594 1 0 0 0 0 0 0 0 0 549755814016 549755830272 4398046511112 549755813952 274877915136 2199023255560 2 0 4294967297 549755814016 4294967424 1099511627777 2199023255680 549755830272 256 128 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {18087936};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// BNSD
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_1)
{
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        24, 48, 196608, 524288, 131072, 201326592, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_2201};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
        // q
        {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // k: S2=256, B=1, H2=128
        {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // v: S2=256, B=1, H2=128
        {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // real_shift
        {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // drop_mask
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // padding_mask
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        // atten_mask: S1=256, S2=256
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
        // sink
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        // 输出Tensor
        // softmaxMax: B=1, N1=1, S1=256, 8
        {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxSum: B=1, N1=1, S1=256, 8
        {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxOut
        {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // attentionOut
        {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
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
        {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
        },
            &compileInfo,"Ascend910B",64,262144,8192);
    int64_t expectTilingKey = 13996131936;
    std::string expectTilingData = "1 1 1 128 128 128 64 4467570831413529805 2147483647 2147483647 0 0 1 16908291 0 4294967424 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 2 549755814016 1 549755814016 1 274877907008 1 4294967297 1 0 0 0 0 0 0 2147483647 2147483647 0 0 35184372097024 8192 2199023263744 2199023256064 35184372154368 0 0 0 0 0 0 0 0 0 549755813889 274877907072 549755813952 274877907072 549755813952 8589934656 8589934593 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967424 274877906945 549755813889 4294967297 4294967360 4294967424 549755813889 1 549755813889 549755813952 549755814016 549755813952 274877907008 8589934720 8589934593 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967424 549755813889 549755813889 4294967297 4294967360 4294967424 274877906945 1 549755813952 274877915136 2199023255560 549755813952 274877915136 2199023255560 1 0 4294967297 64 4294967296 549755813889 549755813952 274877906944 128 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16973824};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TND d=192
TEST_F(FlashAttentionScoreTiling, FlashAttentionScore_tiling_2)
{
    int64_t actual_seq_qlist[] = {13,14,81};
    int64_t actual_seq_kvlist[] = {13,14,81};
    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        24, 48, 196608, 524288, 131072, 201326592, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_2201};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
        // q
        {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
        // k: S2=256, B=1, H2=128
        {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
        // v: S2=256, B=1, H2=128
        {{{81, 2, 192}, {81, 2, 192}}, ge::DT_BF16, ge::FORMAT_ND},
        // real_shift
        {{{1,2,1024,67}, {1,2,1024,67}}, ge::DT_BF16, ge::FORMAT_ND},
        // drop_mask
        {{{81,2,192}, {81,2,192}}, ge::DT_UINT8, ge::FORMAT_ND},
        // padding_mask
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // atten_mask: S1=256, S2=256
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
        // prefix
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // actual_seq_qlen
        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_qlist},
        // actual_seq_kvlen
        {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND,true,actual_seq_kvlist},
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
        // sink
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        // 输出Tensor
        // softmaxMax: B=1, N1=1, S1=256, 8
        // {{{974,2,8}, {974,2,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxSum: B=1, N1=1, S1=256, 8
        // {{{974,2,8}, {974,2,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxOut
        {{{0,0,0}, {0,0,0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        // attentionOut
        {{{81, 2, 192}, {81, 2, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0721687f)},
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
        {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
        },
            &compileInfo,"Ascend910B",64,262144,8192);
    int64_t expectTilingKey = 16177980464;
    std::string expectTilingData = "3 2 1 81 67 80 192 4437115660700903014 67 0 1024 67 1 74027927481745412 65553 2048 0 0 0 0 6 6 1 1 0 1 2 3 4 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 287762808912 1 107374182442 0 287762808912 1 274877907072 2 4294967297 3 4294967297 2 4294967297 1 1 192 67 0 0 0 27487790700800 5497558145280 2748779079680 2748779070080 43980465162240 0 0 0 0 0 0 0 0 0 287762808833 1649267441744 287762809216 824633720899 343597383760 4294967488 4294967297 1 0 0 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 287762808833 287762809216 287762809216 287762809024 824633720912 4294967376 4294967297 1 0 0 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 2748779069450 42949679360 343597383688 2748779069450 42949679360 343597383688 1 0 4294967297 549755813968 12884902016 287762808834 3298534883712 343597393920 134 128 0 0 0 0 0 0 0 0 8796093022209 5 9318 ";
    std::vector<size_t> expectWorkspaces = {18722304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
