/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/op_host/tiling_base.h"
#include "../../../../op_kernel/arch22/flash_attention_score_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "softmax_tiling_mocker.h"

using namespace std;

class FlashAttentionScoreGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
        std::cout << "FlashAttentionScoreGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        SoftmaxTilingMocker::GetInstance().Reset();
        std::cout << "FlashAttentionScoreGradTiling TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_0)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q: S1=256, B=1, H1=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
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
            // dq
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dsink
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
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
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    int64_t expectTilingKey = 134258996;
    std::string expectTilingData =
        "4294967297 1 1 1 256 256 1 128 4575657222443697349 1 65536 65536 4294967296 0 1 0 0 0 0 4294967296 "
        "1099511627776 140737488388096 1125899907104768 2251799814209536 4294967296 131072 131072 131072 0 "
        "549755813952 1099511628032 4294967297 274877907072 256 0 65536 0 0 0 0 0 0 0 4294967296 131941395394560 "
        "131941395337216 17592186044419 17592186044419 1 0 0 4446465451253301249 100605314028800 8589934592 32768 "
        "47004122109248 0 0 0 140737488355330 47004122109248 0 0 0 0 131072 131072 0 0 0 1572864 0 1703936 0 1835008 1 "
        "1 1 256 256 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 1099511627777 549755814144 1099511627904 549755814144 549755814144 17179869216 4294967298 1 "
        "4294967296 2250700302057472 131072 4294967297 4294967297 4294967300 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "1099511627777 1099511627904 1099511628032 1099511627904 549755814144 8589934624 4294967304 1 0 "
        "2250700302057472 131072 4294967297 4294967297 34359738369 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 "
        "1099511627904 1099511628032 1099511627904 549755814144 8589934624 4294967304 1 0 2250700302057472 131072 "
        "4294967297 4294967297 34359738369 0 8589934594 1 0 0 0 0 0 0 0 0 549755813952 274877915136 2199023255560 "
        "549755813920 137438957568 1099511627784 2 0 34359738432 274877907456 2199023255560 34359738432 274877907456 "
        "2199023255560 1 0 ";

    std::vector<size_t> expectWorkspaces = {35520512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_sink_bnsd_bf16)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 8, 256, 8}, {1, 8, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 8, 256, 8}, {1, 8, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8, 256, 128}, {1, 8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ret);
    ASSERT_GT(tilingInfo.tilingKey, 0);
    ASSERT_GT(tilingInfo.workspaceSizes.size(), 0U);
    int64_t expectTilingKey = 134304820;
    std::string expectTilingData =
        "1 8 1 256 256 128 0 128 4575657222443697395 65536 65536 0 256 64 0 0 0 0 1 0 4294967297 0 4294967296 0 1 "
        "1099511627904 1099511627904 1 1099511627840 35184372088896 64 0 1099511627777 549755814144 1099511627904 "
        "549755814144 549755814016 8589934720 8589934594 2 0 2250700302057472 131072 4294967297 4294967297 4294967297 "
        "0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 1099511627904 1099511628032 1099511627904 549755814016 "
        "17179869312 8589934594 1 0 2250700302057472 131072 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 "
        "0 0 0 1099511627777 1099511627904 1099511628032 1099511627904 549755814016 17179869312 8589934594 1 0 "
        "2250700302057472 131072 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627808 "
        "137438961664 1099511627784 1099511627808 137438961664 1099511627784 1 0 549755813984 412316872704 "
        "3298534883336 549755813920 137438957568 1099511627784 3 0 524288 8192 274877911040 4096 0 274877911040 4096 0 "
        "274877911040 274877911040 131941395394560 131941395337216 35184372088833 35184372088833 1 3215360 0 "
        "105553116266560 13194139582464 37888 96 1 1 32 32 32 3215360 1 8 1 256 128 0 128 4446465648821796928 84480 "
        "4294967296 262144 37383395386624 0 0 0 1125899906842625 37383395386624 0 0 1125899906842625 37383395386624 "
        "262144 262144 0 0 262144 65536 0 1114624 0 2163712 1 8 1 256 256 128 0 128 3212800 8192 0 0 ";
    std::vector<size_t> expectWorkspaces = {57807360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_sink_sbh_fp16)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q: S1=256, B=1, H1=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
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
            // dq
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dsink
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
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
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    int64_t expectTilingKey = 142639412;
    std::string expectTilingData =
        "4294967297 1 1 1 256 256 1 128 4575657222443697349 1 65536 65536 4294967296 0 1 0 0 0 0 4294967296 "
        "1099511627776 140737488388096 1125899907104768 1125899907104768 4294967296 131072 131072 131072 0 "
        "549755813952 1099511628032 4294967297 274877907072 256 0 65536 0 0 0 0 0 0 0 4294967296 131941395394560 "
        "131941395337216 17592186044419 17592186044419 1 0 0 4446465451253301249 134140418675968 4294967296 32768 "
        "140737488398976 0 0 0 140737488355329 140737488398976 0 0 0 0 65536 65536 0 0 0 1048576 0 1179648 0 1310720 1 "
        "1 1 256 256 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 1099511627777 549755814144 1099511627904 549755814144 549755814144 8589934656 4294967300 2 4294967296 "
        "2250700302057472 131072 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 "
        "1099511627904 1099511628032 1099511627904 549755814016 17179869312 8589934594 1 0 2250700302057472 131072 "
        "4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 0 0 1099511627777 1099511627904 1099511628032 "
        "1099511627904 549755814144 17179869248 4294967300 1 0 2250700302057472 131072 4294967297 4294967297 "
        "17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 549755813952 274877915136 2199023255560 549755813920 137438957568 "
        "1099511627784 2 0 34359738432 274877907456 2199023255560 34359738432 274877907456 2199023255560 1 0 ";

    std::vector<size_t> expectWorkspaces = {34996224};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// tnd
TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_1)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {{{4, 2, 5, 1}, {4, 2, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{4, 1, 5, 1}, {4, 1, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{4, 1, 5, 1}, {4, 1, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{4, 2, 5, 1}, {4, 2, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // sink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dsink
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    int64_t expectTilingKey = 134341684;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {88598016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{1, 1}, {1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
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
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // sink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dsink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    int64_t expectTilingKey = 180326;
    std::string expectTilingData =
        "1 1 1 1 1 16 32 32 1 1 1 16 4575657222473777152 65536 0 1 1 0 0 27 2 4 0 4294967296 0 1 13743895347200 64 64 "
        "512 512 1024 1024 0 0 1 101632 1 1 4294967297 1 32768 8176 1 1 16 16384 1 64 8 2 7 1 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 1 0 0 4575657221408423937 87296 4294967296 1 4295010944 0 0 0 4294967297 4295010944 0 0 0 0 0 1024 0 0 0 "
        "3328 0 3840 0 4352 1 1 1 1 1 1 0 0 0 0 0 0 4294967297 4294967297 4294967297 4294967297 68719476752 4294967312 "
        "4294967297 1 0 4398046511104 1024 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967297 "
        "4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 1 4294967297 4294967297 4294967297 "
        "4294967297 68719476752 4294967312 4294967297 1 0 4398046511104 1024 4294967297 4294967297 4294967297 0 "
        "8589934594 4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 4294967297 1 "
        "4294967297 4294967297 4294967297 4294967297 68719476752 4294967312 4294967297 1 0 4398046511104 1024 "
        "4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967297 4294967297 4294967297 4294967297 "
        "4294967297 4294967297 4294967297 1 68719476737 4294967312 34359738376 68719476737 4294967312 34359738376 1 0 "
        "35115652612097 4294975472 34359738376 35115652612097 4294975472 34359738376 1 0 ";

    std::vector<size_t> expectWorkspaces = {16783616};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_3)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
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
            // dq
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dsink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08804509063256238f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    int64_t expectTilingKey = 74036;
    std::string expectTilingData =
        "4294967297 1 1 1 64 128 1 129 4575657222443651324 2 65536 65536 0 0 1 0 0 0 0 0 0 140737488388096 "
        "140737488388096 281474976776192 4294967296 36864 73728 73728 0 549755813952 274877907072 4294967297 "
        "257698037896 64 0 8192 0 0 0 0 0 0 0 4294967296 131941395394560 131941395337216 35184372088833 35184372088833 "
        "1 0 0 4446267775383502849 100605314028800 4294967296 8256 35459250017600 0 0 0 70918499991553 70918500013376 "
        "0 0 0 0 32768 67584 0 0 0 196608 0 233472 0 307200 1 1 1 64 128 129 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877906945 554050781312 274877907073 "
        "554050781312 549755813952 12884901952 4294967299 1 4294967296 2250700302057472 131072 4294967297 4294967297 "
        "12884901891 0 8589934594 1 0 0 0 0 0 0 0 0 274877906945 549755814017 274877907072 549755814017 549755813952 "
        "8589934656 4294967300 2 0 2250700302057472 131072 4294967297 4294967297 8589934594 0 8589934594 1 0 0 0 0 0 0 "
        "0 0 549755813889 274877907073 549755813952 274877907073 549755814016 4294967360 4294967298 2 0 "
        "2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 549755813952 "
        "274877915136 2199023255560 549755813920 137438957568 1099511627784 2 0 34359738428 257698038240 2061584302088 "
        "34359738428 257698038240 2061584302088 1 0 ";
    std::vector<size_t> expectWorkspaces = {33935360};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_tiling_4)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
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
            {{{3, 1, 1, 1}, {3, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{3, 1, 1, 1}, {3, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{3, 1, 1, 1}, {3, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{3, 1, 1, 1}, {3, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dsink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dsink
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend910B", 64, 262144, 8192);
    int64_t expectTilingKey = 8405300;
    std::string expectTilingData =
        "12884901891 1 3 1 128 121 1 128 4575657222443697395 0 65536 65536 0 0 1 0 0 0 0 0 0 140737488388096 "
        "281474976776192 281474976776192 12884901888 196608 185856 185856 0 549755813952 549755814016 4294967297 "
        "274877907072 128 0 15488 0 0 0 0 0 0 0 12884901888 131941395394560 131941395337216 66520453480449 "
        "66520453480449 1 0 0 4446465648821796867 134140418675968 4294967296 49152 23639500040832 0 0 0 "
        "199561360441345 12094627949184 0 0 0 0 98304 93184 0 0 0 786432 0 983040 0 1168896 3 1 1 128 121 128 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 549755813889 "
        "549755814009 549755814016 549755814016 549755814016 4294967424 4294967297 1 4294967296 2250700302057472 "
        "131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 549755813889 519691042944 549755814009 "
        "549755814016 549755814016 4294967424 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 "
        "0 8589934594 1 0 0 0 0 0 0 0 0 519691042817 549755814016 549755814016 549755814016 549755814016 4294967424 "
        "4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 "
        "549755813952 274877915136 2199023255560 549755813920 137438957568 1099511627784 2 0 34359738432 274877907456 "
        "2199023255560 34359738432 274877907456 2199023255560 1 0 ";
    std::vector<size_t> expectWorkspaces = {34909184};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

namespace {
constexpr int64_t BASIC_DET_BF16_KEY = 4398046529126;
constexpr int64_t BASIC_DET_FP16_KEY = 4398046537318;
constexpr uint32_t BASIC_DET_CUBE_CORES = 20;

gert::TilingContextPara MakeBasicDetContext(int64_t seq, int64_t qHeads, int64_t kvHeads, int64_t sparseMode = 0,
                                            int64_t nextTokens = 65536, ge::DataType dtype = ge::DT_BF16)
{
    using Ops::Transformer::AnyValue;
    using Tensor = gert::TilingContextPara::TensorDescription;
    constexpr int64_t headDim = 256;
    static Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        40,
        BASIC_DET_CUBE_CORES,
        196608,
        524288,
        65536,
        65536,
        131072,
        201326592,
        BASIC_DET_CUBE_CORES,
        platform_ascendc::SocVersion::ASCEND910B};
    const std::string socInfo = R"({"hardware_info":{
        "CORE_NUM":20,"cube_core_cnt":20,"vector_core_cnt":40,"UB_SIZE":196608,"L1_SIZE":524288,
        "L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,"L2_SIZE":201326592,"socVersion":"Ascend910B"
    }})";
    const Tensor query{{{1, qHeads, seq, headDim}, {1, qHeads, seq, headDim}}, dtype, ge::FORMAT_ND};
    const Tensor kv{{{1, kvHeads, seq, headDim}, {1, kvHeads, seq, headDim}}, dtype, ge::FORMAT_ND};
    const Tensor softmax{{{1, qHeads, seq, 8}, {1, qHeads, seq, 8}}, ge::DT_FLOAT, ge::FORMAT_ND};
    const Tensor absent{{{}, {}}, dtype, ge::FORMAT_ND};
    std::vector<Tensor> inputs(25, absent);
    inputs[0] = query;
    inputs[1] = kv;
    inputs[2] = kv;
    inputs[3] = query;
    if (sparseMode != 0) {
        const int64_t maskSize = sparseMode == 3 ? 2048 : seq;
        inputs[7] = Tensor{{{maskSize, maskSize}, {maskSize, maskSize}}, ge::DT_UINT8, ge::FORMAT_ND};
    }
    inputs[8] = softmax;
    inputs[9] = softmax;
    inputs[11] = query;
    return gert::TilingContextPara("FlashAttentionScoreGrad", inputs, {query, kv, kv, absent, absent, absent, absent},
                                   {{"scale_value", AnyValue::CreateFrom<float>(0.0625f)},
                                    {"keep_prob", AnyValue::CreateFrom<float>(1.0f)},
                                    {"pre_tockens", AnyValue::CreateFrom<int64_t>(65536)},
                                    {"next_tockens", AnyValue::CreateFrom<int64_t>(nextTokens)},
                                    {"head_num", AnyValue::CreateFrom<int64_t>(qHeads)},
                                    {"input_layout", AnyValue::CreateFrom<std::string>("BNSD")},
                                    {"inner_precise", AnyValue::CreateFrom<int64_t>(0)},
                                    {"sparse_mode", AnyValue::CreateFrom<int64_t>(sparseMode)},
                                    {"pse_type", AnyValue::CreateFrom<int64_t>(1)},
                                    {"seed", AnyValue::CreateFrom<int64_t>(0)},
                                    {"offset", AnyValue::CreateFrom<int64_t>(0)},
                                    {"out_dtype", AnyValue::CreateFrom<int64_t>(0)},
                                    {"softmax_in_layout", AnyValue::CreateFrom<std::string>("")}},
                                   &compileInfo, "Ascend910B", socInfo, 8192, 1);
}

void CheckBasicDetD256(const gert::TilingContextPara& context, int64_t key)
{
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(context, result));
    ASSERT_EQ(result.tilingKey, key);
    EXPECT_EQ(result.blockNum, BASIC_DET_CUBE_CORES);
    FlashAttentionGradBasicDetTilingData data{};
    ASSERT_EQ(result.tilingDataSize, sizeof(data));
    std::memcpy(&data, result.tilingData.get(), sizeof(data));
    const auto& params = data.basicDetTensorTilingData;
    EXPECT_EQ(params.get_s1(), context.inputTensorDesc_[0].shape_.GetStorageShape().GetDim(2));
    EXPECT_EQ(params.get_s2(), context.inputTensorDesc_[1].shape_.GetStorageShape().GetDim(2));
    constexpr size_t partialSize = BASIC_DET_CUBE_CORES * 512 * 256 * sizeof(float);
    EXPECT_EQ(params.get_dkDetWorkspaceOffset() - params.get_dqDetWorkspaceOffset(), partialSize);
    EXPECT_EQ(params.get_dvDetWorkspaceOffset() - params.get_dkDetWorkspaceOffset(), partialSize);
    ASSERT_EQ(result.workspaceSizes.size(), 1U);
    EXPECT_GE(result.workspaceSizes[0], params.get_dvDetWorkspaceOffset() + partialSize);
}

void CheckBasicDetFallback(const gert::TilingContextPara& context)
{
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(context, result));
    EXPECT_NE(result.tilingKey, BASIC_DET_BF16_KEY);
    EXPECT_NE(result.tilingKey, BASIC_DET_FP16_KEY);
}
} // namespace

TEST_F(FlashAttentionScoreGradTiling, BasicDetD256Dense)
{
    CheckBasicDetD256(MakeBasicDetContext(2048, 3, 3), BASIC_DET_BF16_KEY);
    CheckBasicDetD256(MakeBasicDetContext(3073, 1, 1), BASIC_DET_BF16_KEY);
}

TEST_F(FlashAttentionScoreGradTiling, BasicDetD256RightDownCausal)
{
    CheckBasicDetD256(MakeBasicDetContext(32784, 1, 1, 3, 0, ge::DT_FLOAT16), BASIC_DET_FP16_KEY);
}

TEST_F(FlashAttentionScoreGradTiling, BasicDetD256RejectsUnsupportedShapes)
{
    CheckBasicDetFallback(MakeBasicDetContext(8192, 4, 2));
    CheckBasicDetFallback(MakeBasicDetContext(2047, 4, 4));
    CheckBasicDetFallback(MakeBasicDetContext(32785, 1, 1));
    CheckBasicDetFallback(MakeBasicDetContext(3072, 1, 1));
}

TEST_F(FlashAttentionScoreGradTiling, BasicDetD256RejectsUnsupportedMasks)
{
    CheckBasicDetFallback(MakeBasicDetContext(2048, 3, 3, 0, 2047));
    CheckBasicDetFallback(MakeBasicDetContext(2048, 3, 3, 1));
}
