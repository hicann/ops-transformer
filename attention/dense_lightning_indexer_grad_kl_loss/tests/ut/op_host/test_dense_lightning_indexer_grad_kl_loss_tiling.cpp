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
#include <string>

#include "../../../op_host/dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class DenseLightningIndexerGradKLLossTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- DenseLightningIndexerGradKLLossTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- DenseLightningIndexerGradKLLossTiling UT TearDown ---" << std::endl;
    }
};

namespace {

void ExecuteTndSeqLimitCase(const std::vector<int64_t> &actualSeqQueryData,
                            const std::vector<int64_t> &actualSeqKeyData, int64_t t1, int64_t t2,
                            ge::graphStatus expectStatus, uint64_t expectTilingKey)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo;

    int64_t n1 = 64, n2 = 64, nidx1 = 16, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;
    int64_t batch = static_cast<int64_t>(actualSeqQueryData.size());

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {{{{t1, n1, d}, {t1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, n2, d}, {t2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{t1, n1, dr}, {t1, n1, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, n2, dr}, {t2, n2, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqQueryData.data()},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKeyData.data()}},
        {{{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, expectStatus, expectTilingKey);
}

void ExecuteBsndSeqLimitCase(int64_t s1, int64_t s2, ge::graphStatus expectStatus, uint64_t expectTilingKey)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo;

    int64_t b = 1, n1 = 32, n2 = 32, nidx1 = 8, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;
    int64_t actualSeqQueryData[1] = {s1};
    int64_t actualSeqKeyData[1] = {s2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {{{{b, s1, n1, d}, {b, s1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, n2, d}, {b, s2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, s1, n1, dr}, {b, s1, n1, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, n2, dr}, {b, s2, n2, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqQueryData},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKeyData}},
        {{{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, expectStatus, expectTilingKey);
}

} // namespace

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_bsnd_case0)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo;

    int64_t b = 1, s1 = 128, s2 = 256, n1 = 32, n2 = 32, nidx1 = 8, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;

    int64_t actualSeqQueryData[1] = {s1};
    int64_t actualSeqKeyData[1] = {s2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {{{{b, s1, n1, d}, {b, s1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, n2, d}, {b, s2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, n2, s1, g}, {b, n2, s1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, s1, n1, dr}, {b, s1, n1, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, n2, dr}, {b, s2, n2, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqQueryData},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKeyData}},
        {{{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}},
        &compileInfo);

    uint64_t expectTilingKey = 1UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_bsnd_seq_limit_max)
{
    constexpr int64_t maxSeqLength = 1024L * 1024L;
    ExecuteBsndSeqLimitCase(maxSeqLength, maxSeqLength, ge::GRAPH_SUCCESS, 1UL);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_bsnd_seq_limit_exceed)
{
    constexpr int64_t maxSeqLength = 1024L * 1024L;
    ExecuteBsndSeqLimitCase(maxSeqLength + 1, maxSeqLength + 1, ge::GRAPH_FAILED, 0UL);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_tnd_case0)
{
    optiling::DenseLightningIndexerGradKLLossCompileInfo compileInfo;

    int64_t t1 = 128, t2 = 256, n1 = 64, n2 = 64, nidx1 = 16, nidx2 = 1;
    int64_t d = 128, dr = 64, g = n1 / n2;
    int64_t batch = 1;

    int64_t actualSeqQueryData[1] = {t1};
    int64_t actualSeqKeyData[1] = {t2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerGradKLLoss",
        {{{{t1, n1, d}, {t1, n1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, n2, d}, {t2, n2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{n2, t1, g}, {n2, t1, g}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{t1, n1, dr}, {t1, n1, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, n2, dr}, {t2, n2, dr}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqQueryData},
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, (void *)actualSeqKeyData}},
        {{{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388)},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}},
        &compileInfo);

    uint64_t expectTilingKey = 35UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_tnd_seq_limit_max_and_total)
{
    constexpr int64_t maxSeqLength = 1024L * 1024L;
    ExecuteTndSeqLimitCase({maxSeqLength, 2L * maxSeqLength}, {maxSeqLength, 2L * maxSeqLength}, 2L * maxSeqLength,
                           2L * maxSeqLength, ge::GRAPH_SUCCESS, 35UL);
}

TEST_F(DenseLightningIndexerGradKLLossTilingTest, tiling_tnd_seq_limit_exceed)
{
    constexpr int64_t maxSeqLength = 1024L * 1024L;
    ExecuteTndSeqLimitCase({maxSeqLength, 2L * maxSeqLength + 1}, {maxSeqLength, 2L * maxSeqLength + 1},
                           2L * maxSeqLength + 1, 2L * maxSeqLength + 1, ge::GRAPH_FAILED, 0UL);
}
