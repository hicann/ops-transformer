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

#include "../../../op_host/dense_lightning_indexer_softmax_lse_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class DenseLightningIndexerSoftmaxLseTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- DenseLightningIndexerSoftmaxLse Tiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- DenseLightningIndexerSoftmaxLse Tiling UT TearDown ---" << std::endl;
    }
};

TEST_F(DenseLightningIndexerSoftmaxLseTilingTest, tiling_bsnd_case0)
{
    struct DenseLightningIndexerSoftmaxLseCompileInfo {
    } compileInfo;

    int64_t b = 1, s1 = 128, s2 = 256, nidx1 = 8, nidx2 = 1, d = 128;

    int64_t actualSeqQueryData[1] = {s1};
    int64_t actualSeqKeyData[1] = {s2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerSoftmaxLse",
        {{{{b, s1, nidx1, d}, {b, s1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s2, nidx2, d}, {b, s2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b, s1, nidx1}, {b, s1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqQueryData},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqKeyData}},
        {{{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, nidx2, s1}, {b, nidx2, s1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    uint64_t expectTilingKey = 0UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(DenseLightningIndexerSoftmaxLseTilingTest, tiling_tnd_case0)
{
    struct DenseLightningIndexerSoftmaxLseCompileInfo {
    } compileInfo;

    int64_t b = 1, t1 = 128, t2 = 256, nidx1 = 16, nidx2 = 1, d = 128;

    int64_t actualSeqQueryData[1] = {t1};
    int64_t actualSeqKeyData[1] = {t2};

    gert::TilingContextPara tilingContextPara(
        "DenseLightningIndexerSoftmaxLse",
        {{{{t1, nidx1, d}, {t1, nidx1, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t2, nidx2, d}, {t2, nidx2, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{t1, nidx1}, {t1, nidx1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqQueryData},
         {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND, true, (void*)actualSeqKeyData}},
        {{{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{nidx2, t1}, {nidx2, t1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(9223372036854775807)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);

    uint64_t expectTilingKey = 1UL;

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
