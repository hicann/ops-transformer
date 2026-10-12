/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "../../../../op_host/sparse_lightning_indexer_grad_kl_loss_tiling_common.h"
#include "../../../../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "../../../../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_template_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr const char* kOpName = "SparseLightningIndexerGradKLLoss";
constexpr uint32_t kCoreNum = 1U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr uint64_t kL1Size = 524288ULL;
constexpr uint64_t kL0CSize = 131072ULL;
constexpr uint64_t kL2Size = 33554432ULL;
constexpr size_t kTilingDataSize = 8192;

static const uint64_t kTilingKeyBsndFp16 = GET_TPL_TILING_KEY(1, 2048, static_cast<uint8_t>(LayoutType::LAYOUT_BSND),
                                                              static_cast<uint8_t>(LayoutType::LAYOUT_BSND), 3, 0);
static const uint64_t kTilingKeyTndBf16 = GET_TPL_TILING_KEY(1, 2048, static_cast<uint8_t>(LayoutType::LAYOUT_TND),
                                                             static_cast<uint8_t>(LayoutType::LAYOUT_TND), 3, 0);

optiling::SparseLightningIndexerGradKLLossCompileInfo MakeCompileInfo()
{
    optiling::SparseLightningIndexerGradKLLossCompileInfo info{};
    info.core_num = kCoreNum;
    info.aivNum = kCoreNum;
    info.aicNum = kCoreNum;
    info.ubSize = kUbSize;
    info.l1Size = kL1Size;
    info.l0cSize = kL0CSize;
    info.l2CacheSize = kL2Size;
    return info;
}

std::vector<OpAttr> MakeAttrs(const std::string& layout, int64_t sparseMode = 3, bool deterministic = false)
{
    return {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
            {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(deterministic)}};
}

gert::TilingContextPara MakeBsndPara(optiling::SparseLightningIndexerGradKLLossCompileInfo& compileInfo,
                                     ge::DataType dataType = ge::DT_FLOAT16, ge::DataType weightType = ge::DT_FLOAT16,
                                     const std::string& layout = "BSND", int64_t sparseMode = 3, int64_t topK = 2048,
                                     int64_t queryHeadNum = 32)
{
    constexpr int64_t b = 1;
    constexpr int64_t s1 = 128;
    constexpr int64_t s2 = 256;
    constexpr int64_t n2 = 1;
    constexpr int64_t nIdx = 8;
    constexpr int64_t d = 512;
    constexpr int64_t dIdx = 128;
    constexpr int64_t dRope = 64;

    return gert::TilingContextPara(
        kOpName,
        {{{{b, s1, queryHeadNum, d}, {b, s1, queryHeadNum, d}}, dataType, ge::FORMAT_ND},
         {{{b, s2, n2, d}, {b, s2, n2, d}}, dataType, ge::FORMAT_ND},
         {{{b, s1, nIdx, dIdx}, {b, s1, nIdx, dIdx}}, dataType, ge::FORMAT_ND},
         {{{b, s2, n2, dIdx}, {b, s2, n2, dIdx}}, dataType, ge::FORMAT_ND},
         {{{b, s1, nIdx}, {b, s1, nIdx}}, weightType, ge::FORMAT_ND},
         {{{b, n2, s1, topK}, {b, n2, s1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{b, n2, s1, nIdx}, {b, n2, s1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, n2, s1, nIdx}, {b, n2, s1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{b, s1, queryHeadNum, dRope}, {b, s1, queryHeadNum, dRope}}, dataType, ge::FORMAT_ND},
         {{{b, s2, n2, dRope}, {b, s2, n2, dRope}}, dataType, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{b, s1, nIdx, dIdx}, {b, s1, nIdx, dIdx}}, dataType, ge::FORMAT_ND},
         {{{b, s2, n2, dIdx}, {b, s2, n2, dIdx}}, dataType, ge::FORMAT_ND},
         {{{b, s1, nIdx}, {b, s1, nIdx}}, weightType, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        MakeAttrs(layout, sparseMode), std::vector<uint32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        std::vector<uint32_t>{1, 1, 1, 1}, &compileInfo, "Ascend910B", kCoreNum, kUbSize, kTilingDataSize);
}

gert::TilingContextPara MakeTndPara(optiling::SparseLightningIndexerGradKLLossCompileInfo& compileInfo,
                                    int64_t* actualSeqQuery, int64_t* actualSeqKey)
{
    constexpr int64_t t1 = 128;
    constexpr int64_t t2 = 256;
    constexpr int64_t n1 = 64;
    constexpr int64_t n2 = 1;
    constexpr int64_t nIdx = 16;
    constexpr int64_t d = 512;
    constexpr int64_t dIdx = 128;
    constexpr int64_t dRope = 64;
    constexpr int64_t topK = 2048;
    constexpr int64_t batch = 1;

    return gert::TilingContextPara(kOpName,
                                   {{{{t1, n1, d}, {t1, n1, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t2, n2, d}, {t2, n2, d}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t1, nIdx, dIdx}, {t1, nIdx, dIdx}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t2, n2, dIdx}, {t2, n2, dIdx}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t1, nIdx}, {t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{n2, t1, topK}, {n2, t1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
                                    {{{n2, t1, nIdx}, {n2, t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{n2, t1, nIdx}, {n2, t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{t1, n1, dRope}, {t1, n1, dRope}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t2, n2, dRope}, {t2, n2, dRope}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqQuery},
                                    {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, actualSeqKey}},
                                   {{{{t1, nIdx, dIdx}, {t1, nIdx, dIdx}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t2, n2, dIdx}, {t2, n2, dIdx}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{t1, nIdx}, {t1, nIdx}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                   MakeAttrs("TND"), std::vector<uint32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   std::vector<uint32_t>{1, 1, 1, 1}, &compileInfo, "Ascend910B", kCoreNum, kUbSize,
                                   kTilingDataSize);
}
} // namespace

class SparseLightningIndexerGradKLLossTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseLightningIndexerGradKLLossTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseLightningIndexerGradKLLossTiling TearDown" << std::endl;
    }
};

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_bsnd_fp16_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyBsndFp16);
}

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_tnd_bf16_success)
{
    auto compileInfo = MakeCompileInfo();
    int64_t actualSeqQuery[] = {128};
    int64_t actualSeqKey[] = {256};
    auto para = MakeTndPara(compileInfo, actualSeqQuery, actualSeqKey);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyTndBf16);
}

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_invalid_layout_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, ge::DT_FLOAT16, "BNSD");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_invalid_sparse_mode_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, ge::DT_FLOAT16, "BSND", 0);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_invalid_dtype_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, ge::DT_BF16);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_invalid_topk_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, ge::DT_FLOAT16, "BSND", 3, 1536);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerGradKLLossTiling, tiling_invalid_query_head_num_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, ge::DT_FLOAT16, "BSND", 3, 2048, 16);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
