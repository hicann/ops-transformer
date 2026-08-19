/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../../op_host/lightning_indexer_grad_tiling_data.h"
#include "../../../../op_kernel/lightning_indexer_grad_template_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr const char *kOpName = "LightningIndexerGrad";
constexpr uint32_t kCoreNum = 1U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr size_t kTilingDataSize = 8192;

static const uint64_t kTilingKeyBsndFp16 = GET_TPL_TILING_KEY(LIG_TPL_FP16, LIG_LAYOUT_BSND);
static const uint64_t kTilingKeyTndBf16 = GET_TPL_TILING_KEY(LIG_TPL_BF16, LIG_LAYOUT_TND);

optiling::LIGCompileInfo MakeCompileInfo()
{
    return {};
}

std::vector<OpAttr> MakeAttrs(const std::string &layout, int64_t sparseMode = 3, bool determinstic = false)
{
    return {
        {"headNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
        {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
        {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"determinstic", Ops::Transformer::AnyValue::CreateFrom<bool>(determinstic)}
    };
}

gert::TilingContextPara MakeBsndPara(optiling::LIGCompileInfo &compileInfo,
                                     ge::DataType dataType = ge::DT_FLOAT16,
                                     const std::string &layout = "BSND",
                                     int64_t sparseMode = 3,
                                     int64_t headNumQ = 64,
                                     int64_t headNumK = 1,
                                     int64_t headDim = 128)
{
    return gert::TilingContextPara(
        kOpName,
        {
            {{{2, 8, headNumQ, headDim}, {2, 8, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
            {{{2, 16, headNumK, headDim}, {2, 16, headNumK, headDim}}, dataType, ge::FORMAT_ND},
            {{{2, 8, headNumQ, headDim}, {2, 8, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 8, headNumQ}, {2, 8, headNumQ}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{2, 8, headNumQ, headDim}, {2, 8, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
            {{{2, 16, headNumK, headDim}, {2, 16, headNumK, headDim}}, dataType, ge::FORMAT_ND},
            {{{2, 8, headNumQ}, {2, 8, headNumQ}}, dataType, ge::FORMAT_ND}
        },
        MakeAttrs(layout, sparseMode),
        std::vector<uint32_t>{1, 1, 1, 1, 1, 0, 0},
        std::vector<uint32_t>{1, 1, 1},
        &compileInfo,
        "Ascend910B",
        kCoreNum,
        kUbSize,
        kTilingDataSize);
}

gert::TilingContextPara MakeTndPara(optiling::LIGCompileInfo &compileInfo,
                                    int32_t *actualSeqQ,
                                    int32_t *actualSeqK,
                                    bool withActualSeq = true,
                                    ge::DataType dataType = ge::DT_BF16,
                                    int64_t sparseMode = 3,
                                    int64_t headNumQ = 64,
                                    int64_t headNumK = 1,
                                    int64_t headDim = 128)
{
    if (withActualSeq) {
        return gert::TilingContextPara(
            kOpName,
            {
                {{{64, headNumQ, headDim}, {64, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
                {{{128, headNumK, headDim}, {128, headNumK, headDim}}, dataType, ge::FORMAT_ND},
                {{{64, headNumQ, headDim}, {64, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
                {{{64, 2048}, {64, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
                {{{64, headNumQ}, {64, headNumQ}}, dataType, ge::FORMAT_ND},
                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQ},
                {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqK}
            },
            {
                {{{64, headNumQ, headDim}, {64, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
                {{{128, headNumK, headDim}, {128, headNumK, headDim}}, dataType, ge::FORMAT_ND},
                {{{64, headNumQ}, {64, headNumQ}}, dataType, ge::FORMAT_ND}
            },
            MakeAttrs("TND", sparseMode),
            std::vector<uint32_t>{1, 1, 1, 1, 1, 1, 1},
            std::vector<uint32_t>{1, 1, 1},
            &compileInfo,
            "Ascend910B",
            kCoreNum,
            kUbSize,
            kTilingDataSize);
    }

    return gert::TilingContextPara(
        kOpName,
        {
            {{{64, headNumQ, headDim}, {64, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
            {{{128, headNumK, headDim}, {128, headNumK, headDim}}, dataType, ge::FORMAT_ND},
            {{{64, headNumQ, headDim}, {64, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
            {{{64, 2048}, {64, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64, headNumQ}, {64, headNumQ}}, dataType, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {{{64, headNumQ, headDim}, {64, headNumQ, headDim}}, dataType, ge::FORMAT_ND},
            {{{128, headNumK, headDim}, {128, headNumK, headDim}}, dataType, ge::FORMAT_ND},
            {{{64, headNumQ}, {64, headNumQ}}, dataType, ge::FORMAT_ND}
        },
        MakeAttrs("TND", sparseMode),
        std::vector<uint32_t>{1, 1, 1, 1, 1, 0, 0},
        std::vector<uint32_t>{1, 1, 1},
        &compileInfo,
        "Ascend910B",
        kCoreNum,
        kUbSize,
        kTilingDataSize);
}
} // namespace

class LightningIndexerGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerGradTiling TearDown" << std::endl;
    }
};

TEST_F(LightningIndexerGradTiling, tiling_bsnd_fp16_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyBsndFp16);
}

TEST_F(LightningIndexerGradTiling, tiling_tnd_bf16_success)
{
    auto compileInfo = MakeCompileInfo();
    int32_t actualSeqQ[] = {64};
    int32_t actualSeqK[] = {128};
    auto para = MakeTndPara(compileInfo, actualSeqQ, actualSeqK);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyTndBf16);
}

TEST_F(LightningIndexerGradTiling, tiling_invalid_layout_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, "BNSD");
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradTiling, tiling_invalid_sparse_mode_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, "BSND", 1);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradTiling, tiling_invalid_head_dim_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, "BSND", 3, 64, 1, 64);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradTiling, tiling_invalid_key_head_num_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeBsndPara(compileInfo, ge::DT_FLOAT16, "BSND", 3, 64, 2, 128);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerGradTiling, tiling_tnd_missing_actual_seq_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakeTndPara(compileInfo, nullptr, nullptr, false);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
