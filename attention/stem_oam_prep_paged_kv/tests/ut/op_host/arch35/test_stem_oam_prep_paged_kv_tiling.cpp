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

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class StemOamPrepPagedKvTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StemOamPrepPagedKvTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StemOamPrepPagedKvTilingArch35 TearDown" << std::endl; }
};

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct StemOamPrepPagedKvCompileInfo {};

struct InputBundle {
    std::vector<TensorDesc> descs;
    std::vector<int32_t> kvSeqLensData;
};

InputBundle MakeInputs(int64_t totalBlocks = 8, int64_t kvBlockSize = 64, int64_t numKvHeads = 4, int64_t batch = 1,
                       int64_t maxKvBlocks = 2, const std::string &kvLayout = "BBND",
                       std::vector<int32_t> seqLens = {128})
{
    InputBundle bundle;
    bundle.kvSeqLensData = std::move(seqLens);

    if (kvLayout == "BBND") {
        bundle.descs = {
            {{{totalBlocks, kvBlockSize, numKvHeads, 128}, {totalBlocks, kvBlockSize, numKvHeads, 128}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_ND},
            {{{totalBlocks, kvBlockSize, numKvHeads, 128}, {totalBlocks, kvBlockSize, numKvHeads, 128}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_ND},
            {{{batch, maxKvBlocks}, {batch, maxKvBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{totalBlocks, kvBlockSize, numKvHeads, 1}, {totalBlocks, kvBlockSize, numKvHeads, 1}},
             ge::DT_FLOAT,
             ge::FORMAT_ND},
            {{{numKvHeads}, {numKvHeads}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };
    } else {
        bundle.descs = {
            {{{totalBlocks, numKvHeads, kvBlockSize, 128}, {totalBlocks, numKvHeads, kvBlockSize, 128}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_ND},
            {{{totalBlocks, numKvHeads, kvBlockSize, 128}, {totalBlocks, numKvHeads, kvBlockSize, 128}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_ND},
            {{{batch, maxKvBlocks}, {batch, maxKvBlocks}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{totalBlocks, numKvHeads, kvBlockSize, 1}, {totalBlocks, numKvHeads, kvBlockSize, 1}},
             ge::DT_FLOAT,
             ge::FORMAT_ND},
            {{{numKvHeads}, {numKvHeads}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };
    }

    bundle.descs[3].isConst_ = true;
    bundle.descs[3].constValue_ = bundle.kvSeqLensData.data();

    return bundle;
}

std::vector<TensorDesc> MakeOutputs(int64_t batch, int64_t numKvHeads, int64_t maxKb, int64_t stemStride)
{
    int64_t kflatDim = stemStride * 128;
    return {
        {{{batch, numKvHeads, maxKb, kflatDim}, {batch, numKvHeads, maxKb, kflatDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{batch, numKvHeads, maxKb}, {batch, numKvHeads, maxKb}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeAttrs(float lambdaMag = 0.3f, const std::string &kvLayout = "BBND",
                              int64_t stemBlockSize = 128, int64_t stemStride = 16)
{
    return {
        {"lambdaMag", Ops::Transformer::AnyValue::CreateFrom<float>(lambdaMag)},
        {"kvLayout", Ops::Transformer::AnyValue::CreateFrom<const char*>(kvLayout.c_str())},
        {"stemBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(stemBlockSize)},
        {"stemStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(stemStride)},
    };
}

gert::TilingContextPara BuildTilingPara(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                                        const std::vector<OpAttr> &attrs)
{
    static StemOamPrepPagedKvCompileInfo compileInfo;
    return gert::TilingContextPara("StemOamPrepPagedKv", inputs, outputs, attrs, &compileInfo, "Ascend950", 64, 262144,
                                   4096);
}

void ExpectTilingResult(const gert::TilingContextPara &para, bool expectSuccess)
{
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(para, tilingInfo);
    EXPECT_EQ(ok, expectSuccess);
    if (expectSuccess) {
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    }
}
} // namespace

TEST_F(StemOamPrepPagedKvTilingArch35, basic_layout0)
{
    auto inputBundle = MakeInputs(8, 64, 4, 1, 2, "BBND", {128});
    auto outputs = MakeOutputs(1, 4, 1, 16);
    auto attrs = MakeAttrs(0.3f, "BBND", 128, 16);
    ExpectTilingResult(BuildTilingPara(inputBundle.descs, outputs, attrs), true);
}

TEST_F(StemOamPrepPagedKvTilingArch35, basic_layout1)
{
    auto inputBundle = MakeInputs(8, 64, 4, 1, 2, "BNBD", {128});
    auto outputs = MakeOutputs(1, 4, 1, 16);
    auto attrs = MakeAttrs(0.3f, "BNBD", 128, 16);
    ExpectTilingResult(BuildTilingPara(inputBundle.descs, outputs, attrs), true);
}

TEST_F(StemOamPrepPagedKvTilingArch35, multibatch_kvblock128)
{
    auto inputBundle = MakeInputs(16, 128, 8, 2, 4, "BBND", {128, 256});
    auto outputs = MakeOutputs(2, 8, 2, 32);
    auto attrs = MakeAttrs(0.5f, "BBND", 128, 32);
    ExpectTilingResult(BuildTilingPara(inputBundle.descs, outputs, attrs), true);
}

TEST_F(StemOamPrepPagedKvTilingArch35, non_aligned_kvseqlens)
{
    auto inputBundle = MakeInputs(8, 64, 4, 1, 2, "BBND", {100});
    auto outputs = MakeOutputs(1, 4, 1, 16);
    auto attrs = MakeAttrs(0.3f, "BBND", 128, 16);
    ExpectTilingResult(BuildTilingPara(inputBundle.descs, outputs, attrs), true);
}

TEST_F(StemOamPrepPagedKvTilingArch35, stemblock256_stemstride64)
{
    auto inputBundle = MakeInputs(4, 64, 2, 1, 1, "BBND", {200});
    auto outputs = MakeOutputs(1, 2, 1, 64);
    auto attrs = MakeAttrs(0.3f, "BBND", 256, 64);
    ExpectTilingResult(BuildTilingPara(inputBundle.descs, outputs, attrs), true);
}

TEST_F(StemOamPrepPagedKvTilingArch35, layout1_stemstride16_maxKb2)
{
    auto inputBundle = MakeInputs(8, 64, 4, 1, 2, "BNBD", {256});
    auto outputs = MakeOutputs(1, 4, 2, 16);
    auto attrs = MakeAttrs(0.3f, "BNBD", 128, 16);
    ExpectTilingResult(BuildTilingPara(inputBundle.descs, outputs, attrs), true);
}
