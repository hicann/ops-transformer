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

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using AnyValue = Ops::Transformer::AnyValue;
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct CompileInfo {};

std::vector<TensorDesc> Inputs(ge::DataType dtype = ge::DT_BF16)
{
    return {{{{1, 64, 512}, {1, 64, 512}}, dtype, ge::FORMAT_ND},
            {{{65, 128, 1, 512}, {65, 128, 1, 512}}, dtype, ge::FORMAT_ND},
            {{{17, 128, 1, 258}, {17, 128, 1, 258}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // ori_sparse_indices must be absent
            {{{1, 1, 512}, {1, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 65}, {1, 65}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 17}, {1, 17}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}};
}

std::vector<TensorDesc> Outputs(ge::DataType dtype = ge::DT_BF16)
{
    return {{{{1, 64, 512}, {1, 64, 512}}, dtype, ge::FORMAT_ND}, {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}};
}

std::vector<OpAttr> Attrs()
{
    return {{"softmax_scale", AnyValue::CreateFrom<float>(0.044194173f)},
            {"cmp_ratio", AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", AnyValue::CreateFrom<int64_t>(65536)},
            {"cmp_kv_stride", AnyValue::CreateFrom<int64_t>(33024)},
            {"ori_win_left", AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", AnyValue::CreateFrom<bool>(false)},
            {"kv_quant_mode", AnyValue::CreateFrom<int64_t>(3)}};
}

std::vector<OpAttr> AttrsWith(const std::string &name, const AnyValue &value)
{
    auto attrs = Attrs();
    for (auto &attr : attrs) {
        if (attr.attrName_ == name) {
            attr.attr_ = value;
        }
    }
    return attrs;
}

bool RunTilingCaseWithOutputs(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                              const std::vector<OpAttr> &attrs, TilingInfo &info)
{
    CompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseAttnSharedkv", inputs, outputs, attrs, &compileInfo, "Ascend910B", 64,
                                 262144, 16384);
    return ExecuteTiling(para, info);
}

bool RunTilingCase(const std::vector<TensorDesc> &inputs, const std::vector<OpAttr> &attrs, TilingInfo &info)
{
    return RunTilingCaseWithOutputs(inputs, Outputs(inputs[0].dtype_), attrs, info);
}
} // namespace

TEST(TurboQuantSparseAttnSharedkvTiling, AcceptsBf16AndFp16Scfa)
{
    for (ge::DataType dtype : {ge::DT_BF16, ge::DT_FLOAT16}) {
        TilingInfo info;
        EXPECT_TRUE(RunTilingCase(Inputs(dtype), Attrs(), info));
        EXPECT_GT(info.blockNum, 0U);
        EXPECT_GT(info.tilingDataSize, 0U);
    }
}

TEST(TurboQuantSparseAttnSharedkvTiling, AcceptsModelConfiguredSlidingWindow)
{
    for (int64_t oriWinLeft : {0, 127, 255}) {
        TilingInfo info;
        EXPECT_TRUE(RunTilingCase(Inputs(), AttrsWith("ori_win_left", AnyValue::CreateFrom<int64_t>(oriWinLeft)), info))
            << oriWinLeft;
    }
}

TEST(TurboQuantSparseAttnSharedkvTiling, AcceptsSupportedCompressionRatios)
{
    for (int64_t cmpRatio : {4, 128}) {
        TilingInfo info;
        EXPECT_TRUE(RunTilingCase(Inputs(), AttrsWith("cmp_ratio", AnyValue::CreateFrom<int64_t>(cmpRatio)), info))
            << cmpRatio;
    }
}

TEST(TurboQuantSparseAttnSharedkvTiling, RejectsWrongTurboQuantContract)
{
    for (const auto &entry :
         std::vector<std::pair<std::string, AnyValue>>{{"cmp_ratio", AnyValue::CreateFrom<int64_t>(8)},
                                                       {"cmp_kv_stride", AnyValue::CreateFrom<int64_t>(0)},
                                                       {"kv_quant_mode", AnyValue::CreateFrom<int64_t>(1)},
                                                       {"cmp_kv_stride", AnyValue::CreateFrom<int64_t>(33025)},
                                                       {"ori_mask_mode", AnyValue::CreateFrom<int64_t>(3)},
                                                       {"cmp_mask_mode", AnyValue::CreateFrom<int64_t>(4)},
                                                       {"ori_win_left", AnyValue::CreateFrom<int64_t>(-1)},
                                                       {"ori_win_right", AnyValue::CreateFrom<int64_t>(1)},
                                                       {"layout_q", AnyValue::CreateFrom<std::string>("BSND")},
                                                       {"layout_kv", AnyValue::CreateFrom<std::string>("BSND")}}) {
        TilingInfo info;
        EXPECT_FALSE(RunTilingCase(Inputs(), AttrsWith(entry.first, entry.second), info)) << entry.first;
    }
}

TEST(TurboQuantSparseAttnSharedkvTiling, RejectsWrongDtype)
{
    auto floatQuery = Inputs();
    floatQuery[0] = {{{1, 64, 512}, {1, 64, 512}}, ge::DT_FLOAT, ge::FORMAT_ND};
    TilingInfo floatQueryInfo;
    EXPECT_FALSE(RunTilingCase(floatQuery, Attrs(), floatQueryInfo));

    auto int8CmpKv = Inputs();
    int8CmpKv[2] = {{{17, 128, 1, 258}, {17, 128, 1, 258}}, ge::DT_INT8, ge::FORMAT_ND};
    TilingInfo int8CmpKvInfo;
    EXPECT_FALSE(RunTilingCase(int8CmpKv, Attrs(), int8CmpKvInfo));

    auto mismatchedOriKv = Inputs();
    mismatchedOriKv[1] = {{{65, 128, 1, 512}, {65, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    TilingInfo mismatchedOriKvInfo;
    EXPECT_FALSE(RunTilingCase(mismatchedOriKv, Attrs(), mismatchedOriKvInfo));
}

TEST(TurboQuantSparseAttnSharedkvTiling, RejectsWrongHeadShape)
{
    auto wrongHeadDim = Inputs();
    wrongHeadDim[0] = {{{1, 64, 256}, {1, 64, 256}}, ge::DT_BF16, ge::FORMAT_ND};
    TilingInfo wrongHeadDimInfo;
    EXPECT_FALSE(RunTilingCase(wrongHeadDim, Attrs(), wrongHeadDimInfo));

    auto wrongHeadCount = Inputs();
    wrongHeadCount[0] = {{{1, 6, 512}, {1, 6, 512}}, ge::DT_BF16, ge::FORMAT_ND};
    TilingInfo wrongHeadCountInfo;
    EXPECT_FALSE(RunTilingCase(wrongHeadCount, Attrs(), wrongHeadCountInfo));

    auto wrongKvHeads = Inputs();
    wrongKvHeads[1] = {{{65, 128, 2, 512}, {65, 128, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND};
    TilingInfo wrongKvHeadsInfo;
    EXPECT_FALSE(RunTilingCase(wrongKvHeads, Attrs(), wrongKvHeadsInfo));
}

TEST(TurboQuantSparseAttnSharedkvTiling, RejectsWrongAuxiliaryInput)
{
    auto oriSparseIndices = Inputs();
    oriSparseIndices[3] = {{{1, 1, 512}, {1, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo oriSparseIndicesInfo;
    EXPECT_FALSE(RunTilingCase(oriSparseIndices, Attrs(), oriSparseIndicesInfo));

    auto sequsedQ = Inputs();
    sequsedQ[10] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo sequsedQInfo;
    EXPECT_FALSE(RunTilingCase(sequsedQ, Attrs(), sequsedQInfo));

    auto wrongSinks = Inputs();
    wrongSinks[12] = {{{63}, {63}}, ge::DT_FLOAT, ge::FORMAT_ND};
    TilingInfo wrongSinksInfo;
    EXPECT_FALSE(RunTilingCase(wrongSinks, Attrs(), wrongSinksInfo));

    auto wrongMetadata = Inputs();
    wrongMetadata[13] = {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo wrongMetadataInfo;
    EXPECT_FALSE(RunTilingCase(wrongMetadata, Attrs(), wrongMetadataInfo));

    auto emptyOriBlockTable = Inputs();
    emptyOriBlockTable[5] = {{{1, 0}, {1, 0}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo emptyOriBlockTableInfo;
    EXPECT_FALSE(RunTilingCase(emptyOriBlockTable, Attrs(), emptyOriBlockTableInfo));
}

TEST(TurboQuantSparseAttnSharedkvTiling, RejectsMissingRequiredTurboQuantInput)
{
    auto missingCmpKv = Inputs();
    missingCmpKv[2] = {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND};
    TilingInfo missingCmpKvInfo;
    EXPECT_FALSE(RunTilingCase(missingCmpKv, Attrs(), missingCmpKvInfo));

    auto missingCmpBlockTable = Inputs();
    missingCmpBlockTable[6] = {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo missingCmpBlockTableInfo;
    EXPECT_FALSE(RunTilingCase(missingCmpBlockTable, Attrs(), missingCmpBlockTableInfo));
}

TEST(TurboQuantSparseAttnSharedkvTiling, ChecksSlotAndTopkShape)
{
    auto wrongSlot = Inputs();
    wrongSlot[2] = {{{17, 128, 1, 320}, {17, 128, 1, 320}}, ge::DT_UINT8, ge::FORMAT_ND};
    TilingInfo slotInfo;
    EXPECT_FALSE(RunTilingCase(wrongSlot, Attrs(), slotInfo));

    auto wrongTopk = Inputs();
    wrongTopk[4] = {{{1, 1, 256}, {1, 1, 256}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo topkInfo;
    EXPECT_FALSE(RunTilingCase(wrongTopk, Attrs(), topkInfo));

    auto proTopk = Inputs();
    proTopk[4] = {{{1, 1, 1024}, {1, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND};
    TilingInfo proTopkInfo;
    EXPECT_TRUE(RunTilingCase(proTopk, Attrs(), proTopkInfo));
}
