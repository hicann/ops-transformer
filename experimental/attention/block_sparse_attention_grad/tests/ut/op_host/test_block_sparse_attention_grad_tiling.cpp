/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <cstring>

// 引入自动生成的 Tiling 头文件
#include "../../../op_host/block_sparse_attention_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class BlockSparseAttentionGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- BlockSparseAttentionGradTiling UT SetUp ---" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "--- BlockSparseAttentionGradTiling UT TearDown ---" << std::endl;
    }
};

namespace {
// 公共 BNSD 参数: b=1, n=4, n_kv=2, s=s_kv=128, d=128, blockShape=[64,64]
// qBlockNum = kvBlockNum = maxBlockNum = 2
constexpr int64_t B = 1;
constexpr int64_t N = 4;
constexpr int64_t NKV = 2;
constexpr int64_t S = 128;
constexpr int64_t SKV = 128;
constexpr int64_t D = 128;
constexpr int64_t MAX_BLOCK_NUM = 2;

using TensorDescPara = gert::TilingContextPara::TensorDescription;

// attenMaskOptional 缺省（未传入）占位
const TensorDescPara ATTEN_MASK_ABSENT = {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND};

// 注意：TilingData 经 SaveToBuffer 序列化为紧凑布局（按字段自然对齐写入），与 C++ 对象布局
// 不同（结构体内嵌 SoftMaxTiling 等子结构），因此不能直接 reinterpret_cast 结构体读取。
// 序列化布局：17 个 uint64_t + float(softmaxScale) + 3 个 uint32_t(sftgTmpSpaceSize/BlockX/BlockY)
// = 152 字节；hasPerBlockSizeMask(uint8_t) 位于 152，maxBlockNum(uint32_t) 经 4 字节对齐位于 156。
constexpr size_t OFFSET_SOFTMAX_SCALE = 136; // 17 * 8，用于锚定布局
constexpr size_t OFFSET_HAS_PER_BLOCK_SIZE_MASK = 152;
constexpr size_t OFFSET_MAX_BLOCK_NUM = 156;

uint8_t ReadHasPerBlockSizeMask(const TilingInfo &tilingInfo)
{
    // 锚点校验：softmaxScale=0.088388f，若布局变化此处会先行暴露
    float scale = 0.0F;
    std::memcpy(&scale, tilingInfo.tilingData.get() + OFFSET_SOFTMAX_SCALE, sizeof(float));
    EXPECT_NEAR(scale, 0.088388F, 1e-6F);
    return *(tilingInfo.tilingData.get() + OFFSET_HAS_PER_BLOCK_SIZE_MASK);
}

uint32_t ReadMaxBlockNum(const TilingInfo &tilingInfo)
{
    uint32_t value = 0;
    std::memcpy(&value, tilingInfo.tilingData.get() + OFFSET_MAX_BLOCK_NUM, sizeof(uint32_t));
    return value;
}

gert::TilingContextPara MakeBnsdCase(optiling::BlockSparseAttentionGradCompileInfo &compileInfo,
                                     const TensorDescPara &attenMaskDesc, int64_t maskType, int64_t *blockShapeData,
                                     int64_t *seqQData, int64_t *seqKvData)
{
    return gert::TilingContextPara(
        "BlockSparseAttentionGrad",
        {// --- Input Info ---
         // 0: dout [B, N, S, D]
         {{{B, N, S, D}, {B, N, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: query [B, N, S, D]
         {{{B, N, S, D}, {B, N, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: key [B, NKV, SKV, D]
         {{{B, NKV, SKV, D}, {B, NKV, SKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3: value [B, NKV, SKV, D]
         {{{B, NKV, SKV, D}, {B, NKV, SKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4: attentionOut [B, N, S, D]
         {{{B, N, S, D}, {B, N, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 5: softmaxLse [B, N, S]
         {{{B, N, S}, {B, N, S}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 6: blockSparseMaskOptional [B, N, qBlockNum, kvBlockNum]
         {{{B, N, MAX_BLOCK_NUM, MAX_BLOCK_NUM}, {B, N, MAX_BLOCK_NUM, MAX_BLOCK_NUM}}, ge::DT_UINT8, ge::FORMAT_ND},
         // 7: attenMaskOptional
         attenMaskDesc,
         // 8: blockShapeOptional
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         // 9: actualSeqLengthsOptional
         {{{B}, {B}}, ge::DT_INT64, ge::FORMAT_ND, true, seqQData},
         // 10: actualSeqLengthsKvOptional
         {{{B}, {B}}, ge::DT_INT64, ge::FORMAT_ND, true, seqKvData}},
        {// --- Output Info ---
         {{{B, N, S, D}, {B, N, S, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, NKV, SKV, D}, {B, NKV, SKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{B, NKV, SKV, D}, {B, NKV, SKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// --- Attr Info ---
         {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(NKV)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskType)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
}
} // namespace

// ============================================================================
// 用例 1: maskType=0 基线（BNSD/FP16），不传 attenMask，tiling key=1003，
//         hasPerBlockSizeMask 应为 0（向后兼容回归保护）
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type0_bnsd_baseline)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[1] = {S};
    int64_t seqKvData[1] = {SKV};

    auto tilingContextPara = MakeBnsdCase(compileInfo, ATTEN_MASK_ABSENT, 0, blockShapeData, seqQData, seqKvData);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1003UL);

    EXPECT_EQ(ReadHasPerBlockSizeMask(tilingInfo), 0);
}

// ============================================================================
// 用例 2: maskType=1（BNSD/FP16），attenMask=[B, N, maxBlockNum, 2] INT32 合法，
//         tiling key=1003，hasPerBlockSizeMask=1 且 maxBlockNum=2
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type1_bnsd_ok)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[1] = {S};
    int64_t seqKvData[1] = {SKV};

    const TensorDescPara attenMask = {
        {{B, N, MAX_BLOCK_NUM, 2}, {B, N, MAX_BLOCK_NUM, 2}}, ge::DT_INT32, ge::FORMAT_ND};
    auto tilingContextPara = MakeBnsdCase(compileInfo, attenMask, 1, blockShapeData, seqQData, seqKvData);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1003UL);

    EXPECT_EQ(ReadHasPerBlockSizeMask(tilingInfo), 1);
    EXPECT_EQ(ReadMaxBlockNum(tilingInfo), static_cast<uint32_t>(MAX_BLOCK_NUM));
}

// ============================================================================
// 用例 3: maskType=1（TND/FP16 变长），batch=2，actualSeq 累加值 [128, 256]，
//         各 batch 最大序列 128 -> maxBlockNum=2，attenMask=[2, N, 2, 2] INT32 合法，
//         tiling key=1005，hasPerBlockSizeMask=1 且 maxBlockNum=2
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type1_tnd_ok)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    constexpr int64_t batch = 2;
    constexpr int64_t totalQ = 256;
    constexpr int64_t totalKv = 256;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[2] = {128, 256};  // 各 batch q 序列: 128, 128
    int64_t seqKvData[2] = {128, 256}; // 各 batch kv 序列: 128, 128

    gert::TilingContextPara tilingContextPara(
        "BlockSparseAttentionGrad",
        {// --- Input Info ---
         // 0: dout [T, N, D]
         {{{totalQ, N, D}, {totalQ, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 1: query [T, N, D]
         {{{totalQ, N, D}, {totalQ, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 2: key [T_kv, NKV, D]
         {{{totalKv, NKV, D}, {totalKv, NKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 3: value [T_kv, NKV, D]
         {{{totalKv, NKV, D}, {totalKv, NKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 4: attentionOut [T, N, D]
         {{{totalQ, N, D}, {totalQ, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 5: softmaxLse [T, N]
         {{{totalQ, N}, {totalQ, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 6: blockSparseMaskOptional [B, N, qBlockNum, kvBlockNum]
         {{{batch, N, MAX_BLOCK_NUM, MAX_BLOCK_NUM}, {batch, N, MAX_BLOCK_NUM, MAX_BLOCK_NUM}},
          ge::DT_UINT8,
          ge::FORMAT_ND},
         // 7: attenMaskOptional [B, N, maxBlockNum, 2]
         {{{batch, N, MAX_BLOCK_NUM, 2}, {batch, N, MAX_BLOCK_NUM, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         // 8: blockShapeOptional
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeData},
         // 9: actualSeqLengthsOptional（累加值）
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, seqQData},
         // 10: actualSeqLengthsKvOptional（累加值）
         {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND, true, seqKvData}},
        {// --- Output Info ---
         {{{totalQ, N, D}, {totalQ, N, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{totalKv, NKV, D}, {totalKv, NKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{totalKv, NKV, D}, {totalKv, NKV, D}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {// --- Attr Info ---
         {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(NKV)},
         {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"preTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)},
         {"nextTokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2147483647)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1005UL);

    EXPECT_EQ(ReadHasPerBlockSizeMask(tilingInfo), 1);
    EXPECT_EQ(ReadMaxBlockNum(tilingInfo), static_cast<uint32_t>(MAX_BLOCK_NUM));
}

// ============================================================================
// 用例 4: maskType=1 但未传 attenMaskOptional -> 应报错
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type1_attenmask_absent_fail)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[1] = {S};
    int64_t seqKvData[1] = {SKV};

    auto tilingContextPara = MakeBnsdCase(compileInfo, ATTEN_MASK_ABSENT, 1, blockShapeData, seqQData, seqKvData);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 5: maskType=1 但 attenMask dtype 非 INT32 -> 应报错
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type1_attenmask_wrong_dtype_fail)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[1] = {S};
    int64_t seqKvData[1] = {SKV};

    const TensorDescPara attenMask = {
        {{B, N, MAX_BLOCK_NUM, 2}, {B, N, MAX_BLOCK_NUM, 2}}, ge::DT_UINT8, ge::FORMAT_ND};
    auto tilingContextPara = MakeBnsdCase(compileInfo, attenMask, 1, blockShapeData, seqQData, seqKvData);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 6: maskType=1 但 attenMask dim2 != maxBlockNum -> 应报错
//         （dim2 多/少都会导致 kernel 越界读或错位读，tiling 强校验拦截）
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type1_attenmask_wrong_dim2_fail)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[1] = {S};
    int64_t seqKvData[1] = {SKV};

    const TensorDescPara attenMask = {
        {{B, N, MAX_BLOCK_NUM + 1, 2}, {B, N, MAX_BLOCK_NUM + 1, 2}}, ge::DT_INT32, ge::FORMAT_ND};
    auto tilingContextPara = MakeBnsdCase(compileInfo, attenMask, 1, blockShapeData, seqQData, seqKvData);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 用例 7: maskType=2 非法取值 -> 应报错（当前仅支持 0/1）
// ============================================================================
TEST_F(BlockSparseAttentionGradTilingTest, mask_type2_invalid_fail)
{
    optiling::BlockSparseAttentionGradCompileInfo compileInfo;
    int64_t blockShapeData[2] = {64, 64};
    int64_t seqQData[1] = {S};
    int64_t seqKvData[1] = {SKV};

    auto tilingContextPara = MakeBnsdCase(compileInfo, ATTEN_MASK_ABSENT, 2, blockShapeData, seqQData, seqKvData);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
