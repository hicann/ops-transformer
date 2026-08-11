/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {
constexpr int64_t HEAD_DIM = 128;
// 算子不限制 block_size，这里只是 BuildTilingContext 的默认值；
// 取值覆盖见 tiling_success_with_various_block_size
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;

class UndGenQkvRmsNormRopeCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
};

// gert::StorageShape 只有 initializer_list 构造，维数在编译期固定；
// 输出 shape 用例需要按运行期 vector 造 shape，这里补一个helper
gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

// 全部可选输入均提供（hasGen = true, hasCatIndices = true）
// qOut / kCacheOut / vCacheOut 传空表示用推导出的正确 shape，非空则覆盖（用于输出 shape 校验用例）
gert::TilingContextPara BuildTilingContext(int64_t undLen, int64_t genLen, int64_t numHeadQ, int64_t numHeadK,
                                           int64_t numHeadV, int64_t headDim = HEAD_DIM,
                                           int64_t blockSize = BLOCK_SIZE, int64_t maxPos = 4096,
                                           const std::vector<int64_t>& mropeSection = {16, 16, 16},
                                           const std::vector<int64_t>& qOutOverride = {},
                                           const std::vector<int64_t>& kCacheOutOverride = {},
                                           const std::vector<int64_t>& vCacheOutOverride = {})
{
    const int64_t total = undLen + genLen;
    const int64_t numHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (total + blockSize - 1) / blockSize + 1;

    const std::vector<int64_t> qOut =
        qOutOverride.empty() ? std::vector<int64_t>{total, numHeadQ, headDim} : qOutOverride;
    const std::vector<int64_t> kCacheOut = kCacheOutOverride.empty()
                                               ? std::vector<int64_t>{blockNum, blockSize, numHeadK, headDim}
                                               : kCacheOutOverride;
    const std::vector<int64_t> vCacheOut = vCacheOutOverride.empty()
                                               ? std::vector<int64_t>{blockNum, blockSize, numHeadV, headDim}
                                               : vCacheOutOverride;

    static optiling::UndGenQkvRmsNormRopeCacheCompileInfo compileInfo = {};

    return gert::TilingContextPara(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, headDim}, {undLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, headDim}, {maxPos, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, blockSize, numHeadK, headDim}, {blockNum, blockSize, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, blockSize, numHeadV, headDim}, {blockNum, blockSize, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, total}, {MROPE_AXIS_NUM, total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{genLen, numHead, headDim}, {genLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {MakeStorageShape(qOut), ge::DT_BF16, ge::FORMAT_ND},
            {MakeStorageShape(kCacheOut), ge::DT_BF16, ge::FORMAT_ND},
            {MakeStorageShape(vCacheOut), ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeSection)},
        },
        &compileInfo,
        "Ascend950");
}
// TilingData 的 POD 镜像，字段顺序必须与 op_host/und_gen_qkv_rms_norm_rope_cache_tiling.h
// 中的 BEGIN_TILING_DATA_DEF 完全一致。CheckTilingData 会用 tilingDataSize 断言二者未漂移。
#pragma pack(1)
struct TilingDataMirror {
    int64_t totalTokens;
    int64_t undLen;
    int64_t genLen;
    int64_t numHead;
    int64_t numHeadQ;
    int64_t numHeadK;
    int64_t numHeadV;
    int64_t headDim;
    int64_t maxPos;
    int64_t blockNum;
    int64_t blockSize;
    int64_t hasGen;
    int64_t hasCatIndices;
    int64_t mropeSectionT;
    int64_t mropeSectionH;
    int64_t mropeSectionW;
    float epsilon;
    float reciprocal;
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t ubFactor;
};
#pragma pack()

// UT faker 的 Ascend950 平台参数。真机上 tiling 走 GetCoreNumAiv / GetCoreMemSize 运行时读取，
// 实际核数可能与此处不同；这两个常量只用来复算 UT 里的期望值，不代表真机取值。
constexpr int64_t PLATFORM_CORE_NUM = 64;
constexpr int64_t PLATFORM_UB_BYTES = 256 * 1024;

const TilingDataMirror* AsTilingData(const TilingInfo& tilingInfo)
{
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(TilingDataMirror))
        << "TilingData 布局已变，同步更新本文件的 TilingDataMirror";
    return reinterpret_cast<const TilingDataMirror*>(tilingInfo.tilingData.get());
}

// 按 tiling.h 的 "UB 划分" 注释复算一遍 ubFactor 的期望值
int64_t ExpectedUbFactor(int64_t numHeadQ, int64_t numHeadK, int64_t numHeadV, int64_t blockFactor)
{
    const int64_t n = numHeadQ + numHeadK + numHeadV;
    const int64_t d = HEAD_DIM;
    const int64_t resident = 4 * d * 2 +                       // wBf16Buf
                             4 * d * 4 +                       // wFp32Buf
                             (((d / 2) * 4 + 31) / 32) * 32 +  // gatherIdxBuf
                             5 * 256 * 8;                      // idxBuf（5 区 x IDX_WINDOW_TOKENS 个 int64）
    // gamma 是 TBuf 常驻、不随 token 数伸缩（VF 按 undMask 算基址直接取 wFp32Buf），
    // 所以 perToken 只有三个队列
    const int64_t perToken = 2 * n * d * 2 + 2 * MROPE_AXIS_NUM * d * 4 + 2 * n * d * 2;
    // MAX_UB_FACTOR 是 kernel 侧 und/gen 位图的宽度上限，见 op_host/..._tiling.h
    return std::min({(PLATFORM_UB_BYTES - resident) / perToken, blockFactor, static_cast<int64_t>(64)});
}

// 校验多核切分：核数拉满、总量守恒、核间负载差 ≤ 1
void CheckBlockTiling(const TilingDataMirror* td, int64_t total)
{
    const int64_t expectUsedCore = std::min(PLATFORM_CORE_NUM, total);
    EXPECT_EQ(td->usedCoreNum, expectUsedCore);
    EXPECT_EQ(td->formerCoreNum, total % expectUsedCore);
    EXPECT_EQ(td->tailBlockFactor, total / expectUsedCore);
    EXPECT_EQ(td->blockFactor, td->formerCoreNum > 0 ? td->tailBlockFactor + 1 : td->tailBlockFactor);
    // 总量守恒
    EXPECT_EQ(td->formerCoreNum * td->blockFactor + (td->usedCoreNum - td->formerCoreNum) * td->tailBlockFactor,
              total);
    // 核间负载差恒 ≤ 1
    EXPECT_LE(td->blockFactor - td->tailBlockFactor, 1);
}
} // namespace

// ==================== 成功路径 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_h8_1_1)
{
    constexpr int64_t undLen = 5;
    constexpr int64_t genLen = 3;
    auto para = BuildTilingContext(undLen, genLen, 8, 1, 1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);

    const auto* td = AsTilingData(tilingInfo);
    ASSERT_NE(td, nullptr);
    // total=8 < 64 核：每核 1 个 token，blockDim 只开 8 个
    CheckBlockTiling(td, undLen + genLen);
    EXPECT_EQ(td->usedCoreNum, 8);
    EXPECT_EQ(tilingInfo.blockNum, 8U);
    // 每核只有 1 个 token，ubFactor 被 blockFactor 卡到 1
    EXPECT_EQ(td->ubFactor, 1);
    EXPECT_EQ(td->ubFactor, ExpectedUbFactor(8, 1, 1, td->blockFactor));
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_h16_2_2)
{
    constexpr int64_t undLen = 6652;
    constexpr int64_t genLen = 4100;
    auto para = BuildTilingContext(undLen, genLen, 16, 2, 2);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);

    const auto* td = AsTilingData(tilingInfo);
    ASSERT_NE(td, nullptr);
    CheckBlockTiling(td, undLen + genLen);
    EXPECT_EQ(td->usedCoreNum, PLATFORM_CORE_NUM);
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(PLATFORM_CORE_NUM));
    // N=20 档位受 UB 限制，ubFactor 远小于每核 token 数
    EXPECT_EQ(td->ubFactor, ExpectedUbFactor(16, 2, 2, td->blockFactor));
    EXPECT_GT(td->ubFactor, 1);
    EXPECT_LT(td->ubFactor, td->blockFactor);
}

// T 不设人为上限：切分与偏移全是 int64_t，对 T 的绝对大小没有假设，真实上限只有
// KV Cache 容量。这里用 1M token 守住这一点，谁再加回上限都会被这条挡下。
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_when_total_tokens_far_exceeds_64k)
{
    constexpr int64_t undLen = 1000000;
    constexpr int64_t genLen = 48576; // total = 1048576 = 16 * 64K
    auto para = BuildTilingContext(undLen, genLen, 16, 2, 2);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    const auto* td = AsTilingData(tilingInfo);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalTokens, undLen + genLen);
    CheckBlockTiling(td, undLen + genLen);
    EXPECT_EQ(td->usedCoreNum, PLATFORM_CORE_NUM);
    EXPECT_EQ(td->ubFactor, ExpectedUbFactor(16, 2, 2, td->blockFactor));
}

// kernel 把 tile 内各 token 的 und/gen 标志压进一个 uint64_t 位图带给 VF，
// ubFactor 超过位图宽度会让 CopyIn 的 1ULL << i 越界且不报错。CalUbTiling 已夹取，
// 这里再钉一道：UB 预算若哪天变大到能容纳 >64 个 token，这条会先失败。
// NOTE: 当前预算下 ubFactor 最大也只有 19，所以这是防回归的绊线而非能触发的用例。
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_ub_factor_never_exceeds_und_mask_width)
{
    const std::vector<std::vector<int64_t>> headCombos{{8, 1, 1}, {16, 2, 2}};
    for (const auto& heads : headCombos) {
        auto para = BuildTilingContext(30000, 30000, heads[0], heads[1], heads[2]);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
        const auto* td = AsTilingData(tilingInfo);
        ASSERT_NE(td, nullptr);
        EXPECT_GT(td->ubFactor, 0);
        EXPECT_LE(td->ubFactor, optiling::MAX_UB_FACTOR)
            << "ubFactor 超过 kernel 侧 undMask 的位宽，会导致 gamma 选错";
    }
}

// total 略大于核数：ceil 切法会大幅少用核，余数分配必须把核数拉满
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_block_split_uses_all_cores_when_total_slightly_exceeds_cores)
{
    constexpr int64_t undLen = 100;
    constexpr int64_t genLen = 13; // total = 113，CeilDiv(113,64)=2 → ceil 切法只会用 57 个核
    auto para = BuildTilingContext(undLen, genLen, 8, 1, 1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    const auto* td = AsTilingData(tilingInfo);
    ASSERT_NE(td, nullptr);
    CheckBlockTiling(td, undLen + genLen);
    EXPECT_EQ(td->usedCoreNum, PLATFORM_CORE_NUM); // 64 个核全用上
    EXPECT_EQ(td->formerCoreNum, 49);              // 113 = 49*2 + 15*1
    EXPECT_EQ(td->blockFactor, 2);
    EXPECT_EQ(td->tailBlockFactor, 1);
}

// N=10 与 N=20 两档的 UB 预算不同，ubFactor 必须跟着变
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_ub_factor_differs_between_head_combos)
{
    TilingInfo infoSmall;
    auto paraSmall = BuildTilingContext(30000, 30000, 8, 1, 1);
    ASSERT_TRUE(ExecuteTiling(paraSmall, infoSmall));
    const auto* tdSmall = AsTilingData(infoSmall);
    ASSERT_NE(tdSmall, nullptr);

    TilingInfo infoLarge;
    auto paraLarge = BuildTilingContext(30000, 30000, 16, 2, 2);
    ASSERT_TRUE(ExecuteTiling(paraLarge, infoLarge));
    const auto* tdLarge = AsTilingData(infoLarge);
    ASSERT_NE(tdLarge, nullptr);

    EXPECT_EQ(tdSmall->ubFactor, ExpectedUbFactor(8, 1, 1, tdSmall->blockFactor));
    EXPECT_EQ(tdLarge->ubFactor, ExpectedUbFactor(16, 2, 2, tdLarge->blockFactor));
    // N=20 单 token 占 UB 更多，一次能处理的 token 数必然更少
    EXPECT_GT(tdSmall->ubFactor, tdLarge->ubFactor);

    // ExpectedUbFactor 是按同一套公式复算的，往常驻区加 buffer 时它会跟着变小而测试仍然通过。
    // 这里再钉一次绝对值：常驻区涨到挤掉一个 token 时必须有人看见，而不是默默少跑一个。
    EXPECT_EQ(tdSmall->ubFactor, 18) << "N=10 档的 ubFactor 变了，检查是不是常驻区又长大了";
    EXPECT_EQ(tdLarge->ubFactor, 10) << "N=20 档的 ubFactor 变了，检查是不是常驻区又长大了";
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_plain_rope_empty_mrope_section)
{
    // mrope_section 为空时退化为 [D/2, 0, 0]，等价标准 RoPE
    auto para = BuildTilingContext(127, 1, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);

    const auto* td = AsTilingData(tilingInfo);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->mropeSectionT, HEAD_DIM / 2);
    EXPECT_EQ(td->mropeSectionH, 0);
    EXPECT_EQ(td->mropeSectionW, 0);
    CheckBlockTiling(td, 128);
}

// ==================== 失败路径 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_head_dim_unsupported)
{
    // 仅支持 headDim = 128
    auto para = BuildTilingContext(7, 1, 8, 1, 1, 64);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_num_heads_mismatch_n_dim)
{
    // und_qkv 的 N 维按 (8,1,1) 造，但属性声明 (16,2,2)
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    auto badPara = para;
    // 直接改属性，使 num_heads_q + k + v != N
    badPara.attrs_[0] = {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_mrope_section_size_invalid)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_mrope_section_sum_exceeds_half)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {32, 32, 32});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_kv_cache_capacity_insufficient)
{
    // Bn * Bs 小于 total：block_size 造成容量不足
    auto para = BuildTilingContext(7, 1, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096);
    auto badPara = para;
    // k_cache/v_cache 改成只有 1 页 1 行
    badPara.inputTensorDesc_[4] = {{{1, 1, 1, HEAD_DIM}, {1, 1, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    badPara.inputTensorDesc_[5] = {{{1, 1, 1, HEAD_DIM}, {1, 1, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_positions_shape_invalid)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    auto badPara = para;
    // positions 第一维必须是 3
    badPara.inputTensorDesc_[7] = {{{2, 8}, {2, 8}}, ge::DT_INT64, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_slot_mapping_length_mismatch)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    auto badPara = para;
    badPara.inputTensorDesc_[6] = {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

// ==================== dtype 校验 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_und_qkv_dtype_invalid)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{8, 10, HEAD_DIM}, {8, 10, HEAD_DIM}}, ge::DT_FLOAT16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_cos_sin_cache_dtype_invalid)
{
    // cos_sin_cache 必须是 FLOAT32
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[3] = {{{4096, HEAD_DIM}, {4096, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_kv_cache_dtype_invalid)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[4] = {{{2, BLOCK_SIZE, 1, HEAD_DIM}, {2, BLOCK_SIZE, 1, HEAD_DIM}},
                                ge::DT_FLOAT16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_slot_mapping_dtype_invalid)
{
    // 索引类张量统一 int64，int32 应被拒
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[6] = {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_positions_dtype_invalid)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[7] = {{{MROPE_AXIS_NUM, 8}, {MROPE_AXIS_NUM, 8}}, ge::DT_INT32, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ==================== 支持范围校验 ====================

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_head_combo_unsupported)
{
    // 本期只支持 (8,1,1) 与 (16,2,2)
    auto para = BuildTilingContext(7, 1, 4, 2, 2);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// block_size 不参与任何寻址与切分：cache 强制连续 BBND，slot 直接当扁平行号用，
// 所以 Bs 取什么都该过。这里连 2 的幂都不要求（100），并顺带确认 tilingData 原样透传。
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_success_with_various_block_size)
{
    const std::vector<int64_t> blockSizes{1, 16, 64, 100, 256, 512};
    for (int64_t bs : blockSizes) {
        auto para = BuildTilingContext(7, 1, 8, 1, 1, HEAD_DIM, bs);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(para, tilingInfo)) << "block_size=" << bs << " 应当支持";
        const auto* td = AsTilingData(tilingInfo);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->blockSize, bs);
        // 切分只看 total 与 UB，与 Bs 无关
        EXPECT_EQ(td->totalTokens, 8);
        EXPECT_GT(td->ubFactor, 0);
    }
}

// Bs 唯一的真实约束是容量：Bn * Bs 必须够放下 T
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_small_block_size_lacks_capacity)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1, HEAD_DIM, 2);
    auto badPara = para;
    // Bn=1, Bs=2 只能放 2 个 token，而 total=8
    badPara.inputTensorDesc_[4] = {{{1, 2, 1, HEAD_DIM}, {1, 2, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    badPara.inputTensorDesc_[5] = {{{1, 2, 1, HEAD_DIM}, {1, 2, 1, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(badPara, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_kv_cache_head_num_mismatch_attr)
{
    // k_cache 的 Hk 与 num_heads_k 不一致
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[4] = {{{2, BLOCK_SIZE, 2, HEAD_DIM}, {2, BLOCK_SIZE, 2, HEAD_DIM}},
                                ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_gen_qkv_n_dim_mismatch)
{
    // gen_qkv 的 N 维与 und_qkv 不一致
    auto para = BuildTilingContext(8, 4, 8, 1, 1);
    para.inputTensorDesc_[8] = {{{4, 20, HEAD_DIM}, {4, 20, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_weights_length_mismatch)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[1] = {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_cos_sin_cache_dim_mismatch)
{
    auto para = BuildTilingContext(7, 1, 8, 1, 1);
    para.inputTensorDesc_[3] = {{{4096, 64}, {4096, 64}}, ge::DT_FLOAT, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ==================== 可选输入缺省：当前不支持 ====================
// gen_qkv / gen_weights_q / gen_weights_k / cat_indices 当前必须全部提供，
// 缺省的退化路径（纯 prefill、单序列恒等映射）暂不支持，tiling 必须拦截。
// IR/OpDef 仍保留 OPTIONAL 声明，后续放开时把这几个用例改回成功路径即可。
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_all_optional_inputs_absent)
{
    constexpr int64_t undLen = 128;
    constexpr int64_t numHeadQ = 8;
    constexpr int64_t numHeadK = 1;
    constexpr int64_t numHeadV = 1;
    constexpr int64_t numHead = numHeadQ + numHeadK + numHeadV;
    constexpr int64_t maxPos = 4096;
    const int64_t blockNum = (undLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    static optiling::UndGenQkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, HEAD_DIM}, {undLen, numHead, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{HEAD_DIM}, {HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{HEAD_DIM}, {HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, HEAD_DIM}, {maxPos, HEAD_DIM}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{undLen}, {undLen}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, undLen}, {MROPE_AXIS_NUM, undLen}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{undLen, numHeadQ, HEAD_DIM}, {undLen, numHeadQ, HEAD_DIM}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadK, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}, {blockNum, BLOCK_SIZE, numHeadV, HEAD_DIM}},
             ge::DT_BF16,
             ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16, 16})},
        },
        &compileInfo,
        "Ascend950");
    para.inputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    para.outputInstanceNum_ = {1, 1, 1};

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// cat_indices 单独缺省同样不支持
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_cat_indices_absent)
{
    auto para = BuildTilingContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_.erase(para.inputTensorDesc_.begin() + 11, para.inputTensorDesc_.end());
    para.inputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};
    para.outputInstanceNum_ = {1, 1, 1};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// gen_qkv 提供但 gen_len == 0（纯 prefill 的另一种表达）同样不支持
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_gen_len_zero)
{
    auto para = BuildTilingContext(8, 0, 8, 1, 1);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// gen_qkv 提供但 gen_weights 缺省时必须失败
TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_gen_weights_missing)
{
    auto para = BuildTilingContext(8, 4, 8, 1, 1);
    // TensorDescription 无默认构造，只能 erase 不能 resize：只保留到 gen_qkv，丢掉 gen_weights_q/k 与 cat_indices
    para.inputTensorDesc_.erase(para.inputTensorDesc_.begin() + 9, para.inputTensorDesc_.end());
    para.inputInstanceNum_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
    para.outputInstanceNum_ = {1, 1, 1};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ==================== 输出 shape 校验 ====================
// 输出 buffer 由调用方分配，tiling 不校验的话 kernel 会按 TilingData 里的 T 写满而静默越界

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_q_out_token_dim_mismatch)
{
    // q 少一行：T = 8 但只开了 7
    auto para = BuildTilingContext(5, 3, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16, 16}, {7, 8, HEAD_DIM});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_q_out_head_dim_mismatch)
{
    // q 的 Hq 与属性不符
    auto para = BuildTilingContext(5, 3, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16, 16}, {8, 4, HEAD_DIM});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_q_out_not_3d)
{
    auto para = BuildTilingContext(5, 3, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16, 16}, {8 * 8, HEAD_DIM});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_k_cache_out_shape_mismatch_input)
{
    // k_cache 是原地写入，输出 shape 必须与输入逐维一致；这里 Bn 少一个 block
    auto para = BuildTilingContext(5, 3, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16, 16}, {},
                                   {1, BLOCK_SIZE, 1, HEAD_DIM});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(UndGenQkvRmsNormRopeCacheTiling, tiling_fail_when_v_cache_out_dim_num_mismatch_input)
{
    auto para = BuildTilingContext(5, 3, 8, 1, 1, HEAD_DIM, BLOCK_SIZE, 4096, {16, 16, 16}, {}, {},
                                   {2 * BLOCK_SIZE, 1, HEAD_DIM});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
