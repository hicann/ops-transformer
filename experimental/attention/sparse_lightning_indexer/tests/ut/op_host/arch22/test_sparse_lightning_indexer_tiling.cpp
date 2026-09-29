/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/tilingdata_base.h"
using namespace std;

// 跳过 tiling key 校验（仅校验 tiling data 时使用）
constexpr uint64_t SKIP_KEY = UINT64_MAX;

class SparseLightningIndexerTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseLightningIndexerTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseLightningIndexerTiling TearDown" << std::endl;
    }
};

// C2/C3 正例 + TilingData 写入断言（candBlocks=64 由 shape 末维推导下发，C5 topk=2048 主规格）：
// BSND/BSND BF16, B=2, S1=39, N1=64, D=128, S2=64, candBlocks=64, mask_mode=3, cmp_ratio=1
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_0)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND} // candidate_block_length input12（None）
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    // TilingKey = DT_Q(27,BF16) | DT_K(27)<<8 | DT_OUT(3)<<16 | PA(0)<<24 | LAYOUT(0)<<25 | K_LAYOUT(0)<<29
    int64_t expectTilingKey = 203547;
    // 尾部：returnValue(0) | candidateTopkBlocks(64)<<32 打包，candidateBlockSize(8)
    std::string expectTilingData = "2 167503724608 8796093022272 274877906944 0 3 "
                                   "9223372036854775807 9223372036854775807 1 0 274877906944 8 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// C2 TND 专式正例（candidate shape [T,N2,candBlocks]，dim0 与 query.dim0 一致）+ N8 钉死：
// TND/TND BF16, T=103（B=2: 39+64）, N1=64, D=128, kT=128（64+64）, candBlocks=64
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_tnd_0)
{
    struct SparseLICompileInfo {
    } compileInfo;
    int64_t cuSeqlensQList[] = {0, 39, 103};
    int64_t cuSeqlensKList[] = {0, 64, 128};
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{103, 64, 128}, {103, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // q (TND)     input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k (TND)     input1
            {{{103, 64}, {103, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},           // w           input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQList}, // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKList}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q   input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k   input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // metadata    input10
            {{{103, 1, 64}, {103, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},     // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // candidate_block_length input12
        },
        {
            {{{103, 1, 64}, {103, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                    // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    // TND/TND BF16: 203547 + (1<<25) + (1<<29) = 570628891
    int64_t expectTilingKey = 570628891;
    // s1Size=103（N8：TND 取 query.dim0，钉死 GetS1Size 防御分支）；s2Size=128（kT）；topk=64
    // w1 = gSize(64) | s1Size(103)<<32 = 442381631552
    std::string expectTilingData = "2 442381631552 274877907072 274877906944 0 3 "
                                   "9223372036854775807 9223372036854775807 1 0 274877906944 8 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// N8：TND 全空输入（cu_seqlens_q=[0,0,0]）——s1Size 仍取 dim0=8（ProcessInvalid 全域清理上界）
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_tnd_allempty)
{
    struct SparseLICompileInfo {
    } compileInfo;
    int64_t cuSeqlensQList[] = {0, 0, 0};
    int64_t cuSeqlensKList[] = {0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{8, 64, 128}, {8, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},      // q (TND)     input0
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},      // k (TND)     input1
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // w           input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQList}, // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKList}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q   input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k   input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // metadata    input10
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},         // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // candidate_block_length input12
        },
        {
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    // s1Size=8（= query.dim0，N8）；s2Size=64（kT）；topk=64；mask=0
    std::string expectTilingData = "2 34359738432 274877907008 274877906944 0 0 "
                                   "9223372036854775807 9223372036854775807 1 0 274877906944 8 ";
    ;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_KEY, expectTilingData);
}

// C7 正例：candidate_block_length 以空 tensor（shape (0,)）形态传入——接受
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_block_length_empty_ok)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input12（空）
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_KEY);
}

// C7 反例：candidate_block_length 非空（shape {2}）→ host 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_block_length_nonempty_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    int64_t blockLengthList[] = {1, 2};
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockLengthList} // candidate_block_length input12（非空）
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C1 反例：candidate_topk_indices dtype=fp32 → 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_cand_dtype_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},    // candidate_topk_indices input11 (fp32)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C2 反例：BSND 下 candidate dimNum=3（TND 专式）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_cand_dimnum_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_topk_indices input11 (3 维)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                  // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C2 反例：candidate dim0（S1）与 q 不一致（39 != 8）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_cand_s1_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 8, 1, 64}, {2, 8, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},      // candidate_topk_indices input11 (S1=8)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C3 反例：candBlocks=100（非 64 的倍数）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_cand_blocks_multiple_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 100}, {2, 39, 1, 100}}, ge::DT_INT32, ge::FORMAT_ND},  // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C3 反例：candBlocks=2112（> 2048）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_cand_blocks_over2k_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},           // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // metadata     input10
            {{{2, 39, 1, 2112}, {2, 39, 1, 2112}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                              // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C4 反例：candidate_block_size=6（非 2 的幂）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_block_size_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C5 反例：topk=2049（> 2048，本算子无 off 回退）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_topk_over2k_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2049}, {2, 39, 1, 2049}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2049)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C6 反例：return_value=1 → 拒绝（leak NEG_HUGE 会污染泄漏槽 value，恒不开放）
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_return_value_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// C9 反例：layout_q=BSND 与 layout_k=TND 不一致（非 PA 必须相同）→ 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_layout_mismatch_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    int64_t cuSeqlensKList[] = {0, 64, 128};
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q (BSND)    input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // k (TND)     input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w           input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKList},    // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q   input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k   input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata    input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 不支持的平台（Ascend310P）→ 拒绝（C8：本算子仅 910b/910_93）
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_tiling_unsupported_arch_failed)
{
    struct SparseLICompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseLightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata     input10
            {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},    // candidate_topk_indices input11
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // candidate_block_length input12
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}},
        &compileInfo, "Ascend310P", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================================================
// 【R18/R15/R16 2026-09-23 新增】arch22 host 拒绝矩阵（与 LIV2 镜像；SLI C6 恒 rv=0）
// ============================================================================
namespace {
struct SLIRejectFixture {
    struct SparseLICompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    std::vector<gert::TilingContextPara::OpAttr> attrs;

    SLIRejectFixture()
        : inputs{
              {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // q
              {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // k
              {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},            // w
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // cu_seqlens_q
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // cu_seqlens_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // seqused_q
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // seqused_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // cmp_residual_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // block_table
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // output_idx_offset
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // metadata
              {{{2, 39, 1, 64}, {2, 39, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},      // candidate_topk_indices
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                              // candidate_block_length
          },
          outputs{
              {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices
              {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // sparse_values
          },
          attrs{
              {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
              {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
              {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
              {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
              {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
              {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
              {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
              {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
          }
    {}
    void SetAttr(const std::string &name, Ops::Transformer::AnyValue v)
    {
        for (auto &a : attrs) {
            if (a.attrName_ == name) {
                a.attr_ = v;
                return;
            }
        }
        attrs.emplace_back(name, v);
    }
    gert::TilingContextPara Build()
    {
        return gert::TilingContextPara("SparseLightningIndexer", inputs, outputs, attrs, &compileInfo, "Ascend910B", 64,
                                       262144, 16384);
    }
};
} // namespace

// R18：seqused_q 非空 → 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_reject_seqused_q)
{
    SLIRejectFixture f;
    int64_t v[] = {39, 39};
    f.inputs[5] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, v};
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R18：output_idx_offset 非空 → 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_reject_output_idx_offset)
{
    SLIRejectFixture f;
    int64_t v[] = {0, 0};
    f.inputs[9] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, v};
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R18/R11：metadata 非空 → 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_reject_metadata)
{
    SLIRejectFixture f;
    static int64_t v[1024] = {0};
    f.inputs[10] = {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, v};
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R18：max_seqlen_q != -1 → 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_reject_max_seqlen_q)
{
    SLIRejectFixture f;
    f.SetAttr("max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// 检视 §3：cmp_ratio 非 2 的幂 → 拒绝
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_reject_cmp_ratio)
{
    for (int64_t r : {3, 6, 129}) {
        SLIRejectFixture f;
        f.SetAttr("cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(r));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
}

// R16：cu_seqlens shape [1] → 拒绝（原 uint32 下溢路径）
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_910b_tiling_reject_cuseqlens_underflow)
{
    SLIRejectFixture f;
    int64_t v1[] = {8};
    f.inputs[0] = {{{8, 64, 128}, {8, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    f.inputs[1] = {{{8, 1, 128}, {8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    f.inputs[2] = {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND};
    f.inputs[3] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, v1};
    f.inputs[4] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, v1};
    f.inputs[11] = {{{8, 1, 64}, {8, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND};
    f.outputs[0] = {{{8, 1, 2048}, {8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND};
    f.SetAttr("layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND"));
    f.SetAttr("layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND"));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// Tiling data classes are registered for the op
TEST_F(SparseLightningIndexerTiling, SparseLightningIndexer_tiling_data_class_registered)
{
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("SparseLightningIndexer"), nullptr);
}
