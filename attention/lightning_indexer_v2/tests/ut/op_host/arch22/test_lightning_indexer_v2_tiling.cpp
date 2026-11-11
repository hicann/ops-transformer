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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/tilingdata_base.h"
using namespace std;

namespace {
constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;
} // namespace

class LightningIndexerV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerV2Tiling TearDown" << std::endl;
    }
};

// when key layout is not PA_BBND, input block_table must be null (BSND/BSND with block_table)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_0)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},            // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input3
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input5
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input7
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // block_table  input8 (NOT null, should fail)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}, // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}  // metadata     input10
        },
        {
            {{{1, 8, 1, 128}, {1, 8, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                        // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key shape[2] is numhead, only support 1 (BSND/PA_BBND, N2=2)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_1)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t seqused_k_list[] = {256, 256};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 16, 64, 128}, {2, 16, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{4, 64, 2, 128}, {4, 64, 2, 128}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                                 // k            input1 (N2=2, should fail)
            {{{2, 16, 64}, {2, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqused_k_list}, // cu_seqlens_k input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // cmp_residual_k input7
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},                 // block_table  input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                  // metadata     input10
        },
        {
            {{{2, 16, 1, 2048}, {2, 16, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// topk must > 0 and <= 8192 (topk=10000)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_2)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t seqused_k_list[] = {256, 256};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 16, 64, 128}, {2, 16, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{4, 64, 1, 128}, {4, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 16, 64}, {2, 16, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqused_k_list},       // cu_seqlens_k input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // seqused_k    input6
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // cmp_residual_k input7
            {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},                       // block_table  input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                       // output_idx_offset input9
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                        // metadata     input10
        },
        {
            {{{2, 16, 1, 10000}, {2, 16, 1, 10000}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                              // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(10000)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// BSND/BSND success: BF16, B=2, S1=39, N1=64, D=128, topk=2048, mask_mode=3, cmp_ratio=1
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_3)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 203547;
    // 尾槽: -4294967296=[returnValue(0)|candidateTopkBlocks(-1)], 8=[candidateBlockSize|batchInvariant],
    // 8589934593=[fastPathCandidate(1)|s1BaseSize(2)](BSND 泛化准入 + N1=64 自适应)
    std::string expectTilingData = "2 167503724608 8796093022272 274877906944 0 3 "
                                   "9223372036854775807 9223372036854775807 1 0 -4294967296 8 8589934593 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// BSND/PA_BBND success: BF16, B=2, S1=39, N1=64, D=128, block_size=16, topk=2048, mask_mode=3
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_4)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t seqused_k_list[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}},
             ge::DT_BF16,
             ge::FORMAT_ND}, // k            input1 (block_num=2, block_size=16, N2=1, D=128)
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},       // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqused_k_list}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true},                 // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                 // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1090722587;
    // 尾槽: -4294967296=[returnValue(0)|candidateTopkBlocks(-1)], 8=[candidateBlockSize|batchInvariant],
    // 8589934593=[fastPathCandidate(1)|s1BaseSize(2)](BSND 泛化准入 + N1=64 自适应)
    std::string expectTilingData = "2 167503724608 8796093022224 274877906944 4294967312 3 "
                                   "9223372036854775807 9223372036854775807 1 0 -4294967296 8 8589934593 ";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// topk=2049 on 910B must be rejected: >2048 and not a multiple of 1024
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_5)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2049)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// topk=3000 on 910B must be rejected: >2048 and not a multiple of 1024
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_6)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 3000}, {2, 39, 1, 3000}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3000)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype of q and k must be same: q=fp16, k=bf16 should fail
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_dtype_mismatch_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0 (fp16)
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},      // k            input1 (bf16)
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                                // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype of w must be float32: fp16 should fail
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_w_dtype_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // w            input2 (fp16)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                                // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype of sparse_indices must be int32: fp32 should fail
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_out_dtype_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                                // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sparse_indices (fp32)
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype of sparse_values must be float32: fp16 should fail
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_values_dtype_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                                // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND}                          // sparse_values (fp16)
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// unsupported npu arch (Ascend310P) should fail
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_tiling_unsupported_arch_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                                // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend310P", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Tiling data classes are registered for the op
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_tiling_data_class_registered)
{
    auto& factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("LightningIndexerV2"), nullptr);
}

// TND/TND success on 910B: BF16, T1=78, N1=64, D=128, topk=2048, cu_seqlens_q/cu_seqlens_k provided
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_tnd_success)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t cuSeqlensKData[] = {0, 64, 128};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// TND on 910B: cu_seqlens_q is required, missing should fail
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_cu_seqlens_q_missing_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensKData[] = {0, 64, 128};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens_q input3 (null)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND on 910B: shape size of cu_seqlens_q must be greater than 1 (size=1 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_cu_seqlens_q_size_zero_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0};
    int64_t cuSeqlensKData[] = {0, 64, 128};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND/TND on 910B: lengths of cu_seqlens_q and cu_seqlens_k must be same (3 vs 2 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_tnd_cu_seqlens_size_mismatch_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t cuSeqlensKData[] = {0, 64};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},      // k            input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND/TND on 910B: dim 0 of q, w and sparse_indices must be same (w T=40 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_tnd_weights_t_mismatch_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t cuSeqlensKData[] = {0, 64, 128};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
            {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
            {{{40, 64}, {40, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2 (T=40)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND/PA_BBND success on 910B: BF16, T1=78, block_num=2, block_size=16, topk=2048
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_pa_success)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t sequsedKData[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // k            input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},              // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData},  // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},    // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                           // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// TND/PA_BBND on 910B: shape size of seqused_k must equal batch size (3 vs 2 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_pa_seqused_k_size_mismatch_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t sequsedKData[] = {16, 16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // k            input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},              // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData},  // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // seqused_q    input5
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},    // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                          // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                           // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// PA_BBND on 910B: block_table must be provided (missing should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_pa_block_table_missing_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t sequsedKData[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},      // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8 (null)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// PA_BBND on 910B: seqused_k must be provided (missing should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_pa_seqused_k_missing_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k    input6 (null)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// PA_BBND on 910B: dtype of block_table must be int32 (int64 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_pa_block_table_dtype_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t sequsedKData[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},      // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT64, ge::FORMAT_ND},                    // block_table  input8 (int64)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// PA_BBND on 910B: block_count of k cannot be 0 (k dim0=0 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_pa_block_count_zero_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t sequsedKData[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{0, 16, 1, 128}, {0, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1 (0 blocks)
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},      // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// PA_BBND on 910B: block_size must be a multiple of 16 within (0, 1024] (100 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_pa_block_size_invalid_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t sequsedKData[] = {100, 100};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 100, 1, 128}, {2, 100, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // k            input1 (bs=100)
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},      // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// PA_BBND on 910B: dim 0 of block_table must equal query batch size (3 vs 2 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_pa_batch_mismatch_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t sequsedKData[] = {16, 16};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},      // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{3, 4}, {3, 4}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table  input8 (B=3)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                            // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// BSND/BSND success on 910B with contiguous k stride provided: covers stride parsing path
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_bsnd_k_contiguous_stride_success)
{
    struct LIV2CompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
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
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
    };
    inputs[1].stride_ = gert::Stride({8192, 128, 128, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2", inputs,
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// BSND/BSND on 910B: k only supports non-continuous keying on the 0-axis (axis 1 stride mismatch should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_bsnd_k_noncontiguous_stride_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
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
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
    };
    inputs[1].stride_ = gert::Stride({8192, 256, 128, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2", inputs,
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// BSND on 910B: dtype of seqused_k only supports int32 when provided (int64 should fail)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_bsnd_seqused_k_dtype_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                          // seqused_k    input6 (int64)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype of q and k must be float16 or bfloat16 (hifloat8 should fail and hit unknown-dtype logging)
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_q_dtype_hifloat8_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},              // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                // output_idx_offset input9
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                                 // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND/TND success on 910B with contiguous k stride provided: covers TND stride parsing path
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_k_contiguous_stride_success)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t cuSeqlensKData[] = {0, 64, 128};
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
        {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
        {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
        {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
        {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
    };
    inputs[1].stride_ = gert::Stride({128, 128, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2", inputs,
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// ============================================================================
// candidate (two-level topk) host 校验矩阵（镜像 pytest candidate 套件：
// 值域校验 / topk>2048 拒绝 / return_value 值域∈{0,1} 校验（candidate 模式已放开，
// rv=1 合法且 sparse_values 与 sparse_indices 同 shape）/ block_size 非法拒绝 / 950+candidate 拒绝）
// 公共骨架：BSND/BSND 合法输入（同 tiling_3），按需替换 attr；candidate 属性按
// GetAttrPointer<int64_t> 读取（与既有属性的读取方式一致）
// ============================================================================
namespace {
struct LIV2CandidateFixture {
    struct LIV2CompileInfo {
    } compileInfo;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    std::vector<gert::TilingContextPara::OpAttr> attrs;

    LIV2CandidateFixture()
        : inputs{
              {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // q
              {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // k
              {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},          // w
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_q
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cu_seqlens_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_q
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // seqused_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // cmp_residual_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // output_idx_offset
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // metadata
          },
          outputs{
              {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
              {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // sparse_values
          },
          attrs{
              {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
              {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
              {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
              {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
              {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
              {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
              {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
          }
    {}
    void SetAttr(const std::string& name, Ops::Transformer::AnyValue v)
    {
        for (auto& a : attrs) {
            if (a.attrName_ == name) {
                a.attr_ = v;
                return;
            }
        }
        attrs.emplace_back(name, v);
    }
    gert::TilingContextPara Build()
    {
        return gert::TilingContextPara("LightningIndexerV2", inputs, outputs, attrs, &compileInfo, "Ascend910B", 64,
                                       262144, 16384);
    }
};
} // namespace

// candidate 值域矩阵：非法 blocks（0/-2/100/2112）、非法 block_size（1/3/128/66）、topk>2048+on
// 均被 host 校验拒绝；合法 on（bs=2/8/64 边界与缺省、blocks=2048 上界）tiling 成功
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_candidate_matrix)
{
    // candidate_topk_blocks 值域：(0, 2048] 且 64 的倍数（-1 为缺省关闭）
    const int64_t badBlocks[] = {0, -2, 100, 2112};
    for (int64_t blocks : badBlocks) {
        LIV2CandidateFixture f;
        f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blocks));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
    // candidate_block_size 值域：[2, 64] 内 2 的幂（恒校验，off 时亦拒绝带病配置）
    const int64_t badBlockSizes[] = {1, 3, 128, 66};
    for (int64_t bs : badBlockSizes) {
        LIV2CandidateFixture f;
        f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
        f.SetAttr("candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
    // candidate_block_size 非法值在 candidate 关闭时同样拒绝（恒校验）
    {
        LIV2CandidateFixture f;
        f.SetAttr("candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
    // topk>2048 + candidate on → 拒绝
    {
        LIV2CandidateFixture f;
        f.outputs[0] = {{{2, 39, 1, 3072}, {2, 39, 1, 3072}}, ge::DT_INT32, ge::FORMAT_ND};
        f.SetAttr("topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3072));
        f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
    // 合法 on + 全 64 对齐：成功（含 bs=2/8/64 边界与缺省值）
    for (int64_t bs : {2, 8, 64}) {
        LIV2CandidateFixture ok;
        ok.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
        ok.SetAttr("candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs));
        ExecuteTestCase(ok.Build(), ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
    }
    // blocks=2048 上界（(0, 2048] 闭端）+ topk=2048：成功
    {
        LIV2CandidateFixture ok;
        ok.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048));
        ExecuteTestCase(ok.Build(), ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
    }
}

// return_value=1 + candidate on → 成功：candidate 模式已放开 return_value（值域 ∈{0,1}），
// rv=1 时 sparse_values 与 sparse_indices 同 shape（[B,S1,N2,topk]）；kernel 位置级 topk
// 管线恒跑，值=真实分无哨兵。candidate_topk_blocks 开启本身不受影响。
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_rv1_candidate_on_ok)
{
    LIV2CandidateFixture f;
    f.outputs[1] = {
        {{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}; // sparse_values 同 sparse_indices
    f.SetAttr("return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1));
    f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
    ExecuteTestCase(f.Build(), ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// return_value=2 → host 拒绝（"Return_value only supports 0 or 1"）。host 侧 rv 值域校验
// 位于 GetAndCheckAttrParaInfo 的 DAV_3510(950) 分支（arch22 host 不校验 rv 值域，910B 上
// 该值域由融合 torch 入口 lightning_indexer 的 candidate 路径 TORCH_CHECK 把守），故本反例走 950
// host 路径；candidate 缺省关闭，仅触发 rv 值域拒绝。
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_950_tiling_reject_rv2)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950(DAV_3510) + candidate on → 拒绝：candidate 仅 ascend910b/ascend910_93 生效，
// 950 上启用由 TilingForLightningIndexerV2 入口守卫（useArch35Checker 路由之前）显式拒绝。
// 缺省 -1(off) 不受影响（950 正常走 ParseAndCheckLIV2Arch35 单级 TopK 路径）。
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_950_tiling_reject_candidate_on)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                             // metadata     input10
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
         {"candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// TND + candidate on/off 的 s1Size 行为：
//   on  → s1Size = query.dim0（78，供 kernel ProcessInvalid 的 TND 全域 -1/-inf 清理）；
//   off → s1Size = 0（既有行为：TND 不赋值，缺省路径逐位不变）。
// TilingData 布局（tiling.h）：word1 = [low32: gSize | high32: s1Size]；
// word(total-2) = [low32: returnValue | high32: candidateTopkBlocks]；word(total-1) = candidateBlockSize。
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_candidate_s1size)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t cuSeqlensKData[] = {0, 64, 128};
    auto buildTndCase = [&compileInfo, &cuSeqlensQData, &cuSeqlensKData](int64_t candBlocks) {
        std::vector<gert::TilingContextPara::OpAttr> attrs{
            {"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        };
        if (candBlocks != -1) {
            attrs.emplace_back("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(candBlocks));
        }
        return gert::TilingContextPara(
            "LightningIndexerV2",
            {
                {{{78, 64, 128}, {78, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // q            input0
                {{{128, 1, 128}, {128, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k            input1
                {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
                {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
                {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
                {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
                {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
                {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
                {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
                {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
                {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
            },
            {
                {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
                {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
            },
            attrs, &compileInfo, "Ascend910B", 64, 262144, 16384);
    };

    // candidate on（topk=2048 与 candidate 兼容上限）：s1Size = query.dim0 = 78
    {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(buildTndCase(64), info));
        const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
        ASSERT_EQ(info.tilingDataSize / sizeof(int64_t), 13U);
        EXPECT_EQ(static_cast<uint32_t>(data[1] >> 32), 78U); // s1Size = q.dim0
        EXPECT_EQ(data[10], static_cast<int64_t>(64) << 32);  // returnValue=0 | candTopk=64
        EXPECT_EQ(data[11], 8);                               // candidateBlockSize 缺省 8
    }
    // candidate off（缺省，不传属性）：s1Size = 0（与既有单级 TopK 行为逐位一致）
    {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(buildTndCase(-1), info));
        const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
        ASSERT_EQ(info.tilingDataSize / sizeof(int64_t), 13U);
        EXPECT_EQ(static_cast<uint32_t>(data[1] >> 32), 0U);        // s1Size 保持 0
        EXPECT_EQ(data[10], static_cast<int64_t>(-1) * 4294967296); // returnValue=0 | candTopk=-1
        EXPECT_EQ(data[11], 8);                                     // candidateBlockSize 缺省 8
    }
}
