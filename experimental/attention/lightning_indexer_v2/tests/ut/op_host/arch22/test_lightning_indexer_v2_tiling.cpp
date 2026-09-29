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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
    // 尾部两个值: candidateTopkBlocks(-1) 与 returnValue(0) 打包为 -4294967296, candidateBlockSize(8)
    // （candidate_topk_blocks 缺省 off，TilingData 布局见 tiling.h 尾部 candidate 字段）
    std::string expectTilingData = "2 167503724608 8796093022272 274877906944 0 3 "
                                   "9223372036854775807 9223372036854775807 1 0 -4294967296 8 ";
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
    std::string expectTilingData = "2 167503724608 8796093022224 274877906944 4294967312 3 "
                                   "9223372036854775807 9223372036854775807 1 0 -4294967296 8 ";
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
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("LightningIndexerV2"), nullptr);
}

// ============================================================================
// 【R2/R9/R18 2026-09-23 新增】arch22 host 拒绝矩阵（镜像 pytest CAND_REJECT_*，
// 含 pytest 无法触达的 V5 return_value=1+on、metadata 非空、R16 下溢负例）
// 公共骨架：BSND/BSND 合法输入（同 tiling_3），按需替换 attr / 可选输入
// ============================================================================
namespace {
constexpr uint64_t SKIP_KEY = UINT64_MAX;

struct LIV2RejectFixture {
    struct LIV2CompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    std::vector<gert::TilingContextPara::OpAttr> attrs;

    LIV2RejectFixture()
        : inputs{
              {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},  // q
              {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},    // k
              {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},           // w
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cu_seqlens_q
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cu_seqlens_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // seqused_q
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // seqused_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cmp_residual_k
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // output_idx_offset
              {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // metadata
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
        return gert::TilingContextPara("LightningIndexerV2", inputs, outputs, attrs, &compileInfo, "Ascend910B", 64,
                                       262144, 16384);
    }
};
} // namespace

// R18：seqused_q 非空 → 拒绝（arch22 kernel 不消费，R6 垃圾输出场景入口）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_seqused_q)
{
    LIV2RejectFixture f;
    int64_t v[] = {39, 39};
    f.inputs[5] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, v};
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R18：output_idx_offset 非空 → 拒绝（arch22 输出索引恒不加 offset 的唯一语义定版）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_output_idx_offset)
{
    LIV2RejectFixture f;
    int64_t v[] = {0, 0};
    f.inputs[9] = {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, v};
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R18/R11：metadata 非空 → 拒绝（AICPU 分核协议仅 arch35，实验版不交付该前置算子）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_metadata)
{
    LIV2RejectFixture f;
    static int64_t v[1024] = {0};
    f.inputs[10] = {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND, true, v};
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R18：max_seqlen_q != -1 → 拒绝
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_max_seqlen_q)
{
    LIV2RejectFixture f;
    f.SetAttr("max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// 检视 §3：cmp_ratio 非 2 的幂 → 拒绝（口径统一为 (0,128] 且幂）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_cmp_ratio_not_pow)
{
    LIV2RejectFixture f;
    f.SetAttr("cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_cmp_ratio_129)
{
    LIV2RejectFixture f;
    f.SetAttr("cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(129));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// return_value 仅 0/1
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_return_value)
{
    LIV2RejectFixture f;
    f.SetAttr("return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R9/V5：return_value=1 + candidate on → 拒绝（pytest candidate 重载恒 rv=0 不可达，host 层钉死）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_rv1_candidate_on)
{
    LIV2RejectFixture f;
    f.SetAttr("return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1));
    f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R9：candidate 值域矩阵镜像（V1/V3 拒绝路径的 host 级钉死）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_candidate_matrix)
{
    const int64_t badBlocks[] = {0, -2, 100, 2112};
    for (int64_t blocks : badBlocks) {
        LIV2RejectFixture f;
        f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blocks));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
    const int64_t badBlockSizes[] = {1, 3, 128, 66};
    for (int64_t bs : badBlockSizes) {
        LIV2RejectFixture f;
        f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
        f.SetAttr("candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs));
        ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    }
    // topk>2048 + on
    LIV2RejectFixture f;
    f.outputs[0] = {{{2, 39, 1, 3072}, {2, 39, 1, 3072}}, ge::DT_INT32, ge::FORMAT_ND};
    f.SetAttr("topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3072));
    f.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
    // 合法 on + 全 64 对齐：成功（含 bs=2/64 边界）
    for (int64_t bs : {2, 64}) {
        LIV2RejectFixture ok;
        ok.SetAttr("candidate_topk_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64));
        ok.SetAttr("candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs));
        ExecuteTestCase(ok.Build(), ge::GRAPH_SUCCESS, SKIP_KEY);
    }
}

// R16：cu_seqlens shape [1]（无 [start,total] 对）→ 拒绝（原 uint32 下溢被放行）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_reject_cuseqlens_underflow)
{
    LIV2RejectFixture f;
    int64_t v1[] = {8};
    f.inputs[0] = {{{8, 64, 128}, {8, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}; // q TND rank3
    f.inputs[1] = {{{8, 1, 128}, {8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND};   // k TND rank3
    f.inputs[2] = {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND};          // w
    f.inputs[3] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, v1};        // cu_seqlens_q [1] → R16 下溢负例
    f.inputs[4] = {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, v1};        // cu_seqlens_k [1]
    f.outputs[0] = {{{8, 1, 2048}, {8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND};
    f.SetAttr("layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND"));
    f.SetAttr("layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND"));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}

// R5：TND 全空输入（cu_seqlens 全 0）→ tiling 成功且 s1Size=query.dim0（8）下发
// （全域 -1/-inf 清理行为本身在 ProcessInvalid kernel 侧，板上断言由 pytest
//  CAND_TND_ALL_EMPTY + mssanitizer memcheck 覆盖；此处钉死 host 侧不拒且形状合法）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_tnd_all_empty_success)
{
    LIV2RejectFixture f;
    int64_t qz[] = {0, 0, 0};
    int64_t kz[] = {0, 0, 0};
    f.inputs[0] = {{{8, 64, 128}, {8, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}; // q [T=8,N1,D]
    f.inputs[1] = {{{8, 1, 128}, {8, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND};   // k [kT=8,N2,D]
    f.inputs[2] = {{{8, 64}, {8, 64}}, ge::DT_FLOAT, ge::FORMAT_ND};          // w [T,N1]
    f.inputs[3] = {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, qz};
    f.inputs[4] = {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, kz};
    f.outputs[0] = {{{8, 1, 2048}, {8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}; // [T,N2,K]
    f.SetAttr("layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND"));
    f.SetAttr("layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND"));
    ExecuteTestCase(f.Build(), ge::GRAPH_SUCCESS, SKIP_KEY);
}

// R9：return_value=1（arch22）正例——values 形状校验提升后的成功路径
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_rv1_success)
{
    LIV2RejectFixture f;
    f.outputs[0] = {{{2, 39, 1, 128}, {2, 39, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND};
    f.outputs[1] = {{{2, 39, 1, 128}, {2, 39, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND};
    f.SetAttr("topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128));
    f.SetAttr("return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1));
    ExecuteTestCase(f.Build(), ge::GRAPH_SUCCESS, SKIP_KEY);
}

// R9：return_value=1 但 values rank 与 q 不一致 → 拒绝（校验已从 arch35 门禁提升为全 arch）
TEST_F(LightningIndexerV2Tiling, LightningIndexerV2_910b_tiling_rv1_bad_rank_failed)
{
    LIV2RejectFixture f;
    f.outputs[0] = {{{2, 39, 1, 128}, {2, 39, 1, 128}}, ge::DT_INT32, ge::FORMAT_ND};
    f.outputs[1] = {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND}; // rank1 ≠ 4
    f.SetAttr("topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128));
    f.SetAttr("return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1));
    ExecuteTestCase(f.Build(), ge::GRAPH_FAILED);
}
