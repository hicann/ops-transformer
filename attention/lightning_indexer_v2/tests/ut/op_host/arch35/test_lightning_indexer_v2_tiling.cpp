/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

// DAV_3510 (Ascend950) tiling cases for LightningIndexerV2
class LightningIndexerV2TilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerV2TilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerV2TilingArch35 TearDown" << std::endl;
    }
};

namespace {
constexpr uint64_t SKIP_TILING_KEY = UINT64_MAX;
}

// BSND/BSND success on Ascend950: FP16, B=2, S1=39, N1=64, D=128, topk=2048, mask_mode=3, return_value=1
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_bsnd_bsnd_success)
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
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                        // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// BSND/BSND success on Ascend950 with optional output_idx_offset provided (3 dims = qExpectShapeDim - 1)
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_bsnd_bsnd_with_idx_offset_success)
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
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_INT32, ge::FORMAT_ND},             // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                        // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// TND/TND success on Ascend950: BF16, cu_seqlens_q/cu_seqlens_k provided, return_value=0
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_tnd_tnd_success)
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
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                  // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values (return_value=0)
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// TND/PA_BBND success on Ascend950: FP16, block_table + seqused_k + cu_seqlens_q provided, return_value=0
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_tnd_pa_success)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t sequsedKData[] = {32, 32};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                                 // k (block_num=2, block_size=16) input1
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // cu_seqlens_k input4 (PA: must be null on 950)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData}, // seqused_k    input6 (PA: required)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // cmp_residual_k input7
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values (return_value=0)
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, SKIP_TILING_KEY);
}

// Ascend950: metadata is required on DAV_3510, missing metadata should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_metadata_missing_failed)
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND} // metadata     input10 (null, should fail)
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: metadata size must be 1024, wrong size should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_metadata_size_failed)
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
            {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND} // metadata     input10 (size != 1024)
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: cmp_ratio != 1 and mask_mode != 0 require cmp_residual_k, missing should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_cmp_residual_missing_failed)
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // cmp_residual_k input7 (null, should fail)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND} // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950 BSND: shape size of cmp_residual_k must equal q dim0 (2), wrong size should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_cmp_residual_shape_failed)
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
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},      // cmp_residual_k input7 (size 5 != B=2)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND} // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: dtype of cmp_residual_k only supports int32, int64 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_cmp_residual_dtype_failed)
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
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},      // cmp_residual_k input7 (wrong dtype)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND} // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: cmp_residual_k must be null when cmp_ratio is 1
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_cmp_residual_should_null_failed)
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
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},      // cmp_residual_k input7 (should be null)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},        // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND} // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950 TND: cu_seqlens_q is required, missing should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_tnd_cu_seqlens_q_missing_failed)
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
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens_q input3 (null, should fail)
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensKData}, // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                  // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950 BSND: cu_seqlens_q must not be provided
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_bsnd_cu_seqlens_q_provided_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3 (should be null)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                  // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: return_value only supports 0 or 1, return_value=2 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_return_value_invalid_failed)
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
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                        // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
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

// Ascend950: max_seqlen_q must >= -1, -2 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_max_seqlen_q_invalid_failed)
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
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                        // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-2)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: dtype of output_idx_offset only supports int32, int64 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_idx_offset_dtype_failed)
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
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_INT64, ge::FORMAT_ND}, // output_idx_offset input9 (wrong dtype)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}            // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950 BSND: dtype of optional seqused_q only supports int32, int64 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_seqused_q_dtype_failed)
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
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                             // seqused_q    input5 (wrong dtype)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                        // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: gSize (q head num / k head num) must <= 64, q N1=128 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_gsize_limit_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 128, 128}, {2, 39, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // k            input1
            {{{2, 39, 128}, {2, 39, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                 // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                          // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950 PA_BBND: block_size of k must be a multiple of 16, block_size=17 should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_pa_block_size_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    int64_t cuSeqlensQData[] = {0, 39, 78};
    int64_t sequsedKData[] = {32, 32};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{78, 64, 128}, {78, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 17, 1, 128}, {2, 17, 1, 128}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                                 // k (block_size=17, not multiple of 16)
            {{{78, 64}, {78, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w            input2
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, cuSeqlensQData}, // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused_q    input5
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sequsedKData},   // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cmp_residual_k input7
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},                 // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                  // metadata     input10
        },
        {
            {{{78, 1, 2048}, {78, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                      // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: weights shape must match q (B, S1, N1), wrong last dim should fail
TEST_F(LightningIndexerV2TilingArch35, LightningIndexerV2_950_tiling_weights_shape_failed)
{
    struct LIV2CompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LightningIndexerV2",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q            input0
            {{{2, 64, 1, 128}, {2, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // k            input1
            {{{2, 39, 63}, {2, 39, 63}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w (last dim 63 != N1=64)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k input4
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q    input5
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k    input6
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k input7
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // block_table  input8
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // output_idx_offset input9
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}                        // metadata     input10
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}  // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
