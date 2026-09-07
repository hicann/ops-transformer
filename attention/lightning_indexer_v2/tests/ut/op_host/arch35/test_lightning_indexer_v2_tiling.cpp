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
#include <string>
#include <vector>
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
constexpr uint32_t INT32_BIT_WIDTH = 32U;
constexpr int64_t INVALID_TRUNCATED_INT64 = (static_cast<int64_t>(1) << INT32_BIT_WIDTH) + 1;
} // namespace

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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
            {{{2, 39, 1}, {2, 39, 1}}, ge::DT_INT32, ge::FORMAT_ND},               // output_idx_offset input9
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: return_value validation must use the complete int64 value
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INVALID_TRUNCATED_INT64)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Ascend950: max_seqlen_q must fit in int32_t
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
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INVALID_TRUNCATED_INT64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
            {{{2, 39, 1}, {2, 39, 1}}, ge::DT_INT64, ge::FORMAT_ND}, // output_idx_offset input9 (wrong dtype)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}          // metadata     input10
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
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
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

namespace {
constexpr int64_t TEST_BATCH_SIZE = 2;
constexpr int64_t TEST_Q_SEQ_LEN = 4;
constexpr int64_t TEST_K_SEQ_LEN = 16;
constexpr int64_t TEST_Q_HEAD_NUM = 2;
constexpr int64_t TEST_K_HEAD_NUM = 1;
constexpr int64_t TEST_HEAD_DIM = 128;
constexpr int64_t TEST_TOPK = 4;
constexpr int64_t TEST_CMP_RATIO = 4;
constexpr int64_t TEST_MASK_MODE_COMPRESS = 3;
constexpr int64_t TEST_RETURN_INDICES = 0;
constexpr int64_t TEST_BLOCK_COUNT = 2;
constexpr int64_t TEST_BLOCK_SIZE = 16;
constexpr int64_t TEST_BLOCKS_PER_BATCH = 1;
constexpr int64_t TEST_METADATA_SIZE = 1024;
constexpr int64_t TEST_SINGLETON_DIM = 1;
constexpr int64_t TEST_Q_TOKENS = TEST_BATCH_SIZE * TEST_Q_SEQ_LEN;
constexpr int64_t TEST_K_TOKENS = TEST_BATCH_SIZE * TEST_K_SEQ_LEN;
constexpr int64_t TEST_CU_SEQLENS_SIZE = TEST_BATCH_SIZE + 1;
constexpr uint64_t TEST_CORE_NUM = 56;
constexpr uint64_t TEST_UB_SIZE = 262144;
constexpr uint64_t TEST_L2_SIZE = 16384;

enum ShapeInput : size_t {
    QUERY_INPUT = 0,
    KEY_INPUT = 1,
    WEIGHTS_INPUT = 2,
    CU_SEQLENS_Q_INPUT = 3,
    CU_SEQLENS_K_INPUT = 4,
    SEQUSED_Q_INPUT = 5,
    SEQUSED_K_INPUT = 6,
    CMP_RESIDUAL_K_INPUT = 7,
    OUTPUT_IDX_OFFSET_INPUT = 9
};

struct ShapeCase {
    const char *name;
    const char *layout;
    ShapeInput input;
    std::vector<int64_t> shape;
    ge::graphStatus expected = ge::GRAPH_FAILED;
};

gert::TilingContextPara::TensorDescription MakeShapeTensor(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    gert::StorageShape shape;
    shape.MutableShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return {shape, dtype, ge::FORMAT_ND};
}

void RunShapeCase(const ShapeCase &test)
{
    const bool tnd = std::string(test.layout) == "TND" || std::string(test.layout) == "TND_PA";
    const bool paged = std::string(test.layout) == "PA_BBND" || std::string(test.layout) == "TND_PA";
    std::vector<int64_t> keyShape = {TEST_BATCH_SIZE, TEST_K_SEQ_LEN, TEST_K_HEAD_NUM, TEST_HEAD_DIM};
    if (paged) {
        keyShape = {TEST_BLOCK_COUNT, TEST_BLOCK_SIZE, TEST_K_HEAD_NUM, TEST_HEAD_DIM};
    } else if (tnd) {
        keyShape = {TEST_K_SEQ_LEN, TEST_K_HEAD_NUM, TEST_HEAD_DIM};
    }
    std::vector<std::vector<int64_t>> shapes = {
        tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_Q_HEAD_NUM, TEST_HEAD_DIM} :
              std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM},
        keyShape,
        tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_Q_HEAD_NUM} :
              std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM},
        tnd ? std::vector<int64_t>{TEST_CU_SEQLENS_SIZE} : std::vector<int64_t>{},
        tnd && !paged ? std::vector<int64_t>{TEST_CU_SEQLENS_SIZE} : std::vector<int64_t>{},
        {TEST_BATCH_SIZE},
        {TEST_BATCH_SIZE},
        {TEST_BATCH_SIZE},
        paged ? std::vector<int64_t>{TEST_BATCH_SIZE, TEST_BLOCKS_PER_BATCH} : std::vector<int64_t>{},
        tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_K_HEAD_NUM} :
              std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM},
        {TEST_METADATA_SIZE}};
    shapes[test.input] = test.shape;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const ge::DataType dtype =
            i < WEIGHTS_INPUT ? ge::DT_FLOAT16 : (i == WEIGHTS_INPUT ? ge::DT_FLOAT : ge::DT_INT32);
        inputs.push_back(MakeShapeTensor(shapes[i], dtype));
    }
    const auto output = tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_K_HEAD_NUM, TEST_TOPK} :
                              std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM, TEST_TOPK};
    struct CompileInfo {
    } compileInfo;
    gert::TilingContextPara para(
        "LightningIndexerV2", inputs, {MakeShapeTensor(output, ge::DT_INT32), MakeShapeTensor({0}, ge::DT_FLOAT)},
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(TEST_TOPK)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(TEST_Q_SEQ_LEN)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(tnd ? "TND" : "BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(paged ? "PA_BBND" : test.layout)},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(TEST_MASK_MODE_COMPRESS)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(TEST_CMP_RATIO)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(TEST_RETURN_INDICES)}},
        &compileInfo, "Ascend950", TEST_CORE_NUM, TEST_UB_SIZE, TEST_L2_SIZE);
    ExecuteTestCase(para, test.expected, SKIP_TILING_KEY);
}
} // namespace

class LightningIndexerV2ShapeContract : public testing::TestWithParam<ShapeCase> {};

TEST_P(LightningIndexerV2ShapeContract, RejectInvalidShape)
{
    RunShapeCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Ascend950, LightningIndexerV2ShapeContract,
    testing::Values(
        ShapeCase{"BSND_seqused_q_rank2", "BSND", SEQUSED_Q_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"BSND_seqused_q_short", "BSND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"BSND_seqused_q_long", "BSND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"BSND_seqused_k_rank2", "BSND", SEQUSED_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"BSND_seqused_k_short", "BSND", SEQUSED_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"BSND_seqused_k_long", "BSND", SEQUSED_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"BSND_cmp_residual_k_rank2", "BSND", CMP_RESIDUAL_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"BSND_cmp_residual_k_short", "BSND", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"BSND_cmp_residual_k_long", "BSND", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"BSND_q_rank", "BSND", QUERY_INPUT, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"BSND_k_rank", "BSND", KEY_INPUT, {TEST_K_TOKENS, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"BSND_w_rank", "BSND", WEIGHTS_INPUT, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM}},
        ShapeCase{"BSND_w_batch", "BSND", WEIGHTS_INPUT, {TEST_BATCH_SIZE - 1, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM}},
        ShapeCase{"BSND_output_idx_offset_rank", "BSND", OUTPUT_IDX_OFFSET_INPUT, {TEST_Q_TOKENS, TEST_K_HEAD_NUM}},
        ShapeCase{"BSND_valid", "BSND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"BSND_optional_q_absent", "BSND", SEQUSED_Q_INPUT, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"BSND_optional_k_absent", "BSND", SEQUSED_K_INPUT, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_seqused_q_rank2", "TND", SEQUSED_Q_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_seqused_q_short", "TND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_seqused_q_long", "TND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_seqused_k_rank2", "TND", SEQUSED_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_seqused_k_short", "TND", SEQUSED_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_seqused_k_long", "TND", SEQUSED_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_cmp_residual_k_rank2", "TND", CMP_RESIDUAL_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_cmp_residual_k_short", "TND", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_cmp_residual_k_long", "TND", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_q_rank", "TND", QUERY_INPUT, {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"TND_k_rank",
                  "TND",
                  KEY_INPUT,
                  {TEST_BATCH_SIZE, TEST_K_SEQ_LEN / TEST_BATCH_SIZE, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"TND_w_rank", "TND", WEIGHTS_INPUT, {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM}},
        ShapeCase{"TND_output_idx_offset_rank",
                  "TND",
                  OUTPUT_IDX_OFFSET_INPUT,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM}},
        ShapeCase{"TND_valid", "TND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_optional_q_absent", "TND", SEQUSED_Q_INPUT, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_optional_k_absent", "TND", SEQUSED_K_INPUT, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"PA_BBND_seqused_q_rank2", "PA_BBND", SEQUSED_Q_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"PA_BBND_seqused_q_short", "PA_BBND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"PA_BBND_seqused_q_long", "PA_BBND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"PA_BBND_seqused_k_rank2", "PA_BBND", SEQUSED_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"PA_BBND_seqused_k_short", "PA_BBND", SEQUSED_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"PA_BBND_seqused_k_long", "PA_BBND", SEQUSED_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{
            "PA_BBND_cmp_residual_k_rank2", "PA_BBND", CMP_RESIDUAL_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"PA_BBND_cmp_residual_k_short", "PA_BBND", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"PA_BBND_cmp_residual_k_long", "PA_BBND", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"PA_BBND_q_rank", "PA_BBND", QUERY_INPUT, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"PA_BBND_k_rank", "PA_BBND", KEY_INPUT, {TEST_K_TOKENS, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"PA_BBND_w_rank", "PA_BBND", WEIGHTS_INPUT, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM}},
        ShapeCase{
            "PA_BBND_output_idx_offset_rank", "PA_BBND", OUTPUT_IDX_OFFSET_INPUT, {TEST_Q_TOKENS, TEST_K_HEAD_NUM}},
        ShapeCase{"PA_BBND_valid", "PA_BBND", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"PA_BBND_optional_q_absent", "PA_BBND", SEQUSED_Q_INPUT, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_PA_seqused_q_rank2", "TND_PA", SEQUSED_Q_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_PA_seqused_q_short", "TND_PA", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_PA_seqused_q_long", "TND_PA", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_PA_seqused_k_rank2", "TND_PA", SEQUSED_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_PA_seqused_k_short", "TND_PA", SEQUSED_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_PA_seqused_k_long", "TND_PA", SEQUSED_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_PA_cmp_residual_k_rank2", "TND_PA", CMP_RESIDUAL_K_INPUT, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_PA_cmp_residual_k_short", "TND_PA", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_PA_cmp_residual_k_long", "TND_PA", CMP_RESIDUAL_K_INPUT, {TEST_BATCH_SIZE + 1}},
        ShapeCase{
            "TND_PA_q_rank", "TND_PA", QUERY_INPUT, {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"TND_PA_k_rank", "TND_PA", KEY_INPUT, {TEST_K_TOKENS, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"TND_PA_w_rank", "TND_PA", WEIGHTS_INPUT, {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM}},
        ShapeCase{"TND_PA_output_idx_offset_rank",
                  "TND_PA",
                  OUTPUT_IDX_OFFSET_INPUT,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM}},
        ShapeCase{"TND_PA_valid", "TND_PA", SEQUSED_Q_INPUT, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_PA_optional_q_absent", "TND_PA", SEQUSED_Q_INPUT, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_cu_seqlens_q_rank2", "TND", CU_SEQLENS_Q_INPUT, {TEST_SINGLETON_DIM, TEST_CU_SEQLENS_SIZE}},
        ShapeCase{"TND_cu_seqlens_k_rank2", "TND", CU_SEQLENS_K_INPUT, {TEST_SINGLETON_DIM, TEST_CU_SEQLENS_SIZE}},
        ShapeCase{"BSND_cu_seqlens_k_forbidden", "BSND", CU_SEQLENS_K_INPUT, {TEST_CU_SEQLENS_SIZE}}),
    [](const testing::TestParamInfo<ShapeCase> &info) { return info.param.name; });
