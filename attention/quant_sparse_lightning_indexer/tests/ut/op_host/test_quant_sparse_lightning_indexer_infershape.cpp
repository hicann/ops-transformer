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
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

// QuantSparseLightningIndexer 是两级 TopK（two-level TopK）的第二级 consumer 算子：
// 输出双布局为 [B, S1, N2, K]（BSND）或 [T, N2, K]（TND），K 取属性 topk。
// consumer 侧没有 candidate_block_indices 输出，因此只有 2 个输出。
class QuantSparseLightningIndexerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantSparseLightningIndexerProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantSparseLightningIndexerProto TearDown" << std::endl;
    }
};

// BSND/PA_BBND, return_value=0, topk=2048: sparse_values 为空张量 (0,)
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_bsnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        // 输入Tensor (15个)
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k (PA)       input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, // metadata     input12 (METADATA_LIMIT=1024)
            {{{1, 8, 1, 2048}, {1, 8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input14 (optional)
        },
        // 输出Tensor
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        // 属性
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 1, 2048}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND/PA_BBND, return_value=0, topk=2048: 输出 [T, N2, K]，N2 取 k 的第 2 维
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_tnd_pa)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        {
            {{{64, 32, 128}, {64, 32, 128}}, ge::DT_INT8, ge::FORMAT_ND},     // q            input0
            {{{1, 16, 1, 128}, {1, 16, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k (PA)       input1
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},            // w            input2
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},            // q_descale    input3
            {{{1, 16, 1}, {1, 16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                        // cu_seqlens_q input5
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},               // metadata     input12 (METADATA_LIMIT=1024)
            {{{64, 1, 2048}, {64, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true}                // candidate_block_length input14 (optional)
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 1, 2048}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// return_value=1 时 sparse_values 与 sparse_indices 同形（arch22 由 tiling 拒绝，infershape 仍按同形推导）
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_return_value)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k (PA)       input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, // metadata     input12 (METADATA_LIMIT=1024)
            {{{1, 8, 1, 2048}, {1, 8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input14 (optional)
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 1, 256}, {1, 8, 1, 256}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// layout_q 只支持 BSND/TND，非法取值应失败
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_layout_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k (PA)       input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, // metadata     input12 (METADATA_LIMIT=1024)
            {{{1, 8, 1, 2048}, {1, 8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input14 (optional)
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// layout_k 只支持 BSND/TND/PA_BBND，非法取值应在维度推导前失败
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_layout_k_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},   // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, // metadata     input12 (METADATA_LIMIT=1024)
            {{{1, 8, 1, 2048}, {1, 8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input14 (optional)
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// layout_k=BSND 时 k 必须为 4 维，低秩 shape 应失败（避免越界取维）
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_key_rank_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        {
            {{{1, 8, 8, 128}, {1, 8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // q            input0
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},     // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                // cu_seqlens_q input5  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                      // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, // metadata     input12 (METADATA_LIMIT=1024)
            {{{1, 8, 1, 2048}, {1, 8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input14 (optional)
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// layout_q=BSND 时 q 必须为 4 维，低秩 shape 应失败
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_infershape_query_rank_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "QuantSparseLightningIndexer",
        {
            {{{8, 8, 128}, {8, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},         // q            input0
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND}, // k            input1
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // w            input2
            {{{1, 8, 8}, {1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // q_descale    input3
            {{{1, 64, 1}, {1, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // k_descale    input4
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_q input5  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cu_seqlens_k input6  (optional)
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // seqused_q    input7  (optional)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                        // seqused_k    input8
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // cmp_residual_k input9 (optional)
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table  input10
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true},                  // output_idx_offset input11 (optional)
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, // metadata     input12 (METADATA_LIMIT=1024)
            {{{1, 8, 1, 2048}, {1, 8, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // candidate_block_indices input13
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true} // candidate_block_length input14 (optional)
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
         {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// infer dataType: sparse_indices 固定 int32
TEST_F(QuantSparseLightningIndexerProto, QuantSparseLightningIndexer_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("QuantSparseLightningIndexer")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType inputQ = ge::DT_INT8;
        ge::DataType inputK = ge::DT_INT8;
        ge::DataType inputW = ge::DT_FLOAT16;
        ge::DataType inputScale = ge::DT_FLOAT16;
        ge::DataType inputI32 = ge::DT_INT32;
        ge::DataType outputRef0 = ge::DT_INT32;
        ge::DataType outputRef1 = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(15, 2)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&inputQ, &inputK, &inputW, &inputScale, &inputScale, &inputI32, &inputI32, &inputI32,
                                 &inputI32, &inputI32, &inputI32, &inputI32, &inputI32, &inputI32, &inputI32})
                .NodeAttrs({{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
                            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                            {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"candidate_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef0);
        EXPECT_EQ(context->GetOutputDataType(1), outputRef1);
    }
}
