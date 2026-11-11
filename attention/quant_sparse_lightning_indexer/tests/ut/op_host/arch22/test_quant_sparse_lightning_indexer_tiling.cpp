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
#include "register/tilingdata_base.h"
#include "../test_quant_sparse_lightning_indexer_utils.h"

// DAV_2201（Ascend910B/910_93）tiling 用例：QuantSparseLightningIndexer 是两级 TopK 的
// 第二级 consumer 算子，候选块索引 candidate_block_indices 为必传输入，末维即候选块个数（当前仅支持 2048）。
class QuantSparseLightningIndexerTilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantSparseLightningIndexerTilingArch22 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantSparseLightningIndexerTilingArch22 TearDown" << std::endl;
    }
};

// ---------------- 成功用例 ----------------

// BSND/PA_BBND INT8 基础用例：quant_mode=2, topk=2048, mask_mode=0
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_int8_pa_success)
{
    qsli_ut::CaseParam p;
    qsli_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// 带 cmp_residual_k、mask_mode=3 与 output_idx_offset 的成功用例
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cmp_residual_success)
{
    qsli_ut::CaseParam p;
    p.cmpRatio = 4;
    p.maskMode = 3;
    p.cmpResidual = {2};
    p.idxOffset = {2, 39, 64};
    qsli_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND/PA_BBND 成功用例：T 维为变长拼接总长，输出 [T, N2, K]
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_tnd_pa_success)
{
    qsli_ut::CaseParam p;
    p.qShape = {78, 64, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.cuSeqQ = {3};
    p.layoutQ = "TND";
    p.outShape = {78, 1, 2048};
    p.candIdxShape = {78, 1, 2048};
    qsli_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND/PA_BBND: cu_seqlens_q 长度必须等于 seqused_k 长度 + 1, 不等应失败。
// cmp_ratio=1 时 cmp_residual_k 为 nullptr, 旧实现错误日志解引用它有崩溃风险
// (检视意见②修复): 现打印被校验的 cu_seqlens_q/seqused_k 长度, 不得解引用 cmpResidualK。
TEST_F(QuantSparseLightningIndexerTilingArch22,
       QuantSparseLightningIndexer_910b_tiling_tnd_pa_cu_seq_size_mismatch_failed)
{
    qsli_ut::CaseParam p;
    p.qShape = {78, 64, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.cuSeqQ = {3};         // cu_seqlens_q 长度 3(B=2)
    p.sequsedK = {7, 8, 9}; // seqused_k 长度 3(B=3), 与 cu_seqlens_q 长度不一致
    p.layoutQ = "TND";
    p.outShape = {78, 1, 2048};
    p.candIdxShape = {78, 1, 2048};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// candidate_block_length 为预留接口：传空 tensor（shape {0}）应通过
TEST_F(QuantSparseLightningIndexerTilingArch22,
       QuantSparseLightningIndexer_910b_tiling_candidate_block_length_empty_success)
{
    qsli_ut::CaseParam p;
    p.candBlkLenShape = {0};
    qsli_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// ---------------- candidate（两级 TopK）相关拒绝用例 ----------------

// candidate_block_indices 末维（候选块个数）只支持 2048
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cand_blocks_failed)
{
    qsli_ut::CaseParam p;
    p.candIdxShape = {2, 39, 1, 1024};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// candidate_block_indices 为必传输入，缺失应失败
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cand_idx_missing_failed)
{
    qsli_ut::CaseParam p;
    p.candIdxShape = {};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// candidate_block_indices 只支持 int32
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cand_idx_dtype_failed)
{
    qsli_ut::CaseParam p;
    p.candIdxType = ge::DT_INT64;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// candidate_block_indices 元素个数必须等于 B*S1*N2*候选块个数
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cand_idx_size_failed)
{
    qsli_ut::CaseParam p;
    // S1=20 与 q 的 S1=39 不一致：元素个数 != 2*39*1*2048
    p.candIdxShape = {2, 20, 1, 2048};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// candidate_block_size 当前仅支持 8
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cand_block_size_failed)
{
    qsli_ut::CaseParam p;
    p.candBlockSize = 16;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// candidate_block_length 只接受空 tensor，传非空应失败
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cand_block_length_failed)
{
    qsli_ut::CaseParam p;
    p.candBlkLenShape = {8};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND 布局必须搭配 layout_k=PA_BBND
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_tnd_layout_k_failed)
{
    qsli_ut::CaseParam p;
    p.qShape = {78, 64, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.cuSeqQ = {3};
    p.layoutQ = "TND";
    p.layoutK = "BSND";
    p.kShape = {2, 16, 1, 128};
    p.outShape = {78, 1, 2048};
    p.candIdxShape = {78, 1, 2048};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND 布局必须传 cu_seqlens_q
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_tnd_cu_seqlens_q_failed)
{
    qsli_ut::CaseParam p;
    p.qShape = {78, 64, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.layoutQ = "TND";
    p.outShape = {78, 1, 2048};
    p.candIdxShape = {78, 1, 2048};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// ---------------- 基础属性 / 形状拒绝用例 ----------------

// layout_k 在 910B 上只支持 PA_BBND
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_layout_k_failed)
{
    qsli_ut::CaseParam p;
    p.layoutK = "BSND";
    p.blockTable = {};
    p.sequsedK = {};
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// quant_mode 只支持 2
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_quant_mode_failed)
{
    qsli_ut::CaseParam p;
    p.quantMode = 1;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// return_value 在 910B 上只支持 0
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_return_value_failed)
{
    qsli_ut::CaseParam p;
    p.returnValue = 1;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// topk 必须 > 0 且 <= 2048
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_topk_over_limit_failed)
{
    qsli_ut::CaseParam p;
    p.topk = 4096;
    p.outShape = {2, 39, 1, 4096};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio 必须是 2 的幂
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_cmp_ratio_not_pow2_failed)
{
    qsli_ut::CaseParam p;
    p.cmpRatio = 3;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// mask_mode 只支持 0 或 3
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_mask_mode_failed)
{
    qsli_ut::CaseParam p;
    p.maskMode = 1;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// q 与 k 的 dtype 必须一致且为 int8
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_qk_dtype_mismatch_failed)
{
    qsli_ut::CaseParam p;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// q 与 k 的 dtype 只支持 int8
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_q_dtype_failed)
{
    qsli_ut::CaseParam p;
    p.qType = ge::DT_FLOAT16;
    p.kType = ge::DT_FLOAT16;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// q_descale 与 k_descale 只支持 float16
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_scale_dtype_failed)
{
    qsli_ut::CaseParam p;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// w 只支持 float16
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_w_dtype_failed)
{
    qsli_ut::CaseParam p;
    p.wType = ge::DT_FLOAT;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// sparse_values 只支持 bfloat16
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_values_dtype_failed)
{
    qsli_ut::CaseParam p;
    p.valuesType = ge::DT_FLOAT16;
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// q 的 head dim 只支持 128
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_head_dim_failed)
{
    qsli_ut::CaseParam p;
    p.qShape = {2, 39, 64, 127};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// N1/N2 必须是 64 或 32
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_gsize_failed)
{
    qsli_ut::CaseParam p;
    p.qShape = {2, 39, 128, 128};
    p.wShape = {2, 39, 128};
    p.qScaleShape = {2, 39, 128};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// k 的 block_size 必须是 16 的整数倍
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_block_size_failed)
{
    qsli_ut::CaseParam p;
    p.kShape = {2, 17, 1, 128};
    p.kScaleShape = {2, 17, 1};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// k 的 head 数只支持 1
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_k_headnum_failed)
{
    qsli_ut::CaseParam p;
    p.kShape = {2, 16, 2, 128};
    p.kScaleShape = {2, 16, 2};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// PA_BBND 下 k 的 stride0 必须为正（0 应失败）
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_910b_tiling_pa_k_stride0_zero_failed)
{
    qsli_ut::CaseParam p;
    p.kStride = {0, 128, 128, 1};
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// 不支持的架构（Ascend310P）应失败
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_tiling_unsupported_arch_failed)
{
    qsli_ut::CaseParam p;
    p.soc = "Ascend310P";
    qsli_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// tiling data 类已注册
TEST_F(QuantSparseLightningIndexerTilingArch22, QuantSparseLightningIndexer_tiling_data_class_registered)
{
    auto& factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("QuantSparseLightningIndexer"), nullptr);
}
