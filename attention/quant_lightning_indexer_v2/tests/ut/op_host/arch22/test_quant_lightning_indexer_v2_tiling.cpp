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
#include "register/tilingdata_base.h"
#include "../test_quant_lightning_indexer_v2_utils.h"

// DAV_2201 (Ascend910B) tiling cases for QuantLightningIndexerV2
class QuantLightningIndexerV2TilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch22 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch22 TearDown" << std::endl;
    }
};

// BSND/PA_BBND int8 success on Ascend910B: quant_mode=2, topk=2048, mask_mode=0
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_int8_pa_success)
{
    qliv2_ut::CaseParam p;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// BSND/PA_BBND int8 success with cmp_residual_k and output_idx_offset on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_cmp_residual_success)
{
    qliv2_ut::CaseParam p;
    p.cmpRatio = 4;
    p.maskMode = 3;
    p.cmpResidual = {2};
    p.idxOffset = {2, 39, 64};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// layout_k only supports PA_BBND on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_layout_k_failed)
{
    qliv2_ut::CaseParam p;
    p.layoutK = "BSND";
    p.blockTable = {};
    p.sequsedK = {};
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// quant_mode only supports 2 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_quant_mode_failed)
{
    qliv2_ut::CaseParam p;
    p.quantMode = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// return_value only supports false on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_return_value_failed)
{
    qliv2_ut::CaseParam p;
    p.returnValue = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// topk must > 0 and <= 2048 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_topk_over_limit_failed)
{
    qliv2_ut::CaseParam p;
    p.topk = 4096;
    p.outShape = {2, 39, 1, 4096};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio must be a power of 2 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_cmp_ratio_not_pow2_failed)
{
    qliv2_ut::CaseParam p;
    p.cmpRatio = 3;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be int8 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_q_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.qType = ge::DT_FLOAT16;
    p.kType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q_descale and k_descale must be float16 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_scale_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of w must be float16 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_w_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.wType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// gSize (N1/N2) must equal 64 on Ascend910B
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_gsize_failed)
{
    qliv2_ut::CaseParam p;
    p.qShape = {2, 39, 128, 128};
    p.wShape = {2, 39, 128};
    p.qScaleShape = {2, 39, 128};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// mask_mode only supports 0 or 3
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_mask_mode_failed)
{
    qliv2_ut::CaseParam p;
    p.maskMode = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// metadata is required
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_metadata_missing_failed)
{
    qliv2_ut::CaseParam p;
    p.metadata = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be same: q=int8, k=fp8 should fail
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_qk_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of sparse_values must be bfloat16
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_values_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.valuesType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head dim of q only supports 128
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_head_dim_failed)
{
    qliv2_ut::CaseParam p;
    p.qShape = {2, 39, 64, 127};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// block_size of k must be a multiple of 16
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_block_size_failed)
{
    qliv2_ut::CaseParam p;
    p.kShape = {2, 17, 1, 128};
    p.kScaleShape = {2, 17, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head num of k only supports 1
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_k_headnum_failed)
{
    qliv2_ut::CaseParam p;
    p.kShape = {2, 16, 2, 128};
    p.kScaleShape = {2, 16, 2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// unsupported npu arch (Ascend310P) should fail
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_tiling_unsupported_arch_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend310P";
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// Tiling data classes are registered for the op
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_tiling_data_class_registered)
{
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("QuantLightningIndexerV2"), nullptr);
}

namespace {
// Base of a valid Ascend910B TND/PA_BBND int8 case
qliv2_ut::CaseParam Make910bTndPaInt8()
{
    qliv2_ut::CaseParam p;
    p.qShape = {78, 64, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.cuSeqQ = {3};
    p.layoutQ = "TND";
    p.outShape = {78, 1, 2048};
    return p;
}
} // namespace

// TND/PA_BBND int8 success on Ascend910B: quant_mode=2
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_pa_success)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND on Ascend910B: cu_seqlens_q is required, missing should fail
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_cu_seqlens_q_missing_failed)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    p.cuSeqQ = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND on Ascend910B: shape size of cu_seqlens_q should be greater than 1 (B+1), size=1 should fail
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_cu_seqlens_q_size_one_failed)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    p.cuSeqQ = {1};
    p.sequsedK = {0};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND/PA_BBND on Ascend910B: shape size of cu_seqlens_q must equal seqused_k size + 1 (3 vs 2+1 mismatch)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_pa_seqused_k_size_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    p.cmpRatio = 4;
    p.maskMode = 3;
    p.cmpResidual = {2};
    p.sequsedK = {3};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND on Ascend910B: dtype of seqused_q only supports int32 when provided (int64 should fail)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_seqused_q_dtype_failed)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    p.sequsedQ = {2};
    p.sequsedQType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// BSND on Ascend910B: dtype of seqused_q only supports int32 when provided (int64 should fail)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_bsnd_seqused_q_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.sequsedQ = {2};
    p.sequsedQType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND on Ascend910B: shape size of cmp_residual_k must equal cu_seqlens_q size - 1 (mismatch should fail)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_cmp_residual_size_failed)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    p.cmpRatio = 4;
    p.maskMode = 3;
    p.cmpResidual = {3};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND on Ascend910B: dim 0 of q, w and sparse_indices must be same (w T=40 should fail)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_tnd_w_t_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make910bTndPaInt8();
    p.wShape = {40, 64};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be same and int8 on Ascend910B (e5m2 vs int8 should fail and hit unknown-dtype logging)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_q_dtype_e5m2_failed)
{
    qliv2_ut::CaseParam p;
    p.qType = ge::DT_FLOAT8_E5M2;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// PA_BBND on Ascend910B: key stride0 must be positive (0 should fail)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_pa_k_stride0_zero_failed)
{
    struct QLIV2CompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        qliv2_ut::Desc({2, 39, 64, 128}, ge::DT_INT8), // q              input0
        qliv2_ut::Desc({2, 16, 1, 128}, ge::DT_INT8),  // k              input1
        qliv2_ut::Desc({2, 39, 64}, ge::DT_FLOAT16),   // w              input2
        qliv2_ut::Desc({2, 39, 64}, ge::DT_FLOAT16),   // q_descale      input3
        qliv2_ut::Desc({2, 16, 1}, ge::DT_FLOAT16),    // k_descale      input4
        qliv2_ut::Desc({}, ge::DT_INT32),              // cu_seqlens_q   input5
        qliv2_ut::Desc({}, ge::DT_INT32),              // cu_seqlens_k   input6
        qliv2_ut::Desc({}, ge::DT_INT32),              // seqused_q      input7
        qliv2_ut::Desc({2}, ge::DT_INT32),             // seqused_k      input8
        qliv2_ut::Desc({}, ge::DT_INT32),              // cmp_residual_k input9
        qliv2_ut::Desc({2, 2}, ge::DT_INT32),          // block_table    input10
        qliv2_ut::Desc({}, ge::DT_INT32),              // output_idx_offset input11
        qliv2_ut::Desc({1024}, ge::DT_INT32)           // metadata       input12
    };
    inputs[1].stride_ = gert::Stride({0, 128, 128, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara para("QuantLightningIndexerV2", inputs,
                                 {qliv2_ut::Desc({2, 39, 1, 2048}, ge::DT_INT32), qliv2_ut::Desc({0}, ge::DT_BF16)},
                                 {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                                  {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                  {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                  {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
                                  {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                 &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED, qliv2_ut::SKIP_TILING_KEY);
}

// PA_BBND success on Ascend910B with contiguous k and k_descale strides: covers stride parsing path
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_pa_k_kdescale_stride_success)
{
    struct QLIV2CompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        qliv2_ut::Desc({2, 39, 64, 128}, ge::DT_INT8), // q              input0
        qliv2_ut::Desc({2, 16, 1, 128}, ge::DT_INT8),  // k              input1
        qliv2_ut::Desc({2, 39, 64}, ge::DT_FLOAT16),   // w              input2
        qliv2_ut::Desc({2, 39, 64}, ge::DT_FLOAT16),   // q_descale      input3
        qliv2_ut::Desc({2, 16, 1}, ge::DT_FLOAT16),    // k_descale      input4
        qliv2_ut::Desc({}, ge::DT_INT32),              // cu_seqlens_q   input5
        qliv2_ut::Desc({}, ge::DT_INT32),              // cu_seqlens_k   input6
        qliv2_ut::Desc({}, ge::DT_INT32),              // seqused_q      input7
        qliv2_ut::Desc({2}, ge::DT_INT32),             // seqused_k      input8
        qliv2_ut::Desc({}, ge::DT_INT32),              // cmp_residual_k input9
        qliv2_ut::Desc({2, 2}, ge::DT_INT32),          // block_table    input10
        qliv2_ut::Desc({}, ge::DT_INT32),              // output_idx_offset input11
        qliv2_ut::Desc({1024}, ge::DT_INT32)           // metadata       input12
    };
    inputs[1].stride_ = gert::Stride({2048, 128, 128, 1});
    inputs[1].hasStride_ = true;
    inputs[4].stride_ = gert::Stride({16, 1, 1});
    inputs[4].hasStride_ = true;
    gert::TilingContextPara para("QuantLightningIndexerV2", inputs,
                                 {qliv2_ut::Desc({2, 39, 1, 2048}, ge::DT_INT32), qliv2_ut::Desc({0}, ge::DT_BF16)},
                                 {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                                  {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                  {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                  {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
                                  {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                 &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, qliv2_ut::SKIP_TILING_KEY);
}

// PA_BBND on Ascend910B: k only supports non-continuous keying on the 0-axis (axis 1 mismatch should fail)
TEST_F(QuantLightningIndexerV2TilingArch22, QuantLightningIndexerV2_910b_tiling_pa_k_noncontiguous_stride_failed)
{
    struct QLIV2CompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        qliv2_ut::Desc({2, 39, 64, 128}, ge::DT_INT8), // q              input0
        qliv2_ut::Desc({2, 16, 1, 128}, ge::DT_INT8),  // k              input1
        qliv2_ut::Desc({2, 39, 64}, ge::DT_FLOAT16),   // w              input2
        qliv2_ut::Desc({2, 39, 64}, ge::DT_FLOAT16),   // q_descale      input3
        qliv2_ut::Desc({2, 16, 1}, ge::DT_FLOAT16),    // k_descale      input4
        qliv2_ut::Desc({}, ge::DT_INT32),              // cu_seqlens_q   input5
        qliv2_ut::Desc({}, ge::DT_INT32),              // cu_seqlens_k   input6
        qliv2_ut::Desc({}, ge::DT_INT32),              // seqused_q      input7
        qliv2_ut::Desc({2}, ge::DT_INT32),             // seqused_k      input8
        qliv2_ut::Desc({}, ge::DT_INT32),              // cmp_residual_k input9
        qliv2_ut::Desc({2, 2}, ge::DT_INT32),          // block_table    input10
        qliv2_ut::Desc({}, ge::DT_INT32),              // output_idx_offset input11
        qliv2_ut::Desc({1024}, ge::DT_INT32)           // metadata       input12
    };
    inputs[1].stride_ = gert::Stride({2048, 256, 128, 1});
    inputs[1].hasStride_ = true;
    gert::TilingContextPara para("QuantLightningIndexerV2", inputs,
                                 {qliv2_ut::Desc({2, 39, 1, 2048}, ge::DT_INT32), qliv2_ut::Desc({0}, ge::DT_BF16)},
                                 {{"topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                                  {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                  {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                                  {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BBND")},
                                  {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                  {"return_value", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                 &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED, qliv2_ut::SKIP_TILING_KEY);
}
