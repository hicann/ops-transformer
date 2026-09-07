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
#include "../test_quant_lightning_indexer_v2_utils.h"

// DAV_3510 (Ascend950) tiling cases for QuantLightningIndexerV2
class QuantLightningIndexerV2TilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingArch35 TearDown" << std::endl;
    }
};

namespace {
constexpr uint32_t INT32_BIT_WIDTH = 32U;
constexpr int64_t INVALID_TRUNCATED_INT64 = (static_cast<int64_t>(1) << INT32_BIT_WIDTH) + 1;
constexpr uint64_t TEST_CORE_NUM = 56;

// Base of a valid Ascend950 TND/PA_BBND fp8 case
qliv2_ut::CaseParam Make950TndPaFp8()
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {2, 16, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.kScaleShape = {2, 16, 1};
    p.outShape = {78, 1, 2048};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.layoutQ = "TND";
    p.layoutK = "PA_BBND";
    p.quantMode = 1;
    p.maxSeqlenQ = 64;
    p.maskMode = 3;
    return p;
}
} // namespace

// TND/PA_BBND fp8 success on Ascend950: quant_mode=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_tnd_pa_success)
{
    qliv2_ut::RunTilingCase(Make950TndPaFp8(), ge::GRAPH_SUCCESS);
}

// PA_BBND is not a valid query layout, including for low-rank query shapes.
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_query_pa_layout_failed)
{
    const qliv2_ut::CaseParam base = Make950TndPaFp8();
    for (size_t rank = 0; rank <= base.qShape.size(); ++rank) {
        SCOPED_TRACE(rank);
        qliv2_ut::CaseParam p = base;
        p.layoutQ = "PA_BBND";
        p.qShape.resize(rank);
        qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
    }
}

// TND/PA_BBND fp8 success with cmp_residual_k and output_idx_offset on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_cmp_residual_success)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 4;
    p.cmpResidual = {2};
    p.idxOffset = {78, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND output_idx_offset must match T1 and N2
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_tnd_idx_offset_shape_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.idxOffset = p.outShape;
    p.idxOffset.pop_back();
    p.idxOffset.back() += 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// BSND/BSND mxfp8 success on Ascend950: quant_mode=3, e8m0 scale
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_mxfp8_bsnd_bsnd_success)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1, 2, 2};
    p.qScaleShape = {2, 39, 64, 2, 2};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    p.layoutK = "BSND";
    p.quantMode = 3;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// BSND q and k must have the same batch size
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_bsnd_batch_mismatch_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = TEST_CORE_NUM;
    p.kShape.front() = p.qShape.front() + 1;
    p.kScaleShape.front() = p.kShape.front();
    p.layoutK = "BSND";
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// BSND output_idx_offset must match B, S1 and N2
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_bsnd_idx_offset_shape_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = TEST_CORE_NUM;
    p.idxOffset = p.outShape;
    p.idxOffset.pop_back();
    p.idxOffset.back() += 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND/TND hifloat8 success on Ascend950: quant_mode=4, return_value=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_hif8_tnd_tnd_success)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {128, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {1};
    p.kScaleShape = {1};
    p.outShape = {78, 1, 2048};
    p.valuesShape = {78, 1, 2048};
    p.qType = ge::DT_HIFLOAT8;
    p.kType = ge::DT_HIFLOAT8;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.cuSeqK = {3};
    p.layoutQ = "TND";
    p.layoutK = "TND";
    p.quantMode = 4;
    p.maxSeqlenQ = 64;
    p.returnValue = 1;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// BSND/PA_BBND int8 success on Ascend950: quant_mode=2, return_value=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_int8_pa_rv_success)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.valuesShape = {2, 39, 1, 2048};
    p.returnValue = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// TND/PA_BBND mxfp4 success on Ascend950: quant_mode=5
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_mxfp4_tnd_pa_success)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qType = ge::DT_FLOAT4_E2M1;
    p.kType = ge::DT_FLOAT4_E2M1;
    p.qScaleShape = {78, 64, 2, 2};
    p.kScaleShape = {2, 16, 1, 2, 2};
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    p.quantMode = 5;
    p.maskMode = 0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_SUCCESS);
}

// layout_k only supports PA_BBND, BSND or TND on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_layout_k_invalid_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "XXX";
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// outside of PA, layout_q and layout_k must be the same
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_layout_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "BSND";
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// topk must > 0 and <= 8192 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_topk_over_limit_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.topk = 10000;
    p.outShape = {78, 1, 10000};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio must > 0 and <= 128 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cmp_ratio_over_limit_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 200;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// quant_mode only supports 1-5 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_quant_mode_invalid_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.quantMode = 6;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// quant_mode validation must use the complete int64 value
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_quant_mode_high_bits_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.quantMode = INVALID_TRUNCATED_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// return_value only supports 0 or 1 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_return_value_invalid_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.returnValue = 2;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// max_seqlen_q must >= -1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_max_seqlen_q_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.maxSeqlenQ = -2;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q must match quant_mode: int8 with quant_mode=1 should fail
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_q_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qType = ge::DT_INT8;
    p.kType = ge::DT_INT8;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q_descale must match quant_mode: e8m0 with quant_mode=1 should fail
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_scale_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// when q is int8 (quant_mode=2), dtype of w must be float16
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_int8_w_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.quantMode = 2;
    p.wType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// when q is not int8 (quant_mode=1), dtype of w must be float
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_w_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.wType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q and k must be same
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_qk_dtype_mismatch_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kType = ge::DT_INT8;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of q_descale and k_descale must be same
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_scale_dtype_not_equal_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kScaleType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of sparse_indices must be int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_out_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.outType = ge::DT_FLOAT;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of sparse_values must be bfloat16
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_values_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.valuesType = ge::DT_FLOAT16;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// PA_BBND requires block_table
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_pa_block_table_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// PA_BBND must not provide cu_seqlens_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_pa_cu_seqlens_k_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cuSeqK = {2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND k requires cu_seqlens_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_tnd_k_cu_seqlens_k_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "TND";
    p.kShape = {128, 1, 128};
    p.kScaleShape = {128, 1};
    p.blockTable = {};
    p.sequsedK = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// BSND k must not provide cu_seqlens_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_bsnd_k_cu_seqlens_k_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "BSND";
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    p.blockTable = {};
    p.sequsedK = {};
    p.cuSeqK = {2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// non-PA layout must not provide block_table
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_bsnd_block_table_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "TND";
    p.kShape = {128, 1, 128};
    p.kScaleShape = {128, 1};
    p.sequsedK = {};
    p.cuSeqK = {3};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// cmp_ratio != 1 and mask_mode != 0 require cmp_residual_k
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cmp_residual_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 4;
    p.cmpResidual = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// TND q requires cu_seqlens_q
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_tnd_cu_seqlens_q_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cuSeqQ = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// metadata is required on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_metadata_missing_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.metadata = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// metadata shape size must be 1024
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_metadata_size_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.metadata = {512};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// mxfp8 scale dim num must be q dim num + 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_mxfp8_scale_dim_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.kShape = {2, 64, 1, 128};
    p.kScaleShape = {2, 64, 1};
    p.qScaleShape = {2, 39, 64};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT8_E8M0;
    p.kScaleType = ge::DT_FLOAT8_E8M0;
    p.layoutK = "BSND";
    p.quantMode = 3;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// hifloat8 scale dim num must be 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_hif8_scale_dim_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {128, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.kScaleShape = {1};
    p.outShape = {78, 1, 2048};
    p.qType = ge::DT_HIFLOAT8;
    p.kType = ge::DT_HIFLOAT8;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.cuSeqK = {3};
    p.layoutQ = "TND";
    p.layoutK = "TND";
    p.quantMode = 4;
    p.maxSeqlenQ = 64;
    p.sequsedK = {};
    p.blockTable = {};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// fp8 scale dim num must be q dim num - 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_fp8_scale_dim_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qScaleShape = {78};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head num of k only supports 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_k_headnum_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kShape = {2, 16, 2, 128};
    p.kScaleShape = {2, 16, 2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// gSize must <= 64 on Ascend950
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_gsize_over_limit_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qShape = {2, 39, 128, 128};
    p.wShape = {2, 39, 128};
    p.qScaleShape = {2, 39, 128};
    p.outShape = {2, 39, 1, 2048};
    p.layoutQ = "BSND";
    p.cuSeqQ = {};
    p.maskMode = 0;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head num of q must be at least 1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_q_headnum_zero_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = TEST_CORE_NUM;
    p.wShape.back() = 0;
    p.qScaleShape.back() = 0;
    p.qShape = p.wShape;
    p.qShape.push_back(p.kShape.back());
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// block_size of k must be a multiple of 16
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_block_size_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.kShape = {2, 17, 1, 128};
    p.kScaleShape = {2, 17, 1};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of cu_seqlens_q only supports int32 (TND/TND so that cu_seqlens_k desc is valid for error logging)
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cu_seqlens_q_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.layoutK = "TND";
    p.kShape = {128, 1, 128};
    p.kScaleShape = {128, 1};
    p.blockTable = {};
    p.sequsedK = {};
    p.cuSeqK = {3};
    p.cuSeqQType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of seqused_q only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_seqused_q_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.valuesShape = {2, 39, 1, 2048};
    p.sequsedQ = {2};
    p.sequsedQType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of cmp_residual_k only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_cmp_residual_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.cmpRatio = 4;
    p.cmpResidual = {2};
    p.cmpResidualType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of output_idx_offset only supports int32 (seqused_q provided so its desc is valid for error logging)
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_idx_offset_dtype_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.idxOffset = {2, 39, 1};
    p.idxOffsetType = ge::DT_INT64;
    p.sequsedQ = {2};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of block_table only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_block_table_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.blockTableType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// dtype of seqused_k only supports int32
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_seqused_k_dtype_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.sequsedKType = ge::DT_INT64;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// last dim of sparse_values must be same as topk when return_value=1
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_values_topk_mismatch_failed)
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.valuesShape = {2, 39, 1, 1024};
    p.returnValue = 1;
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
}

// head dim of q only supports 128
TEST_F(QuantLightningIndexerV2TilingArch35, QuantLightningIndexerV2_950_tiling_head_dim_failed)
{
    qliv2_ut::CaseParam p = Make950TndPaFp8();
    p.qShape = {78, 64, 127};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    qliv2_ut::RunTilingCase(p, ge::GRAPH_FAILED);
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
constexpr int64_t TEST_BLOCK_COUNT = 2;
constexpr int64_t TEST_BLOCK_SIZE = 16;
constexpr int64_t TEST_BLOCKS_PER_BATCH = 1;
constexpr int64_t TEST_SINGLETON_DIM = 1;
constexpr int64_t TEST_Q_TOKENS = TEST_BATCH_SIZE * TEST_Q_SEQ_LEN;
constexpr int64_t TEST_K_TOKENS = TEST_BATCH_SIZE * TEST_K_SEQ_LEN;
constexpr int64_t TEST_CU_SEQLENS_SIZE = TEST_BATCH_SIZE + 1;
constexpr int64_t TEST_QUANT_FP8 = 1;
constexpr int64_t TEST_QUANT_INT8 = 2;
constexpr int64_t TEST_QUANT_MXFP8 = 3;
constexpr int64_t TEST_QUANT_HIFLOAT8 = 4;
constexpr int64_t TEST_QUANT_MXFP4 = 5;
constexpr int64_t TEST_MX_SCALE_BLOCK = 64;
constexpr int64_t TEST_MX_SCALE_PACK = 2;

struct ShapeCase {
    const char *name;
    const char *layout;
    std::vector<int64_t> qliv2_ut::CaseParam::*shapeMember;
    std::vector<int64_t> shape;
    ge::graphStatus expected = ge::GRAPH_FAILED;
};

struct ScaleCase {
    std::string layout;
    int64_t mode;
    std::string variant;
};

qliv2_ut::CaseParam Make950ShapeCase(const std::string &layout, int64_t mode)
{
    const bool tnd = layout == "TND" || layout == "TND_PA";
    const bool paged = layout == "PA_BBND" || layout == "TND_PA";
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = TEST_CORE_NUM;
    p.qShape = tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_Q_HEAD_NUM, TEST_HEAD_DIM} :
                     std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM};
    p.kShape = {TEST_BATCH_SIZE, TEST_K_SEQ_LEN, TEST_K_HEAD_NUM, TEST_HEAD_DIM};
    if (paged) {
        p.kShape = {TEST_BLOCK_COUNT, TEST_BLOCK_SIZE, TEST_K_HEAD_NUM, TEST_HEAD_DIM};
    } else if (tnd) {
        p.kShape = {TEST_K_SEQ_LEN, TEST_K_HEAD_NUM, TEST_HEAD_DIM};
    }
    p.wShape = tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_Q_HEAD_NUM} :
                     std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM};
    p.outShape = tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_K_HEAD_NUM, TEST_TOPK} :
                       std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM, TEST_TOPK};
    p.qScaleShape = p.qShape;
    p.kScaleShape = p.kShape;
    p.qScaleShape.pop_back();
    p.kScaleShape.pop_back();
    p.qType = ge::DT_FLOAT8_E4M3FN;
    if (mode == TEST_QUANT_INT8) {
        p.qType = ge::DT_INT8;
    } else if (mode == TEST_QUANT_HIFLOAT8) {
        p.qType = ge::DT_HIFLOAT8;
    } else if (mode == TEST_QUANT_MXFP4) {
        p.qType = ge::DT_FLOAT4_E2M1;
    }
    p.kType = p.qType;
    p.wType = mode == TEST_QUANT_INT8 ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    p.qScaleType = mode == TEST_QUANT_INT8 ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    if (mode == TEST_QUANT_MXFP8 || mode == TEST_QUANT_MXFP4) {
        p.qScaleShape.insert(p.qScaleShape.end(), {TEST_HEAD_DIM / TEST_MX_SCALE_BLOCK, TEST_MX_SCALE_PACK});
        p.kScaleShape.insert(p.kScaleShape.end(), {TEST_HEAD_DIM / TEST_MX_SCALE_BLOCK, TEST_MX_SCALE_PACK});
        p.qScaleType = ge::DT_FLOAT8_E8M0;
    } else if (mode == TEST_QUANT_HIFLOAT8) {
        p.qScaleShape = {TEST_SINGLETON_DIM};
        p.kScaleShape = {TEST_SINGLETON_DIM};
    }
    p.kScaleType = p.qScaleType;
    p.cuSeqQ = tnd ? std::vector<int64_t>{TEST_CU_SEQLENS_SIZE} : std::vector<int64_t>{};
    p.cuSeqK = tnd && !paged ? std::vector<int64_t>{TEST_CU_SEQLENS_SIZE} : std::vector<int64_t>{};
    p.sequsedQ = {TEST_BATCH_SIZE};
    p.sequsedK = {TEST_BATCH_SIZE};
    p.cmpResidual = {TEST_BATCH_SIZE};
    p.blockTable = paged ? std::vector<int64_t>{TEST_BATCH_SIZE, TEST_BLOCKS_PER_BATCH} : std::vector<int64_t>{};
    p.idxOffset = tnd ? std::vector<int64_t>{TEST_Q_TOKENS, TEST_K_HEAD_NUM} :
                        std::vector<int64_t>{TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM};
    p.layoutQ = tnd ? "TND" : "BSND";
    p.layoutK = paged ? "PA_BBND" : layout;
    p.topk = TEST_TOPK;
    p.quantMode = mode;
    p.maxSeqlenQ = TEST_Q_SEQ_LEN;
    p.maskMode = TEST_MASK_MODE_COMPRESS;
    p.cmpRatio = TEST_CMP_RATIO;
    return p;
}

std::vector<ScaleCase> MakeScaleCases()
{
    std::vector<ScaleCase> cases;
    for (const auto *layout : {"BSND", "TND", "PA_BBND", "TND_PA"}) {
        for (const auto mode :
             {TEST_QUANT_FP8, TEST_QUANT_INT8, TEST_QUANT_MXFP8, TEST_QUANT_HIFLOAT8, TEST_QUANT_MXFP4}) {
            for (const auto *variant : {"q_rank", "k_rank", "q_axis", "k_axis", "valid"}) {
                cases.push_back({layout, mode, variant});
            }
        }
    }
    return cases;
}
} // namespace

class QuantLightningIndexerV2ShapeContract : public testing::TestWithParam<ShapeCase> {};

TEST_P(QuantLightningIndexerV2ShapeContract, RejectInvalidShape)
{
    const auto &test = GetParam();
    auto p = Make950ShapeCase(test.layout, TEST_QUANT_FP8);
    p.*test.shapeMember = test.shape;
    qliv2_ut::RunTilingCase(p, test.expected);
}

INSTANTIATE_TEST_SUITE_P(
    Ascend950, QuantLightningIndexerV2ShapeContract,
    testing::Values(
        ShapeCase{
            "BSND_seqused_q_rank2", "BSND", &qliv2_ut::CaseParam::sequsedQ, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"BSND_seqused_q_short", "BSND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"BSND_seqused_q_long", "BSND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE + 1}},
        ShapeCase{
            "BSND_seqused_k_rank2", "BSND", &qliv2_ut::CaseParam::sequsedK, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"BSND_seqused_k_short", "BSND", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"BSND_seqused_k_long", "BSND", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"BSND_cmp_residual_k_rank2",
                  "BSND",
                  &qliv2_ut::CaseParam::cmpResidual,
                  {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"BSND_cmp_residual_k_short", "BSND", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"BSND_cmp_residual_k_long", "BSND", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"BSND_q_rank", "BSND", &qliv2_ut::CaseParam::qShape, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"BSND_k_rank", "BSND", &qliv2_ut::CaseParam::kShape, {TEST_K_TOKENS, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"BSND_w_rank", "BSND", &qliv2_ut::CaseParam::wShape, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM}},
        ShapeCase{"BSND_w_batch",
                  "BSND",
                  &qliv2_ut::CaseParam::wShape,
                  {TEST_BATCH_SIZE - 1, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM}},
        ShapeCase{
            "BSND_output_idx_offset_rank", "BSND", &qliv2_ut::CaseParam::idxOffset, {TEST_Q_TOKENS, TEST_K_HEAD_NUM}},
        ShapeCase{"BSND_valid", "BSND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"BSND_optional_q_absent", "BSND", &qliv2_ut::CaseParam::sequsedQ, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"BSND_optional_k_absent", "BSND", &qliv2_ut::CaseParam::sequsedK, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_seqused_q_rank2", "TND", &qliv2_ut::CaseParam::sequsedQ, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_seqused_q_short", "TND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_seqused_q_long", "TND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_seqused_k_rank2", "TND", &qliv2_ut::CaseParam::sequsedK, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_seqused_k_short", "TND", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_seqused_k_long", "TND", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_cmp_residual_k_rank2",
                  "TND",
                  &qliv2_ut::CaseParam::cmpResidual,
                  {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_cmp_residual_k_short", "TND", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_cmp_residual_k_long", "TND", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_q_rank",
                  "TND",
                  &qliv2_ut::CaseParam::qShape,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"TND_k_rank",
                  "TND",
                  &qliv2_ut::CaseParam::kShape,
                  {TEST_BATCH_SIZE, TEST_K_SEQ_LEN / TEST_BATCH_SIZE, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{
            "TND_w_rank", "TND", &qliv2_ut::CaseParam::wShape, {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM}},
        ShapeCase{"TND_output_idx_offset_rank",
                  "TND",
                  &qliv2_ut::CaseParam::idxOffset,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM}},
        ShapeCase{"TND_valid", "TND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_optional_q_absent", "TND", &qliv2_ut::CaseParam::sequsedQ, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_optional_k_absent", "TND", &qliv2_ut::CaseParam::sequsedK, {}, ge::GRAPH_SUCCESS},
        ShapeCase{"PA_BBND_seqused_q_rank2",
                  "PA_BBND",
                  &qliv2_ut::CaseParam::sequsedQ,
                  {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"PA_BBND_seqused_q_short", "PA_BBND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"PA_BBND_seqused_q_long", "PA_BBND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"PA_BBND_seqused_k_rank2",
                  "PA_BBND",
                  &qliv2_ut::CaseParam::sequsedK,
                  {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"PA_BBND_seqused_k_short", "PA_BBND", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"PA_BBND_seqused_k_long", "PA_BBND", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"PA_BBND_cmp_residual_k_rank2",
                  "PA_BBND",
                  &qliv2_ut::CaseParam::cmpResidual,
                  {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"PA_BBND_cmp_residual_k_short", "PA_BBND", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"PA_BBND_cmp_residual_k_long", "PA_BBND", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE + 1}},
        ShapeCase{
            "PA_BBND_q_rank", "PA_BBND", &qliv2_ut::CaseParam::qShape, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{
            "PA_BBND_k_rank", "PA_BBND", &qliv2_ut::CaseParam::kShape, {TEST_K_TOKENS, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"PA_BBND_w_rank", "PA_BBND", &qliv2_ut::CaseParam::wShape, {TEST_Q_TOKENS, TEST_Q_HEAD_NUM}},
        ShapeCase{"PA_BBND_output_idx_offset_rank",
                  "PA_BBND",
                  &qliv2_ut::CaseParam::idxOffset,
                  {TEST_Q_TOKENS, TEST_K_HEAD_NUM}},
        ShapeCase{"PA_BBND_valid", "PA_BBND", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"PA_BBND_optional_q_absent", "PA_BBND", &qliv2_ut::CaseParam::sequsedQ, {}, ge::GRAPH_SUCCESS},
        ShapeCase{
            "TND_PA_seqused_q_rank2", "TND_PA", &qliv2_ut::CaseParam::sequsedQ, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_PA_seqused_q_short", "TND_PA", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_PA_seqused_q_long", "TND_PA", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE + 1}},
        ShapeCase{
            "TND_PA_seqused_k_rank2", "TND_PA", &qliv2_ut::CaseParam::sequsedK, {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_PA_seqused_k_short", "TND_PA", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_PA_seqused_k_long", "TND_PA", &qliv2_ut::CaseParam::sequsedK, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_PA_cmp_residual_k_rank2",
                  "TND_PA",
                  &qliv2_ut::CaseParam::cmpResidual,
                  {TEST_SINGLETON_DIM, TEST_BATCH_SIZE}},
        ShapeCase{"TND_PA_cmp_residual_k_short", "TND_PA", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE - 1}},
        ShapeCase{"TND_PA_cmp_residual_k_long", "TND_PA", &qliv2_ut::CaseParam::cmpResidual, {TEST_BATCH_SIZE + 1}},
        ShapeCase{"TND_PA_q_rank",
                  "TND_PA",
                  &qliv2_ut::CaseParam::qShape,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{
            "TND_PA_k_rank", "TND_PA", &qliv2_ut::CaseParam::kShape, {TEST_K_TOKENS, TEST_K_HEAD_NUM, TEST_HEAD_DIM}},
        ShapeCase{"TND_PA_w_rank",
                  "TND_PA",
                  &qliv2_ut::CaseParam::wShape,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_Q_HEAD_NUM}},
        ShapeCase{"TND_PA_output_idx_offset_rank",
                  "TND_PA",
                  &qliv2_ut::CaseParam::idxOffset,
                  {TEST_BATCH_SIZE, TEST_Q_SEQ_LEN, TEST_K_HEAD_NUM}},
        ShapeCase{"TND_PA_valid", "TND_PA", &qliv2_ut::CaseParam::sequsedQ, {TEST_BATCH_SIZE}, ge::GRAPH_SUCCESS},
        ShapeCase{"TND_PA_optional_q_absent", "TND_PA", &qliv2_ut::CaseParam::sequsedQ, {}, ge::GRAPH_SUCCESS},
        ShapeCase{
            "TND_cu_seqlens_q_rank2", "TND", &qliv2_ut::CaseParam::cuSeqQ, {TEST_SINGLETON_DIM, TEST_CU_SEQLENS_SIZE}},
        ShapeCase{
            "TND_cu_seqlens_k_rank2", "TND", &qliv2_ut::CaseParam::cuSeqK, {TEST_SINGLETON_DIM, TEST_CU_SEQLENS_SIZE}},
        ShapeCase{"BSND_cu_seqlens_k_forbidden", "BSND", &qliv2_ut::CaseParam::cuSeqK, {TEST_CU_SEQLENS_SIZE}}),
    [](const testing::TestParamInfo<ShapeCase> &info) { return info.param.name; });

class QuantLightningIndexerV2ScaleContract : public testing::TestWithParam<ScaleCase> {};

TEST_P(QuantLightningIndexerV2ScaleContract, CheckScaleShape)
{
    const auto &test = GetParam();
    auto p = Make950ShapeCase(test.layout, test.mode);
    if (test.variant != "valid") {
        auto &shape = test.variant.front() == 'q' ? p.qScaleShape : p.kScaleShape;
        if (test.variant == "q_rank" || test.variant == "k_rank") {
            shape.push_back(TEST_SINGLETON_DIM);
        } else {
            shape.front() += 1;
        }
    }
    qliv2_ut::RunTilingCase(p, test.variant == "valid" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(Ascend950, QuantLightningIndexerV2ScaleContract, testing::ValuesIn(MakeScaleCases()),
                         [](const testing::TestParamInfo<ScaleCase> &info) {
                             return info.param.layout + "_mode" + std::to_string(info.param.mode) + "_" +
                                    info.param.variant;
                         });
