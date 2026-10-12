/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_masked_causal_conv1d_tiling.cpp
 * \brief Unit tests for MaskedCausalConv1d tiling logic
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/masked_causal_conv1d_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MaskedCausalConv1dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaskedCausalConv1dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaskedCausalConv1dTiling TearDown" << std::endl;
    }
};

// Test 1: BF16 baseline, S=2048 B=4 H=768
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_s2048_b4_h768)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 2048, B = 4, H = 768, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);
    int64_t expectTilingKey = 10000;
    std::string expectTilingData =
        "2048 4 768 8 4 128 64 4 0 1 1 2 0 1024 1024 64 1 480 2 64 1 1 3 64 1 64 1 1 3 64 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 4: H not divisible by H_REG=64 → CheckInputParams GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_h_not_aligned)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 100, W = 3; // H=100 not multiple of 64

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 5: Weight K != CONV_WINDOW_SIZE=3 → CheckInputParams GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_wrong_weight_k)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 4; // W=4 != 3

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 6: Unsupported dtype FP32 → GetInputDtypes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_invalid_dtype_fp32)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 8: Situation A + hUb expansion
// H=8192 → hCoreCnt_=64, hBlockFactor_=128; small BS fits at hUb=64,
// then the k-loop expands ubFactorH_ to 128.
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_bf16_situation_a_hexpand_s32_b1_h8192)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 32, B = 1, H = 8192, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    int64_t expectTilingKey = 10000;
    std::string expectTilingData =
        "32 1 8192 64 0 128 128 1 0 1 1 1 0 32 32 128 1 32 1 128 1 1 1 32 1 128 1 1 1 32 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 12: x is 2-D (not 3-D) → GetInputShapes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_x_not_3d)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, H}, {S, H}}, ge::DT_BF16, ge::FORMAT_ND}, // 2-D, should be 3-D
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, H}, {S, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 13: weight dtype != x dtype (x=BF16, weight=FP16) → GetInputDtypes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_weight_dtype_mismatch)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // FP16 != BF16
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 14: mask dtype != DT_BOOL (mask=FP16) → GetInputDtypes GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_mask_dtype_not_bool)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara(
        "MaskedCausalConv1d",
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{B, S}, {B, S}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // should be DT_BOOL
        },
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 15: weight is 1-D (not 2-D) → CheckInputParams GRAPH_FAILED
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_weight_not_2d)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{H}, {H}}, ge::DT_BF16, ge::FORMAT_ND}, // 1-D, should be 2-D [K, H]
                                                  {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 16:
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_weight_h_mismatch)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara(
        "MaskedCausalConv1d",
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{W, 256}, {W, 256}}, ge::DT_BF16, ge::FORMAT_ND}, // weight H=256 != x H=128
            {{{B, S}, {B, S}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// Test 17: mask shape != x shape (mask B or S mismatch) → CheckInputParams GRAPH_FAILED
// Uses mask B=3 while x B=2; the S-mismatch branch is symmetric and not separately tested.
TEST_F(MaskedCausalConv1dTiling, MaskedCausalConv1d_950_tiling_fail_mask_shape_mismatch)
{
    struct MaskedCausalConv1dCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {};

    int64_t S = 1024, B = 2, H = 128, W = 3;

    gert::TilingContextPara tilingContextPara("MaskedCausalConv1d",
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{W, H}, {W, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{3, S}, {3, S}}, ge::DT_BOOL, ge::FORMAT_ND}, // mask B=3 != x B=2
                                              },
                                              {
                                                  {{{S, B, H}, {S, B, H}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              attrs, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
