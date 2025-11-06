/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ts_grouped_matmul_tiling.cpp
 * \brief GroupedMatmul tiling用例.
 */

#include "ts_grouped_matmul.h"
#include "ts_grouped_matmul_utils.h"

using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend310P3;
using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910B2;
using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910B3;
using namespace gmmTestUtils;
namespace {
TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B2, Tc_Kernel_910b2_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_Tiling_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

TEST_P(Ts_GroupedMatmul_WithParam_Ascend310P3, Tc_Tiling310_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Tiling_Case = ::testing::Values(
    GroupedMatmulCase(
        "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedMatmul_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 300}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{300, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 0, -1, false, false, 1, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedMatmul_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 400}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{400, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}, {256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase(
        "GroupedMatmul_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 300}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{300, 128}, {300, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 3, -1, false, false, 2, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor split m */
        "GroupedMatmul_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose w */
        "GroupedMatmul_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 128, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose x */
        "GroupedMatmul_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             1, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{512, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 512, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{4, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, true, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant */
        "GroupedMatmul_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 5}, {16, 5}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{5, 10}, {5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{10}, {10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}, {16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 0, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT8, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 1024, 8191}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8191}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 8191}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{360, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 256, 64, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8192}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 8192}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {20, 20, 40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{360, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{32, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{32, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {32}, ge::DataType::DT_INT64),
              {11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12,
               11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 11, 12, 11, 12, 11, 12}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {30, 30, 30, 30}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int32 weight NZ */
        "GroupedMatmul_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{0}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {120}, 2, 2, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case14", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {200}),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 32, 512, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {200}),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          20)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{9, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{16, 2048, 32768}}, ge::DataType::DT_INT8, ge::FORMAT_ND),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 328768}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{9, 32768}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {9}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {9}),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case17", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{120, 7680}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 7680,1024}}, ge::DataType::DT_INT8, ge::FORMAT_ND),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{120, 1024}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {120}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {120}, 2, 1, false, false, 0, 1, 0, {256}),
        0),
    GroupedMatmulCase(
            "GroupedMatmul_Case18", true, "",
            OpInfo(ControlInfo(true, false), \
              ExpectInfo(true,
                         GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                         20)),
            Param({GenTensorList("x", {{9, 7168}, {9, 7168}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{7168, 2048}, {7168, 2048}}, ge::DataType::DT_INT8),
                GenTensorList("bias", {{2048}, {2048}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{}}, ge::DataType::DT_INT32),
                GenTensorList("offset", {{}}, ge::DataType::DT_INT32),
                GenTensorList("antiquant_scale", {{2048}, {2048}}, ge::DataType::DT_INT32),
                GenTensorList("antiquant_offset", {{2048}, {2048}}, ge::DataType::DT_INT32),
                GenTensorList("y", {{9, 2048}, {9, 2048}}, ge::DataType::DT_INT32)},
            GenTensor("per_token_scale", {}, ge::DataType::DT_INT32),
            GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
            {}, 0, -1, false, false, -1, 0, 0, {9}),
            0),
    GroupedMatmulCase(
              "GroupedMatmul_Case19", true, "", /* CaseName, Enable, DebugInfo */
              OpInfo(ControlInfo(true, false),
                     ExpectInfo(true,
                                GMMEncodeTilingKey(GMM_TPL_INT4, // D_T_A
                                                   GMM_TPL_INT4, // D_T_B
                                                   GMM_TPL_FLOAT16, // D_T_Y
                                                   0, // TRANS_A
                                                   0, // TRANS_B
                                                   GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                                   0, // IS_STATIC_TILING_API
                                                   GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                                   GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                                   GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                            ),
                                20)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
              Param({GenTensorList("x", {{64, 256}}, ge::DataType::DT_INT4),
                     GenTensorList("weight", {{1, 256, 128}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                     GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("scale", {{1,1,128}}, ge::DataType::DT_UINT64),
                     GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                     GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("y", {{64, 128}}, ge::DataType::DT_FLOAT16)},
                     GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                     GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                    {64}, 3, -1, false, false, 0, 1, 0),
              0),
     GroupedMatmulCase(
              "GroupedMatmul_Case20", true, "", /* CaseName, Enable, DebugInfo */
              OpInfo(ControlInfo(true, false),
                     ExpectInfo(true,
                                GMMEncodeTilingKey(GMM_TPL_INT4, // D_T_A
                                                   GMM_TPL_INT4, // D_T_B
                                                   GMM_TPL_FLOAT16, // D_T_Y
                                                   0, // TRANS_A
                                                   0, // TRANS_B
                                                   GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                                   0, // IS_STATIC_TILING_API
                                                   GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                                   GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                                   GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                            ),
                                20)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
              Param({GenTensorList("x", {{64, 256}}, ge::DataType::DT_INT4),
                     GenTensorList("weight", {{1, 256, 128}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                     GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("scale", {{1, 128}}, ge::DataType::DT_UINT64),
                     GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                     GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                     GenTensorList("y", {{64, 128}}, ge::DataType::DT_FLOAT16)},
                     GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                     GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                     {64}, 3, -1, false, false, 0, 1, 0),
              0),
    GroupedMatmulCase(
        "GroupedMatmul_Case21", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{25600, 384}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 384, 2560}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{25600, 2560}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {6400, 6400, 6400, 6400}, 3, -1, false, false, 0, 1, 0),
        0)
);

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Tiling_Case);

const auto Tc_GroupedMatmul_Kernel910B2_Case = ::testing::Values(
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_tiling_Case_0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{576, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{576, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {576}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_tiling_Case_1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{228, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 7168, 4096}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{228, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {228}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_tiling_Case_2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{228, 2048}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{1, 2048, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{1, 7168}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{228, 7168}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
              {228}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {0, 540}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{1080, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 1080, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {540, 540}, 3, 1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 540, 1, 0}}, 3, 1, false, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT32, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_INT32)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 0, 1, 540}}, 3, 1, true, false, 0, 2, 0),
        0),
       GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {540}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {0, 540}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{1080, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1, 1080, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {1080}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {540, 540}, 3, 1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 7168, 4096}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 540, 1, 0}}, 3, 1, false, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 static tiling */
        "GroupedMatmul_static_tiling_Case_10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          24)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{540, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 4096, 7168}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{540, 4096}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {540}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {{0, 0, 1, 540}}, 3, 1, true, false, 0, 2, 0),
        0)
);

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B2, Tc_GroupedMatmul_Kernel910B2_Case);


const auto Tc_GroupedMatmul_Tiling310_Case = ::testing::Values(
       GroupedMatmulCase(
           "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
           OpInfo(ControlInfo(true, false),
                  ExpectInfo(true,
                             GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                                GMM_TPL_FLOAT16, // D_T_B
                                                GMM_TPL_FLOAT16, // D_T_Y
                                                0, // TRANS_A
                                                0, // TRANS_B
                                                GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                                0, // IS_STATIC_TILING_API
                                                GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                                GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                                GROUPED_MATMUL_CUBE_ONLY // AIV_AIC_RATIO
                                            ),
                             8)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
           Param({GenTensorList("x", {{256, 512}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("weight", {{512, 128}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("bias", {{128}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                  GenTensorList("y", {{256, 128}}, ge::DataType::DT_FLOAT16)},
                 GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
                 GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                 {64, 64, 64, 64}, 0, -1, false, false, -1, 1, 0),
           0));

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend310P3, Tc_GroupedMatmul_Tiling310_Case);

} // namespace