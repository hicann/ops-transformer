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
 * \file ts_grouped_matmul_kernel.cpp
 * \brief GroupedMatmul kernel用例.
 */

#include "ts_grouped_matmul.h"
#include "ts_grouped_matmul_utils.h"

using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910B2;
using gmmTestParam::Ts_GroupedMatmul_WithParam_Ascend910B3;
using namespace gmmTestUtils;

namespace {
TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_Kernel_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Kernel_Case = ::testing::Values(
    GroupedMatmulCase( /* no split, m-m-m */
        "GroupedMatmul_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{256, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{256}, {1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {1024, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    //x:4096*4096; m:1,4096,4096
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_FLOAT16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                           ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096,4096}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,64,256,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,4096}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,4096}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{1,1,4096}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{4096,4096}}, ge::DataType::DT_FLOAT16)},
                GenTensor("per_token_scale", {4096,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {4096}, 3, -1, false, false, 0, 1, 0, {16,1}),
        0),
    //x:4096*4096; m:1,4096,4096
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_BF16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                           ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{4096,4096}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,64,256,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,4096}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,4096}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{1,1,4096}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{4096,4096}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {4096,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {4096}, 3, -1, false, false, 0, 1, 0, {16,1}),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_BF16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                           ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,1,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,64}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,64}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0, {16,1}),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_FLOAT16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,1,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,64}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,64}}, ge::DataType::DT_FLOAT16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0, {16,1}),
        0),
    GroupedMatmulCase( /* a8w4 asym quant int8*/
        "GroupedMatmul_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
                ExpectInfo(false,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_FLOAT16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                           ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,255}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,255,64}}, ge::DataType::DT_INT4),
                GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,64}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{1,1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,64}}, ge::DataType::DT_FLOAT16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a4w4 pergroup*/
            "GroupedMatmul_Case6", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, true),
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
                                ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
            Param({GenTensorList("x", {{256,256}}, ge::DataType::DT_INT4),
                    GenTensorList("weight", {{4,32,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                    GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
                    GenTensorList("scale", {{4,16,2048}}, ge::DataType::DT_UINT64),
                    GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                    GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                    GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                    GenTensorList("y", {{256,2048}}, ge::DataType::DT_FLOAT16)},
                    GenTensor("per_token_scale", {256,}, ge::DataType::DT_FLOAT),
                    GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
                    {64,64,64,64}, 3, -1, false, false, 0, 1, 0),
            0),
    GroupedMatmulCase( /* a8w8 asym quant int8*/
        "GroupedMatmul_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_FLOAT16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,256,64}}, ge::DataType::DT_INT4),
                GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,64}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{1,1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,64}}, ge::DataType::DT_FLOAT16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true), // 1,16,256,512
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_FLOAT16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,8,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,512}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{1,1,512}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{1,1,512}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{16,512}}, ge::DataType::DT_FLOAT16)},
                GenTensor("per_token_scale", {16,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {16}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true), // 1,16,256,512
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_BF16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,8,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,512}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{4,1,512}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{16,512}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {16,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {16}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
                ExpectInfo(false,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_BF16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,256,256}}, ge::DataType::DT_INT4),
                GenTensorList("bias", {{1,256}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{4,3,256}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{1}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{1}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,256}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
            "GroupedMatmul_Case11", true, "", /* CaseName, Enable, DebugInfo */
            OpInfo(ControlInfo(true, false),
                    ExpectInfo(false,
                               GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                                  GMM_TPL_INT4, // D_T_B
                                                  GMM_TPL_BF16, // D_T_Y
                                                  0, // TRANS_A
                                                  0, // TRANS_B
                                                  GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                                  0, // IS_STATIC_TILING_API
                                                  GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                                  GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                                  GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                                ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
            Param({GenTensorList("x", {{64,32768}}, ge::DataType::DT_INT8),
                    GenTensorList("weight", {{1,32768,256}}, ge::DataType::DT_INT4),
                    GenTensorList("bias", {{1,256}}, ge::DataType::DT_FLOAT),
                    GenTensorList("scale", {{4,3,256}}, ge::DataType::DT_UINT64),
                    GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                    GenTensorList("antiquant_scale", {{1}}, ge::DataType::DT_FLOAT16),
                    GenTensorList("antiquant_offset", {{1}}, ge::DataType::DT_FLOAT16),
                    GenTensorList("y", {{64,256}}, ge::DataType::DT_BF16)},
                    GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                    GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                    {64}, 3, -1, false, false, 0, 1, 0),
            0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_BF16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,256,256}}, ge::DataType::DT_INT4),
                GenTensorList("bias", {{1,256}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{4,2,256}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{1}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{1}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,256}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case13", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                              GMM_TPL_INT4, // D_T_B
                                              GMM_TPL_BF16, // D_T_Y
                                              0, // TRANS_A
                                              0, // TRANS_B
                                              GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                              0, // IS_STATIC_TILING_API
                                              GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT, // A8W4_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                              GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                             ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64,256}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1,1,16,16,64}}, ge::DataType::DT_INT4, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{1,64}}, ge::DataType::DT_FLOAT),
                GenTensorList("scale", {{4,1,64}}, ge::DataType::DT_UINT64),
                GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{2}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64,64}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {64,1}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {1}, ge::DataType::DT_INT64),
                {64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* split M, m-m-m */
        "GroupedMatmul_Case14", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{128, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{256}, {1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{128, 256}, {1024, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* split M, s-m-s */
        "GroupedMatmul_Case15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{256}, {256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* split M, s-m-m */
        "GroupedMatmul_Case16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("bias", {{256}, {256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{128, 256}, {128, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {128, 128}, 0, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase(  /* split K s-s-s + transpose x */
        "GroupedMatmul_Case17", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{256, 768}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{768, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {256, 512}, 3, -1, false, true, 2, 1, 0),
        0),
    GroupedMatmulCase(  /* split K m-s-m + transpose x */
        "GroupedMatmul_Case18", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{256, 256}, {256, 512}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{768, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, true, 2, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + transpose weight */
        "GroupedMatmul_Case19", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 100729089,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 256, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {1024, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* single tensor + w nz */
        "GroupedMatmul_Case20", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{64, 64}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 4, 4, 16, 16}}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{32, 64}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {32, 32}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant */
        "GroupedMatmul_Case21", true, "", /* CaseName, Enable, DebugInfo */
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
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant */
        "GroupedMatmul_Case22", true, "", /* CaseName, Enable, DebugInfo */
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
               GenTensorList("weight", {{16, 1024, 8192}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 8192}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 8192}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant */
        "GroupedMatmul_Case23", true, "", /* CaseName, Enable, DebugInfo */
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
               GenTensorList("weight", {{16, 8192, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{16, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{360, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {360}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {16}, ge::DataType::DT_INT64),
              {40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20}, 2, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant act */
        "GroupedMatmul_Case24", true, "", /* CaseName, Enable, DebugInfo */
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
        Param({GenTensorList("x", {{16, 10}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 10, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 1, false, false, 0, 1, 1),
        0),
    GroupedMatmulCase( /* quant act */
        "GroupedMatmul_Case25", true, "", /* CaseName, Enable, DebugInfo */
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
        Param({GenTensorList("x", {{16, 20}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 20, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 1, false, false, 0, 1, 1),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case26", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
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
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 0, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* per token quant weight NZ */
        "GroupedMatmul_Case27", true, "", /* CaseName, Enable, DebugInfo */
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
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 8, 8, 16, 32}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 256}}, ge::DataType::DT_BF16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 256}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, 1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case28", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
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
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{32, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 1),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case29", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 20}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 20}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{32, 20}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 2),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case30", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
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
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 30}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 30}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{32, 30}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 4),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case31", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
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
        Param({GenTensorList("x", {{32, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 40}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 40}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{32, 40}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {32}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {16, 16}, 3, 0, false, false, 0, 1, 5),
        0),
    GroupedMatmulCase( /* pertoken quant fp16*/
        "GroupedMatmul_Case32", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 10, 5}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, 0, true, false, 0, 1, 2),
        0),
    GroupedMatmulCase( /* quant + transpose weight*/
        "GroupedMatmul_Case33", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 64}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{4, 64, 32}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 32}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{4, 32}}, ge::DataType::DT_FLOAT),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 32}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, 1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant 有问题*/
        "GroupedMatmul_Case34", false, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_INT8, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             1, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}, {1024, 256}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{256, 256}, {256, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{256}, {1024}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{256}, {1024}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{256}, {1024}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {1024, 1024}}, ge::DataType::DT_INT8)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 fp16 */
        "GroupedMatmul_Case35", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}, {128, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {256, 128}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {128, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 bf16 */
        "GroupedMatmul_Case36", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_BF16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}, {64, 256}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{256, 256}, {256, 64}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{256}, {64}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{256}, {64}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{256}, {64}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 256}, {64, 64}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 bf16 */
        "GroupedMatmul_Case37", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_BF16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{128, 256}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{256, 256}, {256, 256}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{2, 256}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 256}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{2, 256}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{128, 256}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w4 fp16 */
        "GroupedMatmul_Case38", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT4, // D_T_B
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
        Param({GenTensorList("x", {{256, 256},}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 256, 256}}, ge::DataType::DT_INT4),
               GenTensorList("bias", {{4, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{4, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w4 fp16 transpose weight*/
        "GroupedMatmul_Case39", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT4, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256},}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{8, 256, 256}}, ge::DataType::DT_INT4),
               GenTensorList("bias", {{8, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{8, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{8, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {32, 32, 32, 32, 32, 32, 32, 32}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w4 bf16 */
        "GroupedMatmul_Case40", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_BF16, // D_T_A
                                             GMM_TPL_INT4, // D_T_B
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
        Param({GenTensorList("x", {{256, 256},}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{4, 256, 256}}, ge::DataType::DT_INT4),
               GenTensorList("bias", {{4, 256}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 256}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{4, 256}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{256, 256}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {64, 64, 64, 64}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a16w8 fp16 + transpose weight */
        "GroupedMatmul_Case41", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 256}, {128, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{256, 256}, {128, 256}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{256}, {128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{256, 256}, {128, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {}, ge::DataType::DT_INT64),
              {}, 0, -1, true, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant performance */
        "GroupedMatmul_Case42", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{1024, 2048}, {1024, 2048}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2048, 1000}, {2048, 1000}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{1000}, {1000}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{1000}, {1000}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{1000}, {1000}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1024, 1000}, {1024, 1000}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {256, 256, 256, 256}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd */
        "GroupedMatmul_Case43", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{4, 128, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{4, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd + transpsoe weight */
        "GroupedMatmul_Case44", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 1024, 128}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{2, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{2, 1024}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd bf16 */
        "GroupedMatmul_Case45", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_BF16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{4, 128, 1024}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{4, 1024}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{4, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {4, 4, 4, 4}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant msd bf16 + transpsoe weight */
        "GroupedMatmul_Case46", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_BF16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 128}}, ge::DataType::DT_BF16),
               GenTensorList("weight", {{2, 1024, 128}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{2, 1024}}, ge::DataType::DT_FLOAT),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("antiquant_offset", {{2, 1024}}, ge::DataType::DT_BF16),
               GenTensorList("y", {{16, 1024}}, ge::DataType::DT_BF16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, true, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* antiquant pergroup */
        "GroupedMatmul_Case47", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{16, 16}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{2, 16, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{2, 10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("scale", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("offset", {{}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_scale", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{2, 2, 10}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 0, -1, false, false, -1, 1, 0),
        0),
    GroupedMatmulCase(  /* split K s-s-s + transpose x */
        "GroupedMatmul_Case48", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
                          GMMEncodeTilingKey(GMM_TPL_FLOAT16, // D_T_A
                                             GMM_TPL_FLOAT16, // D_T_B
                                             GMM_TPL_FLOAT16, // D_T_Y
                                             1, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{256, 768}}, ge::DataType::DT_FLOAT16),
               GenTensorList("weight", {{768, 256}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{1024, 256}}, ge::DataType::DT_FLOAT16)},
              GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {4}, ge::DataType::DT_INT64),
              {256, 512, 512, 768}, 3, -1, false, true, 2, 0, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case49", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
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
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{5, 10}, {5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_INT8)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* quant int8*/
        "GroupedMatmul_Case50", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true,
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
        Param({GenTensorList("x", {{16, 5}}, ge::DataType::DT_INT8),
               GenTensorList("weight", {{2, 5, 10}}, ge::DataType::DT_INT8),
               GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
               GenTensorList("scale", {{2, 10}}, ge::DataType::DT_UINT64),
               GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
               GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{16, 10}}, ge::DataType::DT_INT8)},
              GenTensor("per_token_scale", {16}, ge::DataType::DT_FLOAT),
              GenTensor("grouped_list", {2}, ge::DataType::DT_INT64),
              {8, 8}, 3, -1, false, false, 0, 1, 0),
        0),
    GroupedMatmulCase( /* a8w8 sparse */
        "GroupedMatmul_Case51", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64, 512}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1, 512, 3072}}, ge::DataType::DT_INT8),
                GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{1, 3072}}, ge::DataType::DT_BF16),
                GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64, 3072}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {0, 1}, ge::DataType::DT_INT64),
                {0, 16}, 3, 1, false, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 sparse */
        "GroupedMatmul_Case52", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64, 512}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1, 3072, 512}}, ge::DataType::DT_INT8),
                GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{1, 3072}}, ge::DataType::DT_BF16),
                GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64, 3072}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {0, 1}, ge::DataType::DT_INT64),
                {0, 16}, 3, 1, true, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 sparse */
        "GroupedMatmul_Case53", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             0, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64, 512}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1, 512, 3072}}, ge::DataType::DT_INT8),
                GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{1, 3072}}, ge::DataType::DT_BF16),
                GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64, 3072}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {64}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {0, 1}, ge::DataType::DT_INT64),
                {0, 16}, 3, 1, false, false, 0, 2, 0),
        0),
    GroupedMatmulCase( /* a8w8 sparse */
        "GroupedMatmul_Case54", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_1 // AIV_AIC_RATIO
                                            ),
                            ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{64, 512}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1, 3072, 512}}, ge::DataType::DT_INT8),
                GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{1, 3072}}, ge::DataType::DT_BF16),
                GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{64, 3072}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {64}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {0, 1}, ge::DataType::DT_INT64),
                {0, 16}, 3, 1, true, false, 0, 2, 0),
        0),
            GroupedMatmulCase( /* a8w8 dynamic tiling */
        "GroupedMatmul_Case55", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
                ExpectInfo(true,
                           GMMEncodeTilingKey(GMM_TPL_INT8, // D_T_A
                                             GMM_TPL_INT8, // D_T_B
                                             GMM_TPL_BF16, // D_T_Y
                                             0, // TRANS_A
                                             1, // TRANS_B
                                             GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, // GROUP_LIST_TYPE
                                             0, // IS_STATIC_TILING_API
                                             GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE, // A8W4_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
                                             GROUPED_MATMUL_AIV_AIC_RATIO_2 // AIV_AIC_RATIO
                                            ),
                            12)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x", {{128, 512}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{1, 3072, 512}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{1, 3072}}, ge::DataType::DT_BF16),
                GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{128, 3072}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {128}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {128}, ge::DataType::DT_INT64),
                {64}, 3, 1, true, false, 0, 1, 0, {128}),
        0),
            GroupedMatmulCase( /* a8w8 dynamic tiling */
        "GroupedMatmul_Case56", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
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
        Param({GenTensorList("x", {{320, 4096}}, ge::DataType::DT_INT8),
                GenTensorList("weight", {{10, 4096, 512}}, ge::DataType::DT_INT8, ge::FORMAT_FRACTAL_NZ),
                GenTensorList("bias", {{0}}, ge::DataType::DT_INT32),
                GenTensorList("scale", {{10, 512}}, ge::DataType::DT_BF16),
                GenTensorList("offset", {{0}}, ge::DataType::DT_FLOAT),
                GenTensorList("antiquant_scale", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("antiquant_offset", {{0}}, ge::DataType::DT_FLOAT16),
                GenTensorList("y", {{320, 512}}, ge::DataType::DT_BF16)},
                GenTensor("per_token_scale", {320}, ge::DataType::DT_FLOAT),
                GenTensor("grouped_list", {10}, ge::DataType::DT_INT64),
                {32,32,32,32,32,32,32,32,32,32}, 3, 1, false, false, 0, 1, 0,{32}),
        0)
    );

INSTANTIATE_TEST_SUITE_P(GroupedMatmul, Ts_GroupedMatmul_WithParam_Ascend910B3, Tc_GroupedMatmul_Kernel_Case);

TEST_P(Ts_GroupedMatmul_WithParam_Ascend910B2, Tc_Kernel_910b2_GroupedMatmul)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_GroupedMatmul_Kernel910B2_Case = ::testing::Values(
    GroupedMatmulCase( /* a8w8 preTiling */
        "GroupedMatmul_Pretiling_Case_0", true, "", /* CaseName, Enable, DebugInfo */
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
        "GroupedMatmul_Pretiling_Case_1", true, "", /* CaseName, Enable, DebugInfo */
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
        "GroupedMatmul_Pretiling_Case_2", true, "", /* CaseName, Enable, DebugInfo */
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
        OpInfo(ControlInfo(true, true),
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
        OpInfo(ControlInfo(true, true),
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
        OpInfo(ControlInfo(true, true),
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
        OpInfo(ControlInfo(true, true),
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
        OpInfo(ControlInfo(true, true),
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
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_FLOAT),
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
        OpInfo(ControlInfo(true, true),
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
        OpInfo(ControlInfo(true, true),
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
        OpInfo(ControlInfo(true, true),
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
               GenTensorList("scale", {{2, 4096}}, ge::DataType::DT_FLOAT),
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
} // namespace