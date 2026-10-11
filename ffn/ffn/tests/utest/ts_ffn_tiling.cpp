/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ts_ffn_tiling.cpp
 * \brief FFN tiling用例.
 */

#include "ts_ffn.h"

namespace {
// TEST_P(Ts_FFN_WithParam_Ascend310P3, Tc_Tiling310P_FFN)
// {
//     ASSERT_TRUE(case_->Init());
//     ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
// }

// TEST_P(Ts_FFN_WithParam_Ascend910B3, Tc_Tiling_FFN)
// {
//     ASSERT_TRUE(case_->Init());
//     ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
// }

const auto Tc_FFN_Tiling310P_Case = ::testing::Values(FFNCase(
    "FFN_Moe_Case0", true, "",                                /* CaseName, Enable, DebugInfo */
    OpInfo(ControlInfo(true, false), ExpectInfo(true, 0, 7)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
    Param({GenTensor("x", {40, 128, 16, 16}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
           GenTensor("weight1", {320, 40, 16, 16}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
           GenTensor("weight2", {40, 320, 16, 16}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ),
           GenTensor("bias1", {5120}, ge::DataType::DT_FLOAT16), GenTensor("bias2", {640}, ge::DataType::DT_FLOAT16),
           GenTensor("y", {40, 128, 16, 16}, ge::DataType::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ)},
          {}, "gelu", 1, -1),
    0));

const auto Tc_FFN_Tiling_Case = ::testing::Values(
    FFNCase(
        "FFN_Moe_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1024, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 20480}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 20480, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 20480}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1024, 5120}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, "gelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1024, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 0, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1024, 5120}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, "gelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1024, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 0, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1024, 5120}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1024, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 0, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1024, 5120}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, "relu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1024, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 0, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 0}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1024, 5120}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, "silu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {3849, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {3849, 5120}, ge::DataType::DT_FLOAT16)},
              {213, 239, 201, 151, 171, 278, 222, 329, 131, 194, 165, 267, 425, 146, 379, 338}, "gelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {340, 5120}, ge::DataType::DT_INT8),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_INT8),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_INT32),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_INT32),
               GenTensor("scale", {16}, ge::DataType::DT_FLOAT16), GenTensor("offset", {16}, ge::DataType::DT_FLOAT16),
               GenTensor("deqScale1", {16, 2560}, ge::DataType::DT_INT64),
               GenTensor("deqScale2", {16, 5120}, ge::DataType::DT_INT64),
               GenTensor("y", {340, 5120}, ge::DataType::DT_FLOAT16)},
              {14, 66, 2, 29, 28, 32, 8, 8, 16, 24, 25, 23, 11, 21, 13, 20}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1, 32, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 256, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 256, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1, 32, 256}, ge::DataType::DT_FLOAT16)},
              {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case8", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 3,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {1024, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 20480}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {16, 20480, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 20480}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {1024, 5120}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, "gelu", 0, -1),
        0),
    FFNCase(
        "FFN_Moe_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 3,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {16, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias1", {4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {16, 512}, ge::DataType::DT_FLOAT16)},
              {4, 4, 4, 4}, "gelu", 0, -1),
        0),
    FFNCase(
        "FFN_Moe_Case10", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 6,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {340, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_INT8),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale1", {16, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset1", {16, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {340, 5120}, ge::DataType::DT_FLOAT16)},
              {14, 66, 2, 29, 28, 32, 8, 8, 16, 24, 25, 23, 11, 21, 13, 20}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 6,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {340, 5120}, ge::DataType::DT_BF16),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_INT8),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_FLOAT),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT),
               GenTensor("antiquant_scale1", {16, 2560}, ge::DataType::DT_BF16),
               GenTensor("antiquant_scale2", {16, 5120}, ge::DataType::DT_BF16),
               GenTensor("antiquant_offset1", {16, 2560}, ge::DataType::DT_BF16),
               GenTensor("antiquant_offset2", {16, 5120}, ge::DataType::DT_BF16),
               GenTensor("y", {340, 5120}, ge::DataType::DT_BF16)},
              {14, 66, 2, 29, 28, 32, 8, 8, 16, 24, 25, 23, 11, 21, 13, 20}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 6,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {340, 5120}, ge::DataType::DT_BF16),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_INT4),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_INT4),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_FLOAT),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT),
               GenTensor("antiquant_scale1", {16, 2560}, ge::DataType::DT_BF16),
               GenTensor("antiquant_scale2", {16, 5120}, ge::DataType::DT_BF16),
               GenTensor("antiquant_offset1", {16, 2560}, ge::DataType::DT_BF16),
               GenTensor("antiquant_offset2", {16, 5120}, ge::DataType::DT_BF16),
               GenTensor("y", {340, 5120}, ge::DataType::DT_BF16)},
              {14, 66, 2, 29, 28, 32, 8, 8, 16, 24, 25, 23, 11, 21, 13, 20}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case14", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 11,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {340, 5120}, ge::DataType::DT_INT8),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_INT8),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_INT32),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_INT32),
               GenTensor("scale", {16}, ge::DataType::DT_FLOAT16), GenTensor("offset", {16}, ge::DataType::DT_FLOAT16),
               GenTensor("deqScale1", {16, 2560}, ge::DataType::DT_BF16),
               GenTensor("deqScale2", {16, 5120}, ge::DataType::DT_BF16),
               GenTensor("y", {340, 5120}, ge::DataType::DT_BF16)},
              {14, 66, 2, 29, 28, 32, 8, 8, 16, 24, 25, 23, 11, 21, 13, 20}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 15,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {4, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {16, 5120, 2560}, ge::DataType::DT_INT8),
               GenTensor("weight2", {16, 2560, 5120}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {16}, ge::DataType::DT_INT64),
               GenTensor("bias1", {16, 2560}, ge::DataType::DT_FLOAT),
               GenTensor("bias2", {16, 5120}, ge::DataType::DT_FLOAT),
               GenTensor("antiquant_scale1", {16, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset1", {16, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset2", {16, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {4, 5120}, ge::DataType::DT_FLOAT16)},
              {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_glu_Case16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param(
            {GenTensor("x", {2048, 640}, ge::DataType::DT_FLOAT16),
             GenTensor("weight1", {640, 5120}, ge::DataType::DT_FLOAT16),
             GenTensor("weight2", {2560, 640}, ge::DataType::DT_FLOAT16),
             GenTensor("bias1", {5120}, ge::DataType::DT_FLOAT16), GenTensor("bias2", {640}, ge::DataType::DT_FLOAT16),
             GenTensor("y", {2048, 640}, ge::DataType::DT_FLOAT16)},
            {}, "geglu", 1, -1),
        0));

// INSTANTIATE_TEST_SUITE_P(FFN, Ts_FFN_WithParam_Ascend310P3, Tc_FFN_Tiling310P_Case);
// INSTANTIATE_TEST_SUITE_P(FFN, Ts_FFN_WithParam_Ascend910B3, Tc_FFN_Tiling_Case);
} // namespace
