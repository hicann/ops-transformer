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
 * \file ts_ffn_kernel.cpp
 * \brief FFN kernel用例.
 */

#include "ts_ffn.h"

namespace {
// TEST_P(Ts_FFN_WithParam_Ascend910B3, Tc_Kernel_FFN)
// {
//     ASSERT_TRUE(case_->Init());
//     ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
// }

const auto Tc_FFN_Kernel_Case = ::testing::Values(
    FFNCase(
        "FFN_Moe_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2000,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {84, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 5120, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {4, 2560, 5120}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias1", {4, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {84, 5120}, ge::DataType::DT_FLOAT16)},
              {50, 15, 4, 15}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64}, "gelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64}, "relu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64}, "silu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {4, 512, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64}, "silu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 2001,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {8, 5120}, ge::DataType::DT_INT8),
               GenTensor("weight1", {4, 5120, 2560}, ge::DataType::DT_INT8),
               GenTensor("weight2", {4, 2560, 5120}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias1", {4, 2560}, ge::DataType::DT_INT32),
               GenTensor("bias2", {4, 5120}, ge::DataType::DT_INT32), GenTensor("scale", {4}, ge::DataType::DT_FLOAT),
               GenTensor("offset", {4}, ge::DataType::DT_FLOAT),
               GenTensor("deqScale1", {4, 2560}, ge::DataType::DT_INT64),
               GenTensor("deqScale2", {4, 5120}, ge::DataType::DT_INT64),
               GenTensor("y", {8, 5120}, ge::DataType::DT_FLOAT16)},
              {2, 2, 2, 2}, "relu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case9", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 6,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_INT8),
               GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias1", {4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale1", {4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset1", {4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case11", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 7,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param(
            {GenTensor("x", {256, 512}, ge::DataType::DT_BF16),
             GenTensor("weight1", {4, 512, 0}, ge::DataType::DT_BF16),
             GenTensor("weight2", {4, 0, 512}, ge::DataType::DT_BF16),
             GenTensor("expertTokens", {4}, ge::DataType::DT_INT64), GenTensor("y", {256, 512}, ge::DataType::DT_BF16)},
            {64, 64, 64, 64}, "fastgelu", 0, -1),
        0),
    FFNCase(
        "FFN_Moe_Case12", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 7,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_BF16),
               GenTensor("weight1", {4, 512, 0}, ge::DataType::DT_BF16),
               GenTensor("weight2", {4, 0, 512}, ge::DataType::DT_BF16),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT), GenTensor("y", {256, 512}, ge::DataType::DT_BF16)},
              {64, 64, 64, 64}, "fastgelu", 0, -1),
        0),
    FFNCase(
        "FFN_Moe_Case15", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 12,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {256, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_INT8),
               GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias1", {4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {4, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale1", {4, 4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale2", {4, 8, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset1", {4, 4, 256}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset2", {4, 8, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
              {64, 64, 64, 64}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 15,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {4, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {4, 2560, 1536}, ge::DataType::DT_INT8),
               GenTensor("weight2", {4, 1536, 2560}, ge::DataType::DT_INT8),
               GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
               GenTensor("bias1", {4, 1536}, ge::DataType::DT_FLOAT16),
               GenTensor("bias2", {4, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale1", {4, 1536}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_scale2", {4, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset1", {4, 1536}, ge::DataType::DT_FLOAT16),
               GenTensor("antiquant_offset2", {4, 2560}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {4, 2560}, ge::DataType::DT_FLOAT16)},
              {1, 1, 1, 1}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_Case17", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 0,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param(
            {GenTensor("x", {64, 768}, ge::DataType::DT_FLOAT16),
             GenTensor("weight1", {768, 1024}, ge::DataType::DT_FLOAT16),
             GenTensor("weight2", {1024, 768}, ge::DataType::DT_FLOAT16),
             GenTensor("bias1", {1024}, ge::DataType::DT_FLOAT16), GenTensor("bias2", {768}, ge::DataType::DT_FLOAT16),
             GenTensor("y", {64, 768}, ge::DataType::DT_FLOAT16)},
            {}, "fastgelu", 1, -1),
        0),
    FFNCase(
        "FFN_geglu_Case19", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {6, 640}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {640, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {256, 640}, ge::DataType::DT_FLOAT16),
               GenTensor("bias1", {512}, ge::DataType::DT_FLOAT16), GenTensor("bias2", {640}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {6, 640}, ge::DataType::DT_FLOAT16)},
              {}, "geglu", 1, -1),
        0),
    FFNCase(
        "FFN_swiglu_Case20", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {6, 640}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {640, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {256, 640}, ge::DataType::DT_FLOAT16),
               GenTensor("bias1", {512}, ge::DataType::DT_FLOAT16), GenTensor("bias2", {640}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {6, 640}, ge::DataType::DT_FLOAT16)},
              {}, "swiglu", 1, -1),
        0),
    FFNCase(
        "FFN_reglu_Case21", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, true),
               ExpectInfo(true, 2,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensor("x", {6, 640}, ge::DataType::DT_FLOAT16),
               GenTensor("weight1", {640, 512}, ge::DataType::DT_FLOAT16),
               GenTensor("weight2", {256, 640}, ge::DataType::DT_FLOAT16),
               GenTensor("bias1", {512}, ge::DataType::DT_FLOAT16), GenTensor("bias2", {640}, ge::DataType::DT_FLOAT16),
               GenTensor("y", {6, 640}, ge::DataType::DT_FLOAT16)},
              {}, "reglu", 1, -1),
        0),
    FFNCase(
        "FFN_Moe_Case22", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 13,
                          ExpectInfo::kFullTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param(
            {GenTensor("x", {8, 5120}, ge::DataType::DT_INT8),
             GenTensor("weight1", {4, 5120, 0}, ge::DataType::DT_INT8),
             GenTensor("weight2", {4, 0, 5120}, ge::DataType::DT_INT8),
             GenTensor("expertTokens", {4}, ge::DataType::DT_INT64), GenTensor("bias1", {4, 0}, ge::DataType::DT_INT32),
             GenTensor("bias2", {4, 5120}, ge::DataType::DT_INT32), GenTensor("scale", {4}, ge::DataType::DT_FLOAT),
             GenTensor("offset", {4}, ge::DataType::DT_FLOAT), GenTensor("deqScale1", {4, 0}, ge::DataType::DT_FLOAT),
             GenTensor("deqScale2", {4, 5120}, ge::DataType::DT_FLOAT),
             GenTensor("y", {8, 5120}, ge::DataType::DT_FLOAT16)},
            {2, 2, 2, 2}, "relu", 1, -1),
        0));

// INSTANTIATE_TEST_SUITE_P(FFN, Ts_FFN_WithParam_Ascend910B3, Tc_FFN_Kernel_Case);
} // namespace
