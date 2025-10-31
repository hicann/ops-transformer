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
 * \file ts_ffn_aclnn.cpp
 * \brief FFN ACLNN 测试用例.
 */

#include "ts_aclnn_ffn.h"

namespace {
TEST_P(Ts_Aclnn_FFN_WithParam_Ascend910B3, Tc_Aclnn_FFN)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_FFN_Aclnn_Case = ::testing::Values(

    AclnnFFNCase("Test_001", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {84, 5120}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight1", {4, 5120, 2560}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight2", {4, 2560, 5120}, ge::DataType::DT_FLOAT16),
                                GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {4, 2560}, ge::DataType::DT_FLOAT16),
                                GenTensor("y", {84, 5120}, ge::DataType::DT_FLOAT16)},
                               {50, 15, 4, 15}, "fastgelu", 1, -1, FunctionType::NO_QUANT, AclnnFFNVersion::V1)),
    AclnnFFNCase("Test_002", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {4, 2560}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight1", {4, 2560, 1530}, ge::DataType::DT_INT8),
                                GenTensor("weight2", {4, 1530, 2560}, ge::DataType::DT_INT8),
                                GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {4, 1530}, ge::DataType::DT_FLOAT16),
                                GenTensor("bias2", {4, 2560}, ge::DataType::DT_FLOAT16),
                                GenTensor("antiquant_scale1", {4, 1530}, ge::DataType::DT_FLOAT16),
                                GenTensor("antiquant_scale2", {4, 2560}, ge::DataType::DT_FLOAT16),
                                GenTensor("antiquant_offset1", {4, 1530}, ge::DataType::DT_FLOAT16),
                                GenTensor("antiquant_offset2", {4, 2560}, ge::DataType::DT_FLOAT16),
                                GenTensor("y", {4, 2560}, ge::DataType::DT_FLOAT16)},
                               {1, 1, 1, 1}, "gelu", 1, -1, FunctionType::ANTIQUANT, AclnnFFNVersion::V2)),
    AclnnFFNCase("Test_003", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {256, 512}, ge::DataType::DT_INT8),
                                GenTensor("weight1", {4, 512, 256}, ge::DataType::DT_INT8),
                                GenTensor("weight2", {4, 256, 512}, ge::DataType::DT_INT8),
                                GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {4, 256}, ge::DataType::DT_INT32),
                                GenTensor("bias2", {4, 512}, ge::DataType::DT_INT32),
                                GenTensor("scale", {4}, ge::DataType::DT_FLOAT),
                                GenTensor("offset", {4}, ge::DataType::DT_FLOAT),
                                GenTensor("deqScale1", {4, 256}, ge::DataType::DT_INT64),
                                GenTensor("deqScale2", {4, 512}, ge::DataType::DT_INT64),
                                GenTensor("y", {256, 512}, ge::DataType::DT_FLOAT16)},
                               {64, 64, 64, 64}, "fastgelu", 1, -1, FunctionType::QUANT, AclnnFFNVersion::V3)),
    AclnnFFNCase("Test_geglu_004", true,                                 /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {6, 328}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight1", {328, 512}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight2", {256, 328}, ge::DataType::DT_FLOAT16),
                                GenTensor("bias1", {512}, ge::DataType::DT_FLOAT16),
                                GenTensor("bias2", {328}, ge::DataType::DT_FLOAT16),
                                GenTensor("y", {6, 328}, ge::DataType::DT_FLOAT16)},
                               {}, "geglu", 1, -1, FunctionType::NO_QUANT, AclnnFFNVersion::V1)),
    AclnnFFNCase("Test_005", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {50, 640}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight1", {4, 640, 256}, ge::DataType::DT_FLOAT16),
                                GenTensor("weight2", {4, 256, 640}, ge::DataType::DT_FLOAT16),
                                GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {4, 256}, ge::DataType::DT_FLOAT16),
                                GenTensor("y", {50, 640}, ge::DataType::DT_FLOAT16)},
                               {15, 15, 4, 16}, "silu", 0, -1, FunctionType::NO_QUANT, AclnnFFNVersion::V1)),
    AclnnFFNCase("Test_006", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {50, 128}, ge::DataType::DT_BF16),
                                GenTensor("weight1", {4, 128, 256}, ge::DataType::DT_BF16),
                                GenTensor("weight2", {4, 256, 128}, ge::DataType::DT_BF16),
                                GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {4, 256}, ge::DataType::DT_FLOAT),
                                GenTensor("y", {50, 128}, ge::DataType::DT_BF16)},
                               {1, 6, 10, 50}, "relu", 0, -1, FunctionType::NO_QUANT, AclnnFFNVersion::V2, true)),
    AclnnFFNCase("Test_007", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {4, 256}, ge::DataType::DT_BF16),
                                GenTensor("weight1", {4, 256, 768}, ge::DataType::DT_INT8),
                                GenTensor("weight2", {4, 768, 256}, ge::DataType::DT_INT8),
                                GenTensor("expertTokens", {4}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {4, 768}, ge::DataType::DT_FLOAT),
                                GenTensor("bias2", {4, 256}, ge::DataType::DT_FLOAT),
                                GenTensor("antiquant_scale1", {4, 768}, ge::DataType::DT_BF16),
                                GenTensor("antiquant_scale2", {4, 256}, ge::DataType::DT_BF16),
                                GenTensor("antiquant_offset1", {4, 768}, ge::DataType::DT_BF16),
                                GenTensor("antiquant_offset2", {4, 256}, ge::DataType::DT_BF16),
                                GenTensor("y", {4, 256}, ge::DataType::DT_BF16)},
                               {1, 1, 1, 1}, "gelu", 1, -1, FunctionType::ANTIQUANT, AclnnFFNVersion::V3)),
    AclnnFFNCase("Test_008", true,                                       /* CaseName,Enable */
                 "",                                                     /* DebugInfo */
                 OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                        ExpectInfo(true,                                 /* ExpectSuccess */
                                   ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                   ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                 AclnnFFNParam({GenTensor("x", {16, 32}, ge::DataType::DT_BF16),
                                GenTensor("weight1", {1, 32, 16}, ge::DataType::DT_INT4),
                                GenTensor("weight2", {1, 16, 32}, ge::DataType::DT_INT4),
                                GenTensor("expertTokens", {1}, ge::DataType::DT_INT64),
                                GenTensor("bias1", {1, 16}, ge::DataType::DT_FLOAT),
                                GenTensor("bias2", {1, 32}, ge::DataType::DT_FLOAT),
                                GenTensor("antiquant_scale1", {1, 2, 16}, ge::DataType::DT_BF16),
                                GenTensor("antiquant_scale2", {1, 2, 32}, ge::DataType::DT_BF16),
                                GenTensor("antiquant_offset1", {1, 2, 16}, ge::DataType::DT_BF16),
                                GenTensor("antiquant_offset2", {1, 2, 32}, ge::DataType::DT_BF16),
                                GenTensor("y", {16, 32}, ge::DataType::DT_BF16)},
                               {16}, "gelu", 1, -1, FunctionType::ANTIQUANT, AclnnFFNVersion::V3)),
    AclnnFFNCase(
        "Test_009", true,                                       /* CaseName,Enable */
        "",                                                     /* DebugInfo */
        OpInfo(ControlInfo(true, true),                         /* RunTiling,RunKernel */
               ExpectInfo(true,                                 /* ExpectSuccess */
                          ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                          ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
        AclnnFFNParam(
            {GenTensor("x", {256, 512}, ge::DataType::DT_INT8), GenTensor("weight1", {512, 256}, ge::DataType::DT_INT8),
             GenTensor("weight2", {256, 512}, ge::DataType::DT_INT8), GenTensor("bias1", {256}, ge::DataType::DT_INT32),
             GenTensor("bias2", {512}, ge::DataType::DT_INT32), GenTensor("scale", {1}, ge::DataType::DT_FLOAT),
             GenTensor("offset", {1}, ge::DataType::DT_FLOAT), GenTensor("deqScale1", {256}, ge::DataType::DT_BF16),
             GenTensor("deqScale2", {512}, ge::DataType::DT_BF16), GenTensor("y", {256, 512}, ge::DataType::DT_BF16)},
            {}, "fastgelu", 1, -1, FunctionType::QUANT, AclnnFFNVersion::V3)));

INSTANTIATE_TEST_SUITE_P(FFN, Ts_Aclnn_FFN_WithParam_Ascend910B3, Tc_FFN_Aclnn_Case);
} // namespace