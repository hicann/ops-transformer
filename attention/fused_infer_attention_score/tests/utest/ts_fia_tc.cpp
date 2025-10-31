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
 * \file ts_fia_tc.cpp
 * \brief FusedInferAttentionScore用例.
 */

#include "ts_fia.h"

TEST_F(Ts_Fia_Ascend910B1, case_001)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Fia_Ascend910B1, case_value_antiquant_scale)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_key_antiquant_scale_1)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_value_antiquant_offset)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_key_antiquant_scale_2)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_key_antiquant_scale_3)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_compare_key_value_offset)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 4}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_compare_key_value_scale)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 4}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_compare_key_antiquant_mode)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 3;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_1)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 3}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd2) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd3) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd4) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd5) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd6) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd7) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 32;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 32, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd8) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 256;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 256, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd9) // for msd PFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd1) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd2) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd3) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd4) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd5) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 17, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd6) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 32;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 32, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 16}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_fp16_msd7) // for msd PFA fp16 pertoken
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 256;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 256, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 256, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {4, 256}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_2)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 2, 2}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 2, 2}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_3)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_actual_share_prefix_bsh)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 4, 2048}, "3", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_actual_share_prefix_bnsd)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1, 10}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1, 2048, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 2048, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 10, 1, 10}, "4", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 10, 1, 10}, "4", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 10, 0, 10}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 10, 0, 10}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1, 10, 0, 10}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_actual_share_pre_fixlen)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 4, 2048}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 4, 2048}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_fia_1)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 6, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 6, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_0) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_1) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_2) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_antiquant_mode_msd_kvSep_3) // for msd IFA
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kvDataType = ge::DT_INT8;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 16, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 4, 16}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 4, 16}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_bsh) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 2, 16384}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 128, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 128, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 2, 16384}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 2, 2048}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BS", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);


    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_no_pa) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 2, 16384}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {32, 256, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {32, 256, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 2, 16384}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 2, 2048}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {32, 256, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_qs_no_equal) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_witmask) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3; // 3, sparsemode 3

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BB", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_witmask_mtp_sparsemode_err) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BB", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_witmask_mtp_maskshape_error) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;  // 3, sparsemode 3

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2047}, "BB", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_witmask_sparsemode) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;  // 3, sparsemode 3

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2047}, "BB", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_keyrope_null) // for mla
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_kv_nz) // for mla kv nz
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 2, 16384}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 2, 16384}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);

    cs.queryRope = Tensor("queryRope", {32, 2, 2048}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 4, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_deepseek_mla_kv_nz_bsnd) // for mla kv nz
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 32;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 2, 32, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 2, 32, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);

    cs.queryRope = Tensor("queryRope", {32, 2, 32, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 4, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_sliding_pa_kv_unequal_bsh) // for IFA sliding page attention
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.sparse_mode = 0;
    cs.mParam.pre_tokens = 128;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 192}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 128, 192}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BSH", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000001)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 192}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 192}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 192}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 16, 1, 192}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000002)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 17, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 17, 192}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 17, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000003)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 1024}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 1024}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 128}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000004)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 1, 4096}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000005)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {30, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000006)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 2, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000007)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 1, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000008)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 128}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000009)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000010)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 1, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000011)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {201, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000012)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {134, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {705, 1, 16, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {705, 1, 16, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {134, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {134}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {134, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {705, 2, 16, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {134, 16}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000013)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 256, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000014)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {134, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {705, 1, 16, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {705, 1, 16, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {134, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {134}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {134, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {705, 1, 16, 32}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {134, 16}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000015)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 256, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {24, 1152}, "BNSD", ge::DT_UINT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000016)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 128;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {64}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {64, 4, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {547, 16, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {64, 60}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BB", ge::DT_BOOL, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000017)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 128;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {64}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {64, 4, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {547, 16, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {64, 60}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2048, 2048}, "BB", ge::DT_BOOL, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000018)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000019)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 256, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000021)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 256;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 256, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 256, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 256, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000022)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BNSD", ge::DT_UINT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000023)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = -1;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000024)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 128;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {64}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {64, 4, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {547, 16, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {64, 60}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BB", ge::DT_BOOL, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000025)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 128;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = -1;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {547, 16, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 4, 128, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {64}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {64, 4, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {547, 16, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {64, 60}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BB", ge::DT_BOOL, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000026)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {24, 1, 1028, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {24, 1, 1028, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {24, 1, 1028, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000027)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;
    cs.mParam.softmax_lse_flag = 1;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000028)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1, 64, 1, 1028}, "B_1_N_S", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000029)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000032)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 0, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000033)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 0, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000034)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 0, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_000036)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {200, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {24, 64, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {24}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24, 64, 1, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {200, 128, 64}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {24, 9}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 1028, 512}, "4", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 1028, 512}, "4", ge::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000038)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 2, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 2, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 2, 4, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000039)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 192}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 192}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 1, 4, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000041)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 16, 128, 32}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 16, 128, 32}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 1, 2, 128, 32}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000042)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 16;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1000, 1, 32, 16, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1000, 1, 32, 16, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1000, 1, 4, 16, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 128}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000043)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 104, 1, 4, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000044)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 1, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000045)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {100, 1, 4, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000046)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 2, 4, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000047)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 1, 4, 16, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000048)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 1, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 1, 64, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 1, 2, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000050)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 2, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 2, 64, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ifa_exception_ds_pa_nz_000051)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 0.044194174f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.sparse_mode = 0;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8, 64, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104, 1, 32, 128, 16}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104, 1, 32, 128, 16}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {8, 64, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {8}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8, 64, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {104, 1, 4, 128, 16}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {8, 16}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_nz_000052)
{
    FiaCase cs;
    cs.mParam.numHeads = 64;
    cs.mParam.layout = "BSND";

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {8,1,64,512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {104,1,32,128,16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {104,1,32,128,16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000054)
{
    FiaCase cs;
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {0,0,0,1048576};

    ASSERT_TRUE(cs.Init());
    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.query = Tensor("query", {1048576,8,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1048576,8,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1048576,8,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000055)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {3,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {3,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000056)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 2;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,2,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,2,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,2,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {4,1,640,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000057)
{
    FiaCase cs;
    cs.mParam.layout = "TND";

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,384}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,384}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000058)
{
    FiaCase cs;
    cs.mParam.layout = "TND";

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000059)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512,1}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512,1}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64,1}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000060)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,20};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {20,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {20,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {20,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000061)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1,2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000062)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {3,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000063)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,8,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000064)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,32}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,32}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000065)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 0;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {640,1,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {640,1,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {640,1,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000066)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 16;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,16,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,16,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,16,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000067)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,256}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,256}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000068)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,0,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000069)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {-1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000070)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640,1};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000071)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {2,640}, "TND", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000072)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 3;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000073)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 3;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {1,2048,2048}, "TND", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000074)
{
    FiaCase cs;
    cs.mParam.layout = "NTD_TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {16,2,512}, "NTD_TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {16,2,512}, "NTD_TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {16,2,64}, "NTD_TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "NTD_TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "NTD_TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "NTD_TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "NTD_TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000075)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 1;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {2,640}, "TND", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000076)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000077)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000078)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = true;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.softmaxLse = Tensor("softmaxLse", {2,16,1}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    
    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000079)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;
    
    cs.mParam.actualSeqLength = {1,1,1,2};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {16,2,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {2,16,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2,16,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {11,1,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000080)
{
    FiaCase cs;
    cs.mParam.layout = "BNSD_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 4095;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;
    
    cs.mParam.actualSeqLengthKV = {3184,1760,3984,240,2880,3232,256,4095};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {16,8,1,512}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.query = Tensor("query", {8,1,8192}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8,1,1024}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.key = TensorList("key", {189,128,512}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {189,128,64}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.value = TensorList("value", {189,128,512}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {8,32}, "BNSD_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000081)
{
    FiaCase cs;
    cs.mParam.layout = "BSND_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 3;
    cs.mParam.next_tokens = 1866;
    
    cs.mParam.sparse_mode = 3;
    cs.mParam.innerPrecise = 0;
    
    cs.mParam.actualSeqLengthKV = {30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,1866,883};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {16,16,3,512}, "BSND_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.query = Tensor("query", {16,3,16,512,1}, "BSND_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {16,3,16,64,1}, "BSND_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.key = TensorList("key", {30,128,512}, "BSND_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {30,128,64}, "BSND_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.value = TensorList("value", {30,128,512}, "BSND_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {2048,2048}, "BSND_NBSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {16,15}, "BSND_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000082)
{
    FiaCase cs;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 10;
    cs.mParam.next_tokens = 384;
    
    cs.mParam.sparse_mode = 3;
    cs.mParam.innerPrecise = 0;
    
    cs.mParam.actualSeqLengthKV = {59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,384};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,24,10,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.query = Tensor("query", {24,4096}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.key = TensorList("key", {72,128,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {72,128,64}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.value = TensorList("value", {72,128,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {2048,2048}, "BSH_NBSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {24,3}, "BSH_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000083)
{
    FiaCase cs;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 15;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;
    
    cs.mParam.actualSeqLength = {0,0,0,1,1,2,3,3,4,5,5,6,6,6,7,7};
    cs.mParam.actualSeqLengthKV = {6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,15};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,7,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {1,7,8,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1,7,8,64}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {16,128,64}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {16,1}, "TND_NTD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000084)
{
    FiaCase cs;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 15;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;
    
    cs.mParam.actualSeqLength = {0,0,0,1,1,2,3,3,4,5,5,6,6,6,7,7};
    cs.mParam.actualSeqLengthKV = {6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,15};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,7,128}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {7,8,128}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {7,8,64}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {16,128,64}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {16,1}, "TND_NTD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000085)
{
    FiaCase cs;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 15;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;
    
    cs.mParam.actualSeqLength = {0,0,0,1,1,2,3,3,4,5,5,6,6,6,7,7};
    cs.mParam.actualSeqLengthKV = {6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,15};

    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,7,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {7,8,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {7,8,32}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {16,128,32}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {16,1}, "TND_NTD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000086)
{
    FiaCase cs;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 2;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 1105;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;

    cs.mParam.actualSeqLengthKV = {656,176,1024,96,416,240,96,912,0,336,320,192,560,928,208,896,1072,816,2,1,304,896,1072,1,752,832,64,672,496,272,1056,1105,132,613,171,418,3,830,676,644,297,261};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {2,32,1,512}, "BSH_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {32,1,1024}, "BSH_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32,1,128}, "BSH_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {185,1,32,128,16}, "BSH_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {185,1,4,128,16}, "BSH_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {185,1,32,128,16}, "BSH_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {32,9}, "BSH_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000087)
{
    FiaCase cs;
    cs.mParam.layout = "BNSD_BSND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 4095;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLengthKV = {3184,1760,3984,240,2880,3232,256,4095};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,1,16,512}, "BNSD_BSND", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {8,16,1,512}, "BNSD_BSND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8,16,1,64}, "BNSD_BSND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {189,128,512}, "BNSD_BSND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {189,128,64}, "BNSD_BSND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {189,128,512}, "BNSD_BSND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {8,32}, "BNSD_BSND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000088)
{
    FiaCase cs;
    cs.mParam.layout = "BSH/NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 1105;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;

    cs.mParam.actualSeqLengthKV = {656,176,1024,96,416,240,96,912,0,336,320,192,560,928,208,896,1072,816,2,1,304,896,1072,1,752,832,64,672,496,272,1056,1105,132,613,171,418,3,830,676,644,297,261};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,32,1,512}, "BSH/NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {32,1,4096}, "BSH/NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32,1,512}, "BSH/NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {185,1,32,128,16}, "BSH/NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {185,1,4,128,16}, "BSH/NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {185,1,32,128,16}, "BSH/NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {32,9}, "BSH/NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000089)
{
    FiaCase cs;
    cs.mParam.layout = "BSH_NBSD_";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 1105;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;

    cs.mParam.actualSeqLengthKV = {656,176,1024,96,416,240,96,912,0,336,320,192,560,928,208,896,1072,816,2,1,304,896,1072,1,752,832,64,672,496,272,1056,1105,132,613,171,418,3,830,676,644,297,261};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,32,1,512}, "BSH_NBSD_", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {32,1,4096}, "BSH_NBSD_", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32,1,512}, "BSH_NBSD_", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {185,1,32,128,16}, "BSH_NBSD_", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {185,1,4,128,16}, "BSH_NBSD_", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {185,1,32,128,16}, "BSH_NBSD_", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {32,9}, "BSH_NBSD_", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000090)
{
    FiaCase cs;
    cs.mParam.layout = "BNSD_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 4095;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLengthKV = {3184,1760,3984,240,2880,3232,256,4095};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,16,1,512}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {8,16,1,512}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8,16,1,64}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {189,128,512}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {189,128,64}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {189,128,512}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {8,32}, "BNSD_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000091)
{
    FiaCase cs;
    cs.mParam.layout = "BSND_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 3;
    cs.mParam.next_tokens = 1866;
    
    cs.mParam.sparse_mode = 3;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLengthKV = {30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,1866,883};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {16,3,16,512}, "BSND_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {16,3,16,512}, "BSND_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {16,3,16,64}, "BSND_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {30,128,512}, "BSND_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {30,128,64}, "BSND_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {30,128,512}, "BSND_NBSD", ge::DT_BF16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {2048,2048}, "BSND_NBSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {16,15}, "BSND_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000092)
{
    FiaCase cs;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 10;
    cs.mParam.next_tokens = 384;
    
    cs.mParam.sparse_mode = 3;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLengthKV = {59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,384};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {24,10,4096}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.query = Tensor("query", {24,10,4096}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {24,10,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.key = TensorList("key", {72,128,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {72,128,64}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.value = TensorList("value", {72,128,512}, "BSH_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.attenMask = Tensor("attenMask", {2048,2048}, "BSH_NBSD", ge::DT_INT8, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {24,3}, "BSH_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000093)
{
    FiaCase cs;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 15;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLength = {0,0,0,1,1,2,3,3,4,5,5,6,6,6,7,7};
    cs.mParam.actualSeqLengthKV = {6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,15};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {7,8,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {7,8,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {7,8,64}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {16,128,64}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {16,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {16,1}, "TND_NTD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000094)
{
    FiaCase cs;
    cs.mParam.layout = "BNSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 986;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;

    cs.mParam.actualSeqLengthKV = {384,384,384,384,384,384,384,986};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,16,1,512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.query = Tensor("query", {8,16,1,512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8,16,1,64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.key = TensorList("key", {29,1,32,128,16}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {29,1,4,128,16}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.value = TensorList("value", {29,1,32,128,16}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {8,8}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000095)
{
    FiaCase cs;
    cs.mParam.layout = "BNSD_NBSD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 16;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 4095;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLengthKV = {3184,1760,3984,240,2880,3232,256,4095};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {16,8,1,512}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.query = Tensor("query", {8,16,1,512}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {8,16,1,64}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.key = TensorList("key", {189,1,32,128,16}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {189,1,4,128,16}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.value = TensorList("value", {189,1,32,128,16}, "BNSD_NBSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {8,32}, "BNSD_NBSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_pa_000110)
{
    FiaCase cs;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 1;
    cs.mParam.next_tokens = 640;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 1;

    cs.mParam.actualSeqLength = {0,0,0,1};
    cs.mParam.actualSeqLengthKV = {16,16,16,640};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {8,1,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {1,8,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {11,1,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {11,1,128,512}, "TND_NTD", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,5}, "TND_NTD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_TND_000111)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 2147483647;
    cs.mParam.next_tokens = 2147483647;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLength = {1,2,4};
    cs.mParam.actualSeqLengthKV = {1024,1024,1024};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {4,8,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {4,8,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4,8,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {32,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {32,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {32,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,16}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, ifa_exception_ds_TND_000112)
{
    FiaCase cs;
    cs.mParam.layout = "TND";
    cs.mParam.softmax_lse_flag = false;

    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;

    cs.mParam.pre_tokens = 2147483647;
    cs.mParam.next_tokens = 2147483647;
    
    cs.mParam.sparse_mode = 0;
    cs.mParam.innerPrecise = 0;

    cs.mParam.actualSeqLength = {1,2,4,4,4};
    cs.mParam.actualSeqLengthKV = {1024,1024,1024,1024,1024};
    
    ASSERT_TRUE(cs.Init());
    cs.attentionOut = Tensor("attentionOut", {4,8,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.query = Tensor("query", {4,8,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4,8,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.key = TensorList("key", {32,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {32,128,64}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.value = TensorList("value", {32,128,512}, "TND", ge::DT_BF16, ge::FORMAT_ND);

    cs.blocktable = Tensor("blockTable", {4,16}, "TND", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}