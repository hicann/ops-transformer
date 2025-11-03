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
 * \file test_incre_flash_attention_tiling.cpp
 * \brief IncreFlashAttention用例.
 */

#include "ts_ifa.h"
class Ts_Ifa_Ascend910B2_Case : public Ts_Ifa_WithParam_Ascend910B2 {};

TEST_F(Ts_Ifa_Ascend910B2, case_atten_mask)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.numHeads = 20;
    // cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_query)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mOpInfo.mExp.mSuccess = false; // expected exec result
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_query_bf16)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_key)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 100;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.key = Tensor("key", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {cs.mParam.b, cs.mParam.n, 0, cs.mParam.d}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_quant_1)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.qDataType = ge::DT_INT8;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::ALL_1;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_atten_mask_1)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {2, 40, 1, 1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_hd)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 11;
    cs.mParam.s = 2014;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Ifa_Ascend910B2, case_atten_mask_2)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.numHeads = 40;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_BN_greater_than_core_number)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 49;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.numHeads = 49;
    cs.mParam.kvNumHeads = 1;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_quant_1)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiQuant_1)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_N_1_D;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_qunat_kvAntiQuant_1)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_N_1_D;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.actualSeqLength = {1000};
    cs.mParam.numHeads = 40;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiQuant_bf16)
{
    IfaCase cs;
    cs.mParam.b = 5;
    cs.mParam.n = 40;
    cs.mParam.s = 16000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_N_1_D;
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.actualSeqLength = {1000, 1000, 1000, 1000, 1000};
    cs.mParam.numHeads = 40;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiQuant_unflash_splitB_largeS)
{
    IfaCase cs;
    cs.mParam.b = 96;
    cs.mParam.n = 11;
    cs.mParam.s = 8192;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_BF16;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_N_D;
    cs.mParam.numHeads = 11;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_kvPadding)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.enbaleKvPaing = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvPadding)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.enbaleKvPaing = true;
    cs.mParam.kvPaddingSize = 1;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvPadding_no_act_sqe_len)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 4096;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.enbaleKvPaing = true;
    cs.mParam.kvPaddingSize = 1;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiquant_bf16_quant_scale2_type_3)
{
    IfaCase cs;
    cs.mParam.b = 5;
    cs.mParam.n = 40;
    cs.mParam.s = 16000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 40;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_INT8;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.quantType = QuantShapeType::POST_1;
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_H;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}
TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiquant_workspace_opt_unflashSplitB_largeS)
{
    IfaCase cs;
    cs.mParam.b = 96;
    cs.mParam.n = 11;
    cs.mParam.s = 8192;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_BF16;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_H;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiquant_workspace_opt_unflashSplitB)
{
    IfaCase cs;
    cs.mParam.b = 96;
    cs.mParam.n = 11;
    cs.mParam.s = 4096;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.qDataType = ge::DT_BF16;
    cs.mParam.kvDataType = ge::DT_INT8;
    cs.mParam.outDataType = ge::DT_BF16;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.antiQuantType = AntiQuantShapeType::_2_H;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_hn_bsh)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 11;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_attenmask_fp16)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 4;
    cs.mParam.s = 8192;
    cs.mParam.d = 256;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 4;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.pseShiftType = PseShiftShapeType::B_N_1_S;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_float)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_bf16)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_bool)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_BOOL, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_BOOL, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_default)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT16, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_layout)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_1)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 20, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_2)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 20, 10, 11}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_bsh)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {3, 1, 10}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_bsnd)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {3, 1, 10, 11}, "BSND", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10, 11}, "BSND", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10, 11}, "BSND", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10, 11}, "BSND", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_list)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {3, 1, 10, 11}, "BSND", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {1, 2048, 10, 11}, "BSND", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {1, 2048, 10, 11}, "BSND", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 1, 10, 11}, "BSH", cs.mParam.outDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_keyshape_size)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 0}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 0}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_all_int8)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_int8_float16)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_all_float)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_int16_float)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_INT16, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_maskshape_size)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {2, 40, 0, 1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_masksize_batchsize)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {2, 40, 1, 1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_masksize_maxcctualseq)
{
    IfaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 40;
    cs.mParam.s = 1000;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.attenMaskType = AttenMaskShapeType::B_N_1_S;
    cs.mParam.actualSeqLength = {1024};
    cs.mParam.numHeads = 40;
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {1, 40, 1, 100}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_bsh)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {4, 1, 10}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_bnsd)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1, 10}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 1, 10}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 1, 10}, "BNSD", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {4, 1, 1, 10}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_bh)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {4, 1}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_b1)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_b5)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_bsh)
{
    IfaCase cs;
    cs.mParam.b = 3;
    cs.mParam.n = 48;
    cs.mParam.s = 10;
    cs.mParam.d = 16;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 48;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.blockSize = 16;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.blocktable = {5, 5};
    ASSERT_TRUE(cs.Init());
    cs.key = Tensor("key", {9, 4, 768}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {9, 4, 768}, "BSH", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_bnsd)
{
    IfaCase cs;
    cs.mParam.b = 3;
    cs.mParam.n = 48;
    cs.mParam.s = 10;
    cs.mParam.d = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 48;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.blockSize = 16;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.blocktable = {5, 5};
    ASSERT_TRUE(cs.Init());
    cs.key = Tensor("key", {4, 1, 1, 10}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 1, 1, 10}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_nz)
{
    IfaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 128;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 128;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.blocktable = {5, 5};
    ASSERT_TRUE(cs.Init());
    cs.key = Tensor("key", {16, 1, 32, 128, 16}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {16, 1, 32, 128, 16}, "BNSD", cs.mParam.qDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quant_offset2_scale2)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quant_offset2)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_exist)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_nullptr)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_inputqtype)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_dim)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantScale_dim_bnsd)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 2, 2, 2}, "4", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("quantScale2", {2, 2, 2, 2}, "4", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_dim_bh)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("quantScale2", {2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_antiquantoffset)
{
    IfaCase cs;
    cs.mParam.b = 4;
    cs.mParam.s = 2048;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.attenMaskType = AttenMaskShapeType::B_1_S;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_bnsd_big_batch_long_sequence)
{
    IfaCase cs;
    cs.mParam.b = 50;
    cs.mParam.n = 8;
    cs.mParam.s = 16 * 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = std::vector<int64_t>(50, 16 * 1024);
    cs.mParam.blocktable = {50, 129};
    ge::DataType qDataType = ge::DataType::DT_FLOAT16;
    ge::DataType kvDataType = ge::DataType::DT_FLOAT16;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {50, 8, 1, 128}, "BNSD", qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {6450, 1, 128, 128}, "BNSD", kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {6450, 1, 128, 128}, "BNSD", kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = true;
    // ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_int4_bnsd)
{
    IfaCase cs;
    cs.mParam.b = 50;
    cs.mParam.n = 8;
    cs.mParam.s = 16 * 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 8;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = std::vector<int64_t>(50, 16 * 1024);
    cs.mParam.blocktable = {50, 129};
    ge::DataType qDataType = ge::DataType::DT_FLOAT16;
    ge::DataType kvDataType = ge::DataType::DT_INT4;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {50, 8, 1, 128}, "BNSD", qDataType, ge::FORMAT_ND);
    cs.key = Tensor("key", {6450, 1, 128, 128}, "BNSD", kvDataType, ge::FORMAT_ND);
    cs.value = Tensor("value", {6450, 1, 128, 128}, "BNSD", kvDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}