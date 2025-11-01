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
 * \file ts_fia_tiling.cpp
 * \brief FusedInferAttentionScore用例-新UT框架.
 */

#include "ts_fia.h"

using CaseMode = ops::adv::tests::fia::CaseMode;
using CaseKvStorageMode = ops::adv::tests::fia::CaseKvStorageMode;

// // Softmax_Lse--------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckLseExist_001) // 存在性校验
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {}, "BNSD", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseShape_002) // layout=TND, lse=(T,N1,1)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {2, 6, 2}, "TND", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseShape_003) // layout=NTD, lse=(T,N1,1)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.layout = "TND_NTD"; 
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 512}, "NTD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {6, 2, 1}, "TND", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseShape_004) // layout=others, shape=[B,N1,S1,1]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.softmax_lse_flag = 1;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 32, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {2, 6, 1, 1}, "TND", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseDType_006) // dtpye校验，仅支持FP32
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.softmax_lse_flag = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 32, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.softmaxLse = Tensor("softmaxLse", {2, 6, 1}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


// Mask-------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_019) // sparsemod = 0, layout=S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_021) // sparsemod = 0, layout=1S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_022) // sparsemod = 0, layout=BS1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {3, 2, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_023) // sparsemod = 0, layout=11S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2, 1, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_024) // sparsemod = 0, layout=B1S1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {3, 2, 1, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_025) // sparsemod = 2, [2048, 2048]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2048}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_026) // sparsemod = 2, [1, 2048, 2048]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2, 2048, 2048}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskLayout_027) // sparsemod = 2, [1, 1, 2048, 2048]
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 1, 2048, 1}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

// PSE --------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftDType_028) // query_type=FP16, PSE_type=FP16
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {2, 1, 1, 1}, "BNSD", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftDType_029) // query_type=BF16, PSE_type=BF16
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {2, 1, 1, 1}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftShape_030) // BNS1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {3, 1, 1, 1}, "BNSD", ge::DataType::DT_BF16, ge::FORMAT_ND); // {3,64,2,2}
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftShape_031) // 1NS1S2
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 64, 2, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1, 1, 1, 1}, "BNSD", ge::DataType::DT_BF16, ge::FORMAT_ND); // {1,64,2,2}
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


// attenmask-------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaAttentionMask_032) // GQA sparse mode = 2/3/4, 必须带mask
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "B", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttentionMask_033) // MLA D=128 sparse mode = 2/3/4, 必须带mask
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 3};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 128}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {3, 64, 64}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 64}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "B", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


// GQA PA------------------------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaPageAttention_034) // 支持layout: BSH/BNSD/NZ
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.blockSize = 128;
    cs.mParam.layout = "BSND_NBSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 16, 64, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 16, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaPageAttention_035) // blocksize是16的倍数
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 129;
    cs.mParam.actualSeqLengthKV = {1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckGqaPageAttention_036) // blocksize<=1024
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 2048;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = {"actualSeqLengthsKV", {1, 1, 1, 1, 1, 1}, "BNSD", ge::DT_INT64, ge::FORMAT_ND};
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

// GQA 非特性------------------------------------------------------------------------
TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaLayout_039) // only supports BSH/BSND、BNSD、TND、NTD、BSH_BNSD、BSND_BNSD、BNSD_BSND、NTD_TND
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "BSND_NBSD";
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 64, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 64, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaNoQuantLayout_041) // BC, 只支持BSND/BSH、BNSD、TND
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "NTD";
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {64, 3, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 512}, "NTD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaNoQuantLayout_042) // TL, 只支持BSND/BSH、BNSD
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::TENSOR_LIST;
    cs.mParam.layout = "NTD";
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {64, 3, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {{3, 128, 512}, {3, 128, 512}}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {{3, 128, 512}, {3, 128, 512}}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {64, 3, 512}, "NTD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {6, 6}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureGqaNoQuantShape_046) // actualSeqLengthsKVSize_ should <= 64 * 1024
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::GQA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;
    cs.mParam.layout = "TND";
    std::vector<int64_t> temp(64 * 1024, 3);
    cs.mParam.actualSeqLength = temp;
    std::vector<int64_t> temp1(64 * 1024 + 1, 1);
    cs.mParam.actualSeqLengthKV = temp1;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 64, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryDtype_mla_001)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.qDataType = ge::DataType::DT_FLOAT;
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryFormat_mla_004)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {48, 1, 32768}, "BSH", ge::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyDtype_mla_005)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.kDataType = ge::DataType::DT_FLOAT;
    cs.mParam.vDataType = ge::DataType::DT_FLOAT;
    cs.mParam.qDataType = ge::DataType::DT_FLOAT;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyFormat_mla_007)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.key = TensorList("key", {48, 128, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueDtype_mla_008)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.vDataType = ge::DataType::DT_UINT64;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueFormat_mla_010)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.value = TensorList("value", {48, 128, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftDtype_mla_011)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1}, "B", ge::DataType::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskDtype_mla_012)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.qs = 20;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("attenMask", {3, 256}, "BS", ge::DT_FLOAT16, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenMaskFormat_mla_013)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("attenMask", {3, 256}, "BS", ge::DT_BOOL, ge::FORMAT_NC); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckDeqScale1Dtype_mla_014)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.deqScale1 = Tensor("deqScale1", {1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuantScale1Dtype_mla_015)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.quantScale1 = Tensor("quantScale1", {1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckDeqScale2Dtype_mla_016)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.deqScale2 = Tensor("deqScale2", {1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuantScale2Dtype_mla_017)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuantOffset2Dtype_mla_018)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "5", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiquantScaleDtype_mla_019)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiquantScaleFormat_mla_020)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiquantOffsetDtype_mla_021)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiquantOffsetFormat_mla_022)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryPaddingSizeFormat_mla_023)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.queryPaddinSize = Tensor("queryPaddinSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKvPaddingSizeFormat_mla_024)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.kvPaddingSize = Tensor("kvPaddingSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantScaleDtype_mla_025)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_UINT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantScaleFormat_mla_026)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantOffsetDtype_mla_027)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantOffsetFormat_mla_028)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantScaleDtype_mla_029)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantScaleFormat_mla_030)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantOffsetDtype_mla_031)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "B", ge::DataType::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantOffsetFormat_mla_032)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeySharedPrefixDtype_mla_033)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKeySharedPrefixFormat_mla_034)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckValueSharedPrefixDtype_mla_035)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckValueSharedPrefixFormat_mla_036)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryRopeDtype_mla_037)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {48, 1, 4096}, "B", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyRopeDtype_mla_038)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keyRope = Tensor("keyRope", {48, 128, 4096}, "B", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckDequantScaleQueryDtype_mla_039)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.dequantScaleQuery = Tensor("dequantScaleQuery", {4, 1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAttenOutDtype_mla_040)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.attentionOut = Tensor("attentionOut", {48, 1, 32768}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckLseOutDtype_mla_041)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.softmaxLse = Tensor("softmaxLse", {2, 16, 1}, "TND", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckInnerPrecise_mla_043)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.innerPrecise = 20;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiquantMode_mla_044)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.antiquant_mode = 3;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantMode_mla_045)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.key_antiquant_mode = 6;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantMode_mla_046)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.value_antiquant_mode = 6;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyRopeExistancce_mla_047)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keyRope = Tensor();
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryRopeExistancce_mla_048)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor();
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaDtypeList_mla_050)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.layout = "BSH";
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLengthsExistence_mla_051)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());
    cs.actualSeqLengths = Tensor();
    cs.actualSeqLengthsKV = Tensor();

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckantiquantScaleExistence_mla_053)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.antiquantScale = Tensor("antiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckantiquantOffsetExistence_mla_054)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantScaleExistence_mla_055)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyAntiquantOffsetExistence_mla_056)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantScaleExistence_mla_057)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckValueAntiquantOffsetExistence_mla_058)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyRopeAntiquantScaleExistence_mla_059)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckDeqScale1Existence_mla_060)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.deqScale1 = Tensor("deqScale1", {1}, "B", ge::DataType::DT_UINT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQuantScale1Existence_mla_061)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.quantScale1 = Tensor("quantScale1", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckDeqScale2Existence_mla_062)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.deqScale2 = Tensor("deqScale2", {1}, "B", ge::DataType::DT_UINT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckDequantScaleQueryExistence_mla_063)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;

    ASSERT_TRUE(cs.Init());
    cs.dequantScaleQuery = Tensor("dequantScaleQuery", {4, 1, 1}, "BNSD", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQuantScale2Existence_mla_064)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQuantOffset2Existence_mla_065)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
        cs.mParam.outDataType = ge::DataType::DT_INT8;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "5", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPseShiftExistence_mla_066)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1, 64, 1, 4096}, "B_1_N_S", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryPaddingSizeExistence_mla_067)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.queryPaddinSize = Tensor("queryPaddinSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKvPaddingSizeExistence_mla_068)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.kvPaddingSize = Tensor("kvPaddingSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKeySharedPrefixExistence_mla_069)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckValueSharedPrefixExistence_mla_070)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSharedPrefixLenExistence_mla_071)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {3}, "3", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_GetKvCache_mla_001)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_GetMaxBlockNumPerBatch_mla_002)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_GetQueryAndOutLayout_mla_006)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BDNS";
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.query = Tensor("query", {3, 512, 32, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("key", {3, 512, 32, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensQ_mla_004)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {2}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensQData_mla_005)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {3, 3, 3, 50};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {4}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensQData_mla_006)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 1, 66};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {3}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLens_mla_009)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {2}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLens_mla_011)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.actualSeqLength = {1, 1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1};

    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {0}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensKvData_mla_012)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 1};
    cs.mParam.actualSeqLengthKV = {899, 1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {2}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensKvData_mla_013)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 1};
    cs.mParam.actualSeqLengthKV = {-1, 1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {2}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckBlockTable_mla_015)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 3, 3, 3, 3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blocktable", {3, 7}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {8}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQShape_mla_018)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {3, 31, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 31, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQRopeShape_mla_022)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckQRopeShape_mla_023)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 4, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKVDType_mla_025)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShapeForPageAttention_mla_036)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.blockSize = 31;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.query = Tensor("query", {3, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 7}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {3}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShapeForPageAttention_mla_037)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {3, 1, 127, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShapeForPageAttention_mla_038)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.value = TensorList("value", {3, 1, 127, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShapeForPageAttention_mla_040)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("keyrope", {3, 1, 127, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShapeForPageAttention_mla_041)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("keyrope", {3, 1, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttentionMask_mla_045)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.sparse_mode = 0;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("attenMask", {1, 1}, "B", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttentionMask_mla_047)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.sparse_mode = 0;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("attenMask", {1, 899}, "B", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantShape_mla_049)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.kvNumHeads = 2;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantShape_mla_050)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 33;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 33, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.query = Tensor("query", {3, 33, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 33, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantShape_mla_051)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 33, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 33, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantShape_mla_053)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 33, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoquantMask_mla_068)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.sparse_mode = 3;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("attenMask", {48, 4096}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoquantUnsupported_mla_072)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.softmax_lse_flag = 1;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoquantUnsupported_mla_073)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensKv_mla_076)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1};
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blocktable", {3, 7}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {3}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_GetMaxBlockNumPerBatch_mla_077)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blocktable", {3, 7, 100}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {3}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckBlockTable_mla_078)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {3, 32, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 32, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blocktable", {2, 7}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckActualSeqLensQData_mla_079)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {3, 3, 4};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blocktable", {3, 7}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {3}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantLayout_mla_080)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureInOutDtype_mla_082)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.qDataType = ge::DataType::DT_FLOAT16;
    cs.mParam.kDataType = ge::DataType::DT_FLOAT16;
    cs.mParam.vDataType = ge::DataType::DT_FLOAT16;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_002)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_004)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 2;
    cs.mParam.kvNumHeads = 3;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_005)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 5;
    cs.mParam.kvNumHeads = 2;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_011)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {32, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("queryRope", {}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_012)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_021)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND";
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_023)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND_NTD";
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_027)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.actualSeqLength = {1};
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_030)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.actualSeqLength = {1, 2};
    cs.mParam.actualSeqLengthKV = {1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_033)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND";
    cs.mParam.actualSeqLength = {1, 2, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_035)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.actualSeqLength = {1, 2, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_036)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.actualSeqLength = {1, 2, 20};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_037)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 4;
    cs.mParam.layout = "TND";
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_038)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 4;
    cs.mParam.layout = "TND_NTD";
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_mla_042)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND";
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {2, 512, 32, 64}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPageAttentionFlag_mla_043)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {3, 128}, "TND", cs.mParam.kDataType, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InitInOutMode_mla_048)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    ASSERT_TRUE(cs.Init());

    cs.keyRope = Tensor("keyRope", {3, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {3, 32, 1, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_mla_054)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = Tensor();
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_mla_055)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {1, 2}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_mla_056)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {-1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant1_mla_058)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.deqScale1 = Tensor("deqScale1", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant1_mla_059)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";    
    ASSERT_TRUE(cs.Init());

    cs.quantScale1 = Tensor("quantScale1", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant1_mla_060)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.deqScale2 = Tensor("deqScale2", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_mla_085)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_113)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_114)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_115)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_116)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_117)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_118)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_mla_119)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.value_antiquant_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_157)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = -1;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_158)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.kvNumHeads = -1;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessBlockTable_mla_159)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = -1;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProces_mla_160)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.innerPrecise = -1;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_mla_087)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_019)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND";
    cs.mParam.b = 1024 * 1024;
    cs.mParam.qs = 1;
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {1}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_020)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.b = 1024 * 1024;
    cs.mParam.qs = 1;
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {1}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_mla_029)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "TND";
    cs.mParam.b = 2;
    cs.mParam.actualSeqLength = {1, 2};
    cs.mParam.actualSeqLengthKV = {1};
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_mla_041)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {0, 32}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessBlockTable_mla_131)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BSH";
    cs.mParam.actualSeqLengthKV = {512, 512, 512};
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantpa_sliding_001)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.b = 2;
    cs.mParam.n = 128;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 128;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.sparse_mode = 4;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLengthKV = {256, 256};

    ASSERT_TRUE(cs.Init());
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 256, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {4, 1, 4, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {2, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_001)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 103000000000000000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 20;           // expected block dim
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

TEST_F(Ts_Fia_Ascend910_9591, case_value_antiquant_scale)
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

TEST_F(Ts_Fia_Ascend910_9591, case_key_antiquant_scale_1)
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

TEST_F(Ts_Fia_Ascend910_9591, case_value_antiquant_offset)
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

TEST_F(Ts_Fia_Ascend910_9591, case_key_antiquant_scale_2)
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

TEST_F(Ts_Fia_Ascend910_9591, case_key_antiquant_scale_3)
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

TEST_F(Ts_Fia_Ascend910_9591, case_compare_key_value_offset)
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

TEST_F(Ts_Fia_Ascend910_9591, case_compare_key_value_scale)
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

TEST_F(Ts_Fia_Ascend910_9591, case_compare_key_antiquant_mode)
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_1)
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd2) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd3) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd4) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd5) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd6) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd7) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd8) // for msd PFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd1) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd2) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd3) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd4) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd5) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd6) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_fp16_msd7) // for msd PFA fp16 pertoken
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_2)
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_3)
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

TEST_F(Ts_Fia_Ascend910_9591, case_actual_share_prefix_bsh)
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd_kvSep) // for msd IFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd_kvSep_0) // for msd IFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd_kvSep_1) // for msd IFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd_kvSep_2) // for msd IFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_antiquant_mode_msd_kvSep_3) // for msd IFA
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

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_qs_no_equal) // for mla
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

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_witmask_mtp_sparsemode_err) // for mla
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

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_witmask_mtp_maskshape_error) // for mla
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

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_witmask_sparsemode) // for mla
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

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_keyrope_null) // for mla
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

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_keyrope_null_qactqlen32) // for mla
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
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.mParam.kvDataType, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_keyrope_null_qactqlen1) // for mla
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
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_deepseek_mla_keyrope_null_qactqlenq) // for mla
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
    cs.mParam.actualSeqLength = {10};

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {64, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {32, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {32}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {32,32,64}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_pa)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLengthKV = {128};

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_pa_invalid_blockSize)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 129;
    cs.mParam.actualSeqLengthKV = {128};

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_ifa_pa_invalid_blockSize)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 1;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.blockSize = 129;
    cs.mParam.actualSeqLengthKV = {128};

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
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
    cs.mOpInfo.mExp.mTilingKey = 103000000000000000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 20;           // expected block dim

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
    cs.mOpInfo.mExp.mTilingKey = 103000000000000000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 20;           // expected block dim

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

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_keyRope_empty)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 0}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_queryRope_empty)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 0}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_queryRope_nullptr)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_keyRope_nullptr)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BNSD_bError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,1,128,128}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BSH_bError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,128,128}, {1,128,128}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BSND_bError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 128, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BNSD_sError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,1,128,128}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 1, 256, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BSH_sError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,128,128}, {1,128,128}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 256, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BSND_sError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 256, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 128, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BNSD_nError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,1,128,128}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 2, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BSH_nError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,128,128}, {1,128,128}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_tensorList_BSND_nError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kvshape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kvshape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", kvshape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 2, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 128, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_BNSD_ropeDError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 77}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_BSH_ropeDError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 77}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 128, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_mla_rope_BNSD_queryDError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 192}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 192}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSH_hError0)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,64}, {1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,128}, {1,128,128}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSH_hError1)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,128}, {1,128,64}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,128}, {1,128,128}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSH_hError2)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,128}, {1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,128}, {1,128,64}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSH_sError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,128}, {1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,64,128}, {1,128,128}};
    cs.query = Tensor("query", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 128}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BNSD_nError0)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,2,128,128}, {1,1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,1,128,128}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BNSD_dError0)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,1,128,128}, {1,1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,1,128,64}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BNSD_nError1)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,1,128,128}, {1,3,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,1,128,64}, {1,3,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BNSD_dError1)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,1,128,128}, {1,1,128,64}};
    std::vector<std::vector<int64_t>> vShape = {{1,1,128,128}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BNSD_dError2)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,1,128,128}, {1,1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,1,128,128}, {1,1,128,64}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BNSD_sError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,1,64,128}, {1,1,128,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,1,128,128}, {1,1,128,128}};
    cs.query = Tensor("query", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSND_nError0)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,2,128}, {1,128,1,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSND_nError1)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,1,128}, {1,128,2,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSND_dError0)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,1,64}, {1,128,1,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSND_dError1)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,1,128}, {1,128,1,64}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSND_dError2)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,1,128}, {1,128,1,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,128,1,128}, {1,128,1,64}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_tensorList_BSND_sError)
{
    FiaCase cs;
    cs.mParam.b = 2;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;

    ASSERT_TRUE(cs.Init());
    std::vector<std::vector<int64_t>> kShape = {{1,128,1,128}, {1,128,1,128}};
    std::vector<std::vector<int64_t>> vShape = {{1,64,1,128}, {1,128,1,128}};
    cs.query = Tensor("query", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", kShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", vShape, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 128, 1, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseType_error)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 4;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1}, "N", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseType2)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1}, "N", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseType3)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 3;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1}, "N", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = true;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseShift_dimError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1,2}, "BN", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseShift_sError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;
    cs.mParam.actualSeqLength = {8};
    cs.mParam.actualSeqLengthKV = {9};

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1,2}, "BN", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseShift_kvsError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 127, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 127, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 127}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1,2}, "BN", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseShift_TNDsError)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;
    cs.mParam.actualSeqLength = {8,128};
    cs.mParam.actualSeqLengthKV = {9,128};

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {128, 1, 128}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {128, 1, 128}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {128, 1, 128}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1,2}, "BN", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseShift_rope)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1,2}, "BN", ge::DT_FLOAT, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pfa_alibi_pseShift_leftPadding)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 1;
    cs.mParam.s = 128;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.kvDataType = ge::DT_FLOAT16;
    cs.mParam.pseType = 2;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 128, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 127, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 127, 128}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 128, 127}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.qStartIdx = Tensor("qStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.kvStartIdx = Tensor("kvStartIdx", {1}, "B", ge::DT_INT64, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {1,2}, "BN", ge::DT_FLOAT, ge::FORMAT_ND);
    cs.queryPaddinSize = Tensor("queryPaddinSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckPABlockSize_001)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckPABlockSize_002)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 513;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckPABlockSize_003)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.blockSize = 16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckPABlockSize_004)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 17;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckPABlockSize_005)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 17;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_006)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_007)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {0, 1, 1, 1}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_008)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_009)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_010)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_011)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.value = TensorList("value", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_012)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.value = TensorList("value", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_013)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.attentionOut = Tensor("attentionOut", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_014)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.value = TensorList("value", {}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_015)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_016)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_017)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_018)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckBaseInputsNull_019)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_020)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, 20, 1, 512}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_021)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {1, 20, 1, 512}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.value = TensorList("value", {1, 20, 1, 512}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_022)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {1, 20, 1, 512}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 20, 1, 512}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_023)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("key", {2048, 2048}, "BNSD", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_024)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.kvPaddingSize = Tensor("kvPaddingSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    ASSERT_TRUE(cs.Run());
    // cs.mOpInfo.mExp.mSuccess = false;
    // ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_025)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {2048, 2048}, "BNSD", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputParameterFormat_026)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {2048, 2048}, "BNSD", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputAntiquantFormat_027)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputAntiquantFormat_028)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantOffset = Tensor("antiquantOffset", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputAntiquantFormat_029)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputAntiquantFormat_030)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputAntiquantFormat_031)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputAntiquantFormat_032)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_NC);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_057)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 128;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 128;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_058)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_INT8;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_059)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_060)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_061)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.blockSize = 17;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {2, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_062)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.blockSize = 17;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {2, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_063)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.blockSize = 16;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {2, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}



TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_069)
{
    FiaCase cs;
    cs.mParam.b = 65537;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_070)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 513;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckInputFormatAndLimits_071)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 512;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 512;
    cs.mParam.kvNumHeads = 512;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVHeadNum_072)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 20, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 20, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_073)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.s, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, cs.mParam.s, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_074)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.s, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, cs.mParam.s, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_075)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_076)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_077)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 1, 10, 128}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 10, 128}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_078)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 1, 10, 128}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, cs.mParam.s, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 1, 10, 128}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_079)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 10, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 10, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_080)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {2, 10, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 10, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVShape_081)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024; 
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {2, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 10, 1, 128}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_082)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_083)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_084)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.attentionOut = Tensor("attentionOut", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_085)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, 1, cs.mParam.d}, "NBSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_086)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {cs.mParam.b, 1, cs.mParam.n, cs.mParam.d}, "BSND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, 1, cs.mParam.d}, "NBSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_087)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, 1, cs.mParam.d}, "NBSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_088)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, 2, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 2, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 2, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 2, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_089)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, 1, cs.mParam.d}, "NBSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_090)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 1, 10 * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {cs.mParam.b, 1,10 * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_091)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 1, 10 * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {cs.mParam.b, 1,10 * cs.mParam.d}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, 1, cs.mParam.d}, "NBSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_092)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_093)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_094)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {1, cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_095)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {1, cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_096)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_097)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {1, cs.mParam.n, cs.mParam.b, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_098)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_099)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 64;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 64;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, 64, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut =
        Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_100)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_101)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.key = TensorList("key", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, cs.mParam.n, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_102)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.attentionOut = Tensor("attentionOut", {cs.mParam.b, cs.mParam.n, 1, cs.mParam.d, 1}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQKOutShape_103)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 1, 256}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKeyShapeTensor_104)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuant2Shape_105)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {0, 20, 1, 512}, "4", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {0, 20, 1, 512}, "4", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuant2Shape_106)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {0, 20, 1}, "3", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {0, 20, 1}, "3", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuant2Shape_107)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {0, 20}, "3", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {0, 20}, "3", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQuant2Shape_108)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {0}, "1", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {0}, "1", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerHead_109)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.key_antiquant_mode = 3;
    cs.mParam.value_antiquant_mode = 3;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerHead_110)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.key_antiquant_mode = 3;
    cs.mParam.value_antiquant_mode = 3;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerHead_111)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.key_antiquant_mode = 3;
    cs.mParam.value_antiquant_mode = 3;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 1, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerHead_112)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.antiquant_mode = 1;
    cs.mParam.key_antiquant_mode = 3;
    cs.mParam.value_antiquant_mode = 3;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerHead_113)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.key_antiquant_mode = 2;
    cs.mParam.value_antiquant_mode = 2;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerChannel_114)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 20, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 20, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerChannel_115)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 20, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerChannel_116)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 20}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 20}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_117)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.antiquant_mode = 3;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 20, 128}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 20, 128}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_118)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_119)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {1, 20, 1024}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {1, 20}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_120)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {1, 20, 1024}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {1, 20, 1024}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_121)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 20, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 20, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_122)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mParam.antiquant_mode = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 1, 1024}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 1, 1024}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParam_123)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_BF16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 20, 128}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2, 20, 128}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckSupportKVLeftPadding_124)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_INT4;
    cs.mParam.vDataType = ge::DataType::DT_INT4;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 20, 128}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.kvPaddingSize = Tensor("kvPaddingSize", {2, 20, 128}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckSupportKVLeftPadding_125)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.kvPaddingSize = Tensor("kvPaddingSize", {2, 20, 128}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_126)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_127)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_128)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_129)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_130)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_131)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_132)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckBasic_133)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.kvPaddingSize = Tensor("kvPaddingSize", {2, 20, 128}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_135)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 10}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_136)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_137)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {2, 1, 1, 10}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {2, 1, 1, 10}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_138)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_139)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_140)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_141)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 2, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 2, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_SharedPrefixCheckShapes_142)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 1, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_143)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_144)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_145)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_146)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_147)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_148)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {2, 1, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_149)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1, 2, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_150)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1, 20, 2, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPseShift_151)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.pseShift = Tensor("pseShift", {1, 20, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_152)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {4, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_153)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.sparse_mode = 3;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_154)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.sparse_mode = 3;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2048, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_155)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.sparse_mode = 3;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2049, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_156)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.sparse_mode = 3;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2049}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_157)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.sparse_mode = 2;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_158)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_159)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 3;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV ={1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_160)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_161)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2049}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_162)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_163)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BS", ge::DT_FLOAT, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_164)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {1, 2048, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_165)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2049, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_166)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2049}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_167)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAttenMask_168)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.sparse_mode = 3;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2048, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND); 
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_169)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 4, 128, 16}, "BSH", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.dequantScaleQuery = Tensor("dequantScaleQuery", {4, 1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_170)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 4, 128, 16}, "BSH", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.dequantScaleQuery = Tensor("dequantScaleQuery", {4, 1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_171)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_172)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH_BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH_BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_173)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH_BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH_BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 2, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_174)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH_BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH_BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_175)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_176)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 2, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_177)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_178)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaQueryRope_179)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 1, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_180)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.actualSeqLength = {1, 1, 2};
    cs.mParam.actualSeqLengthKV = {1, 1, 2};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {2, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 2, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_181)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 2;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 2, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 2, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_182)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 256}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 256}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_183)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "TND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 128}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaAttrs_184)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD_NBSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1, 512}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BNSD_NBSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BNSD_NBSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 4, 1, 512}, "BNSD_NBSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BNSD_NBSD", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BNSD_NBSD", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD_NBSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_185)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_186)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 4, 128, 16}, "BSH", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.dequantScaleQuery = Tensor("dequantScaleQuery", {4, 1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_187)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 4, 128, 16}, "BSH", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.dequantScaleQuery = Tensor("dequantScaleQuery", {4, 1, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_188)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 12000000000222320; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_189)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_190)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_191)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", { 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_192)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {4, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_193)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 128, 513}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {4, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_194)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 128, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {4, 129, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_195)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 128, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 512}, "TND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {4, 128, 65}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {3, 2}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_196)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1 , 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 16, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_197)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1 , 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 129, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_198)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1 , 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 513}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 16, 128, 32}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_199)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1 , 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 2, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_200)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1 , 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 129, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_201)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1 , 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 63}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_202)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 2, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_203)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 17, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_204)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 129, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_205)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 33}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_206)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 2, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_207)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 3, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_208)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 129, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaKeyRope_209)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 33}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_210)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_211)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_212)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 128, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_213)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_214)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 128, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 64}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_215)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_216)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 128, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 128, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {4, 128, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    //cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_BF16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_BF16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_217)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mParam.softmax_lse_flag = 1;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_218)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 128, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.quantScale2 = Tensor("quantScale2", {1}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 128, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_219)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 1;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1, 1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1};
    cs.mParam.blockSize = 256;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 1, 16, 256, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 1, 16, 256, 32}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {4, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 2, 256, 32}, "BSH", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckMlaMisc_220)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 64;
    cs.mOpInfo.mExp.mTilingKey = 15000000020322321; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 4;            // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 512}, "BNSD", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 64, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 64, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 512}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 64, 64}, "BSH", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910_9591, case_pa_bsh_padding)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.antiquant_mode = 0;
    cs.mParam.key_antiquant_mode = 0;
    cs.mParam.value_antiquant_mode = 0;
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.scaleValue = 1.0f;
    cs.mParam.actualSeqLength = {256};
    cs.mParam.actualSeqLengthKV = {512};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 512, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 128, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 128, 128}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
    cs.queryPaddinSize = Tensor("queryPaddingSize", {1}, "1", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.antiquantScale = Tensor("antiquantScale", {2}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DataType::DT_BF16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 512, 128}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 8}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}