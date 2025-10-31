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
 * \brief FusedInferAttentionScore用例_新UT框架.
 */

#include "ts_fia.h"

using CaseMode = ops::adv::tests::fia::CaseMode;
using CaseKvStorageMode = ops::adv::tests::fia::CaseKvStorageMode;

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


TEST_F(Ts_Fia_Ascend910B1, case_CheckBlockTableExistence_mla_052)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::BATCH_CONTINUOUS;

    ASSERT_TRUE(cs.Init());
    cs.blocktable = Tensor();
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

TEST_F(Ts_Fia_Ascend910B1, case_GetActualSeqLenSize_mla_008)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {0}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
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

TEST_F(Ts_Fia_Ascend910B1, case_CheckAttentionMask_mla_044)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.sparse_mode = 1;
    ASSERT_TRUE(cs.Init());

    cs.attenMask = Tensor("attenMask", {}, "B", ge::DT_INT8, ge::FORMAT_ND);
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

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantShape_mla_055)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.blockSize = 64;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantLayout_mla_058)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;

    ASSERT_TRUE(cs.Init());
    cs.key = TensorList("key", {64, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {64, 1, 32, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {32, 32, 1, 64}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {64, 1, 4, 128, 16}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoquantMask_mla_064)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.sparse_mode = 0;
    cs.mParam.actualSeqLength = {3, 3, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.query = Tensor("query", {3, 32, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 128, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 32, 512}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 7}, "TND", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {3}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {3, 896}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoquantMask_mla_066)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.query = Tensor("query", {3, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {3, 1, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {3, 32, 2, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {2, 384}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
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

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoquantUnsupported_mla_075)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.innerPrecise = 3;
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

TEST_F(Ts_Fia_Ascend910B1, case_CheckFeatureMlaNoQuantShape_mla_081)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.b = 65537;
    ASSERT_TRUE(cs.Init());
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


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_mla_039)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BNSD";
    cs.mParam.innerPrecise = 2;
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


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPageAttentionFlag_mla_044)
{
    FiaCase cs;
    cs.mParam.mode = CaseMode::MLA_NOQUANT;
    cs.mParam.storageMode = CaseKvStorageMode::PAGE_ATTENTION;
    cs.mParam.layout = "BSND";
    ASSERT_TRUE(cs.Init());

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
    cs.mParam.b = 1024;
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
    cs.mParam.b = 1024;
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