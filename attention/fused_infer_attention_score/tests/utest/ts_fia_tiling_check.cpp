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
 * \file ts_fia_tiling_check.cpp
 * \brief FusedInferAttentionScore用例.
 */

#include "ts_fia.h"


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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
    cs.mOpInfo.mExp.mTilingKey = 283948879052800; // expected tiling key
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
