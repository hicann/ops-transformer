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
 * \brief FusedInferAttentionScore用例.
 */

#include "ts_fia.h"


TEST_F(Ts_Fia_Ascend910B1, case_SetL2CacheFlag_001)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_INT64;
    cs.mParam.kDataType = ge::DataType::DT_INT64;
    cs.mParam.vDataType = ge::DataType::DT_INT64;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_002)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_FLOAT16;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_003)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 0;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 0;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.b, cs.mParam.n, 1, cs.mParam.d}, "BNSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_004)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 1;
    cs.mParam.kvNumHeads = 2;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_005)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 5;
    cs.mParam.kvNumHeads = 2;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_006)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.h = 21;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_007)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSH_NBSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kvNumHeads = 1;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    cs.query = Tensor("query", {cs.mParam.b, 1, cs.mParam.n * cs.mParam.d + 1}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.b, cs.mParam.s, cs.mParam.kvNumHeads}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.b, cs.mParam.s, cs.mParam.kvNumHeads}, "BSH", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.b, 1, cs.mParam.d}, "NBSD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);

    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_008)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 512;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_009)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 21;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_010)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 127;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_INT4;
    cs.mParam.vDataType = ge::DataType::DT_INT4;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;

    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_011)
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.queryRope = Tensor("queryRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_012)
{
    FiaCase cs;
    cs.mParam.b = 32;
    cs.mParam.n = 32;
    cs.mParam.s = 256;
    cs.mParam.d = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.keyRope = Tensor("keyRope", {32, 32, 2, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_013)
{
    FiaCase cs;
    cs.mParam.t = 20;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_014)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 32;
    cs.mParam.s = 1;
    cs.mParam.d = 512;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.qs = 16;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_015)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_016)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.t, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_017)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mParam.qDataType = ge::DataType::DT_INT64;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    
    cs.queryRope = Tensor("queryRope", {1, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_018)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.blockSize = 128;
    cs.mParam.qDataType = ge::DataType::DT_INT64;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    
    cs.queryRope = Tensor("queryRope", {1, 128, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_019)
{
    FiaCase cs;
    cs.mParam.t = 1024 * 1024;
    cs.mParam.n = 64;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 64;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {1}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_020)
{
    FiaCase cs;
    cs.mParam.t = 1024 * 1024;
    cs.mParam.n = 64;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 64;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.t, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {1}, "B", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_021)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}



TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_022)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_023)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_024)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_025)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_026)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_027)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.t, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_028)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.t, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_029)
{
    FiaCase cs;
    cs.mParam.t = 2;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 2;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 128;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1, 2};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {1, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_030)
{
    FiaCase cs;
    cs.mParam.t = 2;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 2;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {2, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_031)
{
    FiaCase cs;
    cs.mParam.t = 1024 * 1024;
    cs.mParam.n = 64;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 64;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}



TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_032)
{
    FiaCase cs;
    cs.mParam.t = 1024 * 1024;
    cs.mParam.n = 64;
    cs.mParam.d = 512;
    cs.mParam.b = 1;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 64;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.n, cs.mParam.t, cs.mParam.d}, "NTD", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {cs.mParam.b, cs.mParam.n, cs.mParam.s, cs.mParam.d}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_033)
{
    FiaCase cs;
    cs.mParam.t = 4;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_034)
{
    FiaCase cs;
    cs.mParam.t = 4;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 20};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_035)
{
    FiaCase cs;
    cs.mParam.t = 4;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_036)
{
    FiaCase cs;
    cs.mParam.t = 4;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 20};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_037)
{
    FiaCase cs;
    cs.mParam.t = 4;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_QKVPreProcess_038)
{
    FiaCase cs;
    cs.mParam.t = 4;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND_NTD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_039)
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

    cs.mParam.innerPrecise = 2;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_040)
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

    cs.mParam.actualSeqLengthKV = {1};
    cs.mParam.kDataType = ge::DataType::DT_INT4;
    cs.mParam.vDataType = ge::DataType::DT_INT4;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {cs.mParam.b, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_041)
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

    cs.mParam.actualSeqLengthKV = {1};
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {0, 1}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InputAttrsPreProcess_042)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {4, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 512, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {2, 512, 32, 64}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPageAttentionFlag_043)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 128}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 128}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 512, 32, 64}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPageAttentionFlag_044)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "BSND";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {3, 512, 32, 64}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_KvShapePostProcess_045)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 32, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("key", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_KvShapePostProcess_046)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("key", {3, 32, 1}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("key", {3, 32, 1, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessPageAttentionFlag_047)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {3, 32, 128, 512}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_InitInOutMode_048)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.blockSize = 128;
    cs.mParam.numHeads = 32;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
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


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_050)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.blockSize = 32;
    cs.mParam.actualSeqLength = {};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {cs.mParam.t, cs.mParam.n, cs.mParam.d}, "TND", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_051)
{
    FiaCase cs;
    cs.mParam.t = 1;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.actualSeqLengths = Tensor("actualSeqLengths", {1, 0}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_052)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_053)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 512;
    cs.mParam.layout = "TND";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 2, 3};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 32, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_054)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.actualSeqLengths = Tensor("actualSeqLengths", {}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_055)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_056)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {-1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {3, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessActualSeqLen_057)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {129, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.queryRope = Tensor("queryRope", {3, 32, 1, 64}, "TND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {3, 1, 128, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant1_058)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.deqScale1 = Tensor("deqScale1", {1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant1_059)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale1 = Tensor("quantScale1", {1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant1_060)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.deqScale2 = Tensor("deqScale2", {1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryQuantParam4FullQuant1_061)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryQuantParam4FullQuant1_062)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryQuantParam4FullQuant1_063)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryQuantParam4FullQuant1_064)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryQuantParam4FullQuant1_065)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckQueryQuantParam4FullQuant1_066)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVQuantParam4FullQuant_067)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVQuantParam4FullQuant_068)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_069)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_070)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_071)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_072)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_073)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_074)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_075)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_076)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_077)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_078)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_079)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_080)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_081)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_082)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_083)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant_084)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_085)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_086)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_087)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "5", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_088)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "5", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_089)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT4;
    cs.mParam.vDataType = ge::DataType::DT_INT4;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_090)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_INT4;
    cs.mParam.vDataType = ge::DataType::DT_INT4;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_091)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2Dtype_092)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2_093)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {2, 2, 2, 2, 2}, "5", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessQuant2_094)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.qDataType = ge::DataType::DT_BF16;
    cs.mParam.kDataType = ge::DataType::DT_BF16;
    cs.mParam.vDataType = ge::DataType::DT_BF16;
    cs.mParam.outDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.quantScale2 = Tensor("quantScale2", {2, 2, 2, 2, 2}, "5", ge::DT_BF16, ge::FORMAT_ND);
    cs.quantOffset2 = Tensor("quantOffset2", {1, 2}, "2", ge::DT_BF16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParamsShapeInPagedAttention_095)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 5;
    cs.mParam.value_antiquant_mode = 5;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParamsShapeInPagedAttention_096)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 5;
    cs.mParam.value_antiquant_mode = 5;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParamsShapeInPagedAttention_097)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 5;
    cs.mParam.value_antiquant_mode = 5;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParamsShapeInPagedAttention_098)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 4;
    cs.mParam.value_antiquant_mode = 4;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParamsShapeInPagedAttention_099)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 4;
    cs.mParam.value_antiquant_mode = 4;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantMode_100)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 8;
    cs.mParam.value_antiquant_mode = 8;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerToken_101)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3, 128, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3, 128, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerToken_102)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 128, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 128, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerToken_103)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerToken_104)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 3}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 3}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerToken_105)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantPerToken_106)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {3}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {3}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParaShapeLegal_107)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckKVAntiQuantParaShapeLegal_108)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParamKeyType_109)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParamKeyType_110)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParamValueType_111)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_CheckAntiQuantParamValueType_112)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {128}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 1, 1, 1, 1}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {128}, "2", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_113)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_114)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantOffset = Tensor("antiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_115)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_116)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_117)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_118)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_119)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.blockSize = 128;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_120)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_121)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_122)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_123)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_124)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_125)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.qDataType = ge::DataType::DT_FLOAT;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_FLOAT;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_126)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.qDataType = ge::DataType::DT_FLOAT;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.outDataType = ge::DataType::DT_FLOAT;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_127)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantOffset = Tensor("keyAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantOffset = Tensor("valueAntiquantOffset", {2, 512}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keyRopeAntiquantScale = Tensor("keyRopeAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_128)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessAntiQuant_129)
{
    FiaCase cs;
    cs.mParam.t = 3;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.key_antiquant_mode = 4;
    cs.mParam.value_antiquant_mode = 4;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.antiquantScale = Tensor("antiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {2, 512}, "B", ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessBlockTable_130)
{
    FiaCase cs;
    cs.mParam.n = 128;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 128;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 128;
    cs.mParam.actualSeqLength = {1, 1, 1};
    cs.mParam.actualSeqLengthKV = {1, 1, 1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessBlockTable_131)
{
    FiaCase cs;
    cs.mParam.n = 32;
    cs.mParam.d = 512;
    cs.mParam.b = 3;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.s = 128;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 32;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {512, 512, 512};
    cs.mParam.actualSeqLengthKV = {512, 512, 512};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.blocktable = Tensor("blockTable", {32, 2}, "BNSD", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_132)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_133)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 2048, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 2048, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 0, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {3}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_134)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1, 1};
    cs.mParam.actualSharedPrefixLens = {-1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {4, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_135)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mParam.actualSeqLengthKV = {1, 1, 1, 1, 1, 1, 1};
    cs.mParam.actualSharedPrefixLens = {2};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {4, 2048}, "BS", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_136)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.pseShift = Tensor("pseShift", {4, 10, 1, 1024}, "4", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_137)
{
    FiaCase cs;
    cs.mParam.b = 4;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 1;
    cs.mParam.layout = "BSH";
    cs.mParam.numHeads = 10;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.query = Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {4, 1024, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10}, "3", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attenMask = Tensor("attenMask", {4, 1024}, "BS", ge::DT_INT8, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ProcessSharedPrefixLen_138)
{
    FiaCase cs;
    cs.mParam.b = 3;
    cs.mParam.n = 32;
    cs.mParam.s = 128;
    cs.mParam.d = 512;
    cs.mParam.kvNumHeads = 1;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 32;
    cs.mParam.key_antiquant_mode = 1;
    cs.mParam.value_antiquant_mode = 1;
    cs.mParam.kDataType = ge::DataType::DT_INT8;
    cs.mParam.vDataType = ge::DataType::DT_INT8;
    cs.mParam.actualSeqLengthKV = {128, 128, 128};
    cs.mParam.actualSharedPrefixLens = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.keyAntiquantScale = Tensor("keyAntiquantScale", {1, 3, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.valueAntiquantScale = Tensor("valueAntiquantScale", {1, 3, 128}, "B", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    cs.keySharedPrefix = Tensor("keySharedPrefix", {1, 1, 10, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.valueSharedPrefix = Tensor("valueSharedPrefix", {1, 1, 10, 512}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
    cs.actualSharedPrefixLen = Tensor("actualSharedPrefixLen", {1}, "1", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_GetBmm1Tiling_139)
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


TEST_F(Ts_Fia_Ascend910B1, case_GetBmm1Tiling_140)
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


TEST_F(Ts_Fia_Ascend910B1, case_GetBmm1Tiling_141)
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


TEST_F(Ts_Fia_Ascend910B1, case_GetBmm1Tiling_142)
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


TEST_F(Ts_Fia_Ascend910B1, case_GetBmm2Tiling_143)
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


TEST_F(Ts_Fia_Ascend910B1, case_GetBmm2Tiling_144)
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


TEST_F(Ts_Fia_Ascend910B1, case_FillTilingBmm_145)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_INT16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_FillTilingBmm_146)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_INT16;
    cs.mParam.vDataType = ge::DataType::DT_INT16;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());

    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_GenTilingKey_147)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 1;
    cs.mParam.blockSize = 128;
    cs.mParam.actualSeqLength = {1};
    cs.mParam.actualSeqLengthKV = {1};
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.query = Tensor("query", {1, 1, 1, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.key = TensorList("key", {1, 128, 1, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.value = TensorList("value", {1, 128, 1, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.attentionOut = Tensor("attentionOut", {1, 1, 1, 512}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.queryRope = Tensor("queryRope", {1, 1, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.keyRope = Tensor("keyRope", {1, 128, 1, 64}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
    cs.blocktable = Tensor("blockTable", {1, 2}, "BS", ge::DT_INT32, ge::FORMAT_ND);
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_GenTilingKey_148)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BSND";
    cs.mParam.numHeads = 20;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_GenTilingKey_149)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.qDataType = ge::DataType::DT_FLOAT;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_GenTilingKey_150)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.kDataType = ge::DataType::DT_FLOAT;
    cs.mParam.vDataType = ge::DataType::DT_FLOAT;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_GenTilingKey_151)
{
    FiaCase cs;
    cs.mParam.b = 1;
    cs.mParam.n = 20;
    cs.mParam.s = 1024;
    cs.mParam.d = 128;
    cs.mParam.layout = "BNSD";
    cs.mParam.numHeads = 20;
    cs.mParam.outDataType = ge::DataType::DT_FLOAT;
    cs.mOpInfo.mExp.mTilingKey = 11000000000100000; // expected tiling key
    cs.mOpInfo.mExp.mTilingBlockDim = 24;           // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = false;
    ASSERT_TRUE(cs.Init());
    cs.mOpInfo.mExp.mSuccess = false;
    ASSERT_EQ(cs.Run(), cs.mOpInfo.mExp.mSuccess);
}


TEST_F(Ts_Fia_Ascend910B1, case_ConvertContext_152)
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


TEST_F(Ts_Fia_Ascend910B1, case_ConvertContext_153)
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


TEST_F(Ts_Fia_Ascend910B1, case_ConvertContext_154)
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


TEST_F(Ts_Fia_Ascend910B1, case_DataTypeToSerialString_155)
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


TEST_F(Ts_Fia_Ascend910B1, case_DataTypeToSerialString_156)
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