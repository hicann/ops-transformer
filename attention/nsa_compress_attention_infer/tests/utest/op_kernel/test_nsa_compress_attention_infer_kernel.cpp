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
 * \file ts_nsag_param_tc.cpp
 * \brief nsa compress attention infer 用例.
 */

#include "ts_ncai.h"

using Tensor = ops::adv::tests::utils::Tensor;

TEST_F(Ts_ncaInfer, case_nsa_compress_attention_infer_0_kernel)
{
    constexpr uint32_t CASE1_B = 20;
    constexpr uint32_t CASE1_S2 = 4096;
    constexpr uint32_t CASE1_NUMHEADS = 64;
    constexpr uint32_t CASE1_NUMKVHEADS = 4;
    constexpr uint32_t CASE1_SELECTBLOCKSIZE = 64;
    constexpr uint32_t CASE1_SELECTBLOCKCOUNT = 16;
    constexpr uint32_t CASE1_COMPRESSBLOCKSIZE = 32;
    constexpr uint32_t CASE1_COMPRESSSTRIDE = 16;
    constexpr uint32_t CASE1_PAGEBLOCKSIZE = 128;
    constexpr uint32_t CASE1_MTILINGBLOCKDIM = 24;
    constexpr uint32_t CASE1_HEADSIZE_QK = 192;

    ncaInferCase cs;
    /* 设置参数 */
    cs.mParam.B = CASE1_B;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = CASE1_S2;
    cs.mParam.N1 = CASE1_NUMHEADS;
    cs.mParam.G = CASE1_NUMHEADS / CASE1_NUMKVHEADS;
    cs.mParam.D = CASE1_HEADSIZE_QK;
    cs.mParam.N2 = CASE1_NUMKVHEADS;
    
    // attr
    cs.mParam.numHeads = CASE1_NUMHEADS;
    cs.mParam.numKeyValueHeads = CASE1_NUMKVHEADS;
    cs.mParam.selectBlockSize = CASE1_SELECTBLOCKSIZE;
    cs.mParam.selectBlockCount = CASE1_SELECTBLOCKCOUNT;
    cs.mParam.compressBlockSize = CASE1_COMPRESSBLOCKSIZE;
    cs.mParam.compressStride = CASE1_COMPRESSSTRIDE;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.layoutOptional = "TND";
    cs.mParam.pageBlockSize = CASE1_PAGEBLOCKSIZE;
    cs.mParam.sparseMode = 0;

    cs.mOpInfo.mExp.mTilingBlockDim = CASE1_MTILINGBLOCKDIM; // expected block dim
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_ncaInfer, case_nsa_compress_attention_infer_1_kernel)
{
    constexpr uint32_t CASE2_B = 20;
    constexpr uint32_t CASE2_S2 = 2048;
    constexpr uint32_t CASE2_NUMHEADS = 64;
    constexpr uint32_t CASE2_NUMKVHEADS = 4;
    constexpr uint32_t CASE2_SELECTBLOCKSIZE = 64;
    constexpr uint32_t CASE2_SELECTBLOCKCOUNT = 16;
    constexpr uint32_t CASE2_COMPRESSBLOCKSIZE = 32;
    constexpr uint32_t CASE2_COMPRESSSTRIDE = 16;
    constexpr uint32_t CASE2_PAGEBLOCKSIZE = 128;
    constexpr uint32_t CASE2_MTILINGBLOCKDIM = 24;
    constexpr uint32_t CASE2_HEADSIZE_QK = 192;

    ncaInferCase cs;
    /* 设置参数 */
    cs.mParam.B = CASE2_B;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = CASE2_S2;
    cs.mParam.N1 = CASE2_NUMHEADS;
    cs.mParam.G = CASE2_NUMHEADS / CASE2_NUMKVHEADS;
    cs.mParam.D = CASE2_HEADSIZE_QK;
    cs.mParam.N2 = CASE2_NUMKVHEADS;
    
    // attr
    cs.mParam.numHeads = CASE2_NUMHEADS;
    cs.mParam.numKeyValueHeads = CASE2_NUMKVHEADS;
    cs.mParam.selectBlockSize = CASE2_SELECTBLOCKSIZE;
    cs.mParam.selectBlockCount = CASE2_SELECTBLOCKCOUNT;
    cs.mParam.compressBlockSize = CASE2_COMPRESSBLOCKSIZE;
    cs.mParam.compressStride = CASE2_COMPRESSSTRIDE;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.layoutOptional = "BSND";
    cs.mParam.pageBlockSize = CASE2_PAGEBLOCKSIZE;
    cs.mParam.sparseMode = 0;

    cs.mOpInfo.mExp.mTilingBlockDim = CASE2_MTILINGBLOCKDIM;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

// getmaxworkspace test
TEST_F(Ts_ncaInfer, case_nsa_compress_attention_infer_aclgraph_kernel)
{
    constexpr uint32_t CASE3_B = 20;
    constexpr int32_t CASE3_S2 = -1;
    constexpr uint32_t CASE3_NUMHEADS = 64;
    constexpr uint32_t CASE3_NUMKVHEADS = 4;
    constexpr uint32_t CASE3_SELECTBLOCKSIZE = 64;
    constexpr uint32_t CASE3_SELECTBLOCKCOUNT = 16;
    constexpr uint32_t CASE3_COMPRESSBLOCKSIZE = 32;
    constexpr uint32_t CASE3_COMPRESSSTRIDE = 16;
    constexpr uint32_t CASE3_PAGEBLOCKSIZE = 128;
    constexpr uint32_t CASE3_MTILINGBLOCKDIM = 24;
    constexpr uint32_t CASE3_HEADSIZE_QK = 192;

    ncaInferCase cs;
    /* 设置参数 */
    cs.mParam.B = CASE3_B;
    cs.mParam.S1 = 1;
    cs.mParam.S2 = CASE3_S2;
    cs.mParam.N1 = CASE3_NUMHEADS;
    cs.mParam.G = CASE3_NUMHEADS / CASE3_NUMKVHEADS;
    cs.mParam.D = CASE3_HEADSIZE_QK;
    cs.mParam.N2 = CASE3_NUMKVHEADS;
    
    // attr
    cs.mParam.numHeads = CASE3_NUMHEADS;
    cs.mParam.numKeyValueHeads = CASE3_NUMKVHEADS;
    cs.mParam.selectBlockSize = CASE3_SELECTBLOCKSIZE;
    cs.mParam.selectBlockCount = CASE3_SELECTBLOCKCOUNT;
    cs.mParam.compressBlockSize = CASE3_COMPRESSBLOCKSIZE;
    cs.mParam.compressStride = CASE3_COMPRESSSTRIDE;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.layoutOptional = "TND";
    cs.mParam.pageBlockSize = CASE3_PAGEBLOCKSIZE;
    cs.mParam.sparseMode = 0;

    cs.mOpInfo.mExp.mTilingBlockDim = CASE3_MTILINGBLOCKDIM;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_ncaInfer, case_nsa_compress_attention_infer_mtp_kernel)
{
    constexpr uint32_t CASE4_B = 20;
    constexpr uint32_t CASE4_S1 = 2;
    constexpr uint32_t CASE4_S2 = 4096;
    constexpr uint32_t CASE4_NUMHEADS = 64;
    constexpr uint32_t CASE4_NUMKVHEADS = 4;
    constexpr uint32_t CASE4_SELECTBLOCKSIZE = 64;
    constexpr uint32_t CASE4_SELECTBLOCKCOUNT = 16;
    constexpr uint32_t CASE4_COMPRESSBLOCKSIZE = 32;
    constexpr uint32_t CASE4_COMPRESSSTRIDE = 16;
    constexpr uint32_t CASE4_PAGEBLOCKSIZE = 128;
    constexpr uint32_t CASE4_MTILINGBLOCKDIM = 24;
    constexpr uint32_t CASE4_HEADSIZE_QK = 192;

    ncaInferCase cs;
    /* 设置参数 */
    cs.mParam.B = CASE4_B;
    cs.mParam.S1 = CASE4_S1;
    cs.mParam.S2 = CASE4_S2;
    cs.mParam.N1 = CASE4_NUMHEADS;
    cs.mParam.G = CASE4_NUMHEADS / CASE4_NUMKVHEADS;
    cs.mParam.D = CASE4_HEADSIZE_QK;
    cs.mParam.N2 = CASE4_NUMKVHEADS;
    
    // attr
    cs.mParam.numHeads = CASE4_NUMHEADS;
    cs.mParam.numKeyValueHeads = CASE4_NUMKVHEADS;
    cs.mParam.selectBlockSize = CASE4_SELECTBLOCKSIZE;
    cs.mParam.selectBlockCount = CASE4_SELECTBLOCKCOUNT;
    cs.mParam.compressBlockSize = CASE4_COMPRESSBLOCKSIZE;
    cs.mParam.compressStride = CASE4_COMPRESSSTRIDE;
    cs.mParam.scaleValue = 1.0;
    cs.mParam.layoutOptional = "TND";
    cs.mParam.pageBlockSize = CASE4_PAGEBLOCKSIZE;
    cs.mParam.sparseMode = 0;

    cs.mOpInfo.mExp.mTilingBlockDim = CASE4_MTILINGBLOCKDIM;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
}