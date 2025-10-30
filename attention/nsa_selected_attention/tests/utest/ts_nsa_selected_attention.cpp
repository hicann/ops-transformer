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
 * \file ts_nsa_selected_attention.cpp
 * \brief NsaSelectedAttention UTest 用例.
 */

#include <vector>
#include "ts_nsa_selected_attention.h"

/* test cases basic function */
NsaSelectedAttentionCase InitCase(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype,
                                std::string pLayout, float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize,
                                int64_t pSelectedBlockCount, std::vector<int64_t> pActualSeqQLenList,
                                std::vector<int64_t> pActualSeqKVLenList, ge::graphStatus expResult, int64_t tilingKey)
{
    NsaSelectedAttentionCase cs;
    cs.mParam = {pB,
                 pN2,
                 pG,
                 pD,
                 pD2,
                 pDtype,
                 pLayout,
                 pScale,
                 pSparseMode,
                 pSelectedBlockSize,
                 pSelectedBlockCount,
                 pActualSeqQLenList,
                 pActualSeqKVLenList};
    if (expResult == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = true;
    } else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

NsaSelectedAttentionCase InitCase(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype,
                                std::string pLayout, float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize,
                                int64_t pSelectedBlockCount, std::vector<int64_t> pActualSeqQLenList,
                                std::vector<int64_t> pActualSeqKVLenList, ge::graphStatus expResult, int64_t tilingKey, bool needAttenMask)
{
    NsaSelectedAttentionCase cs;
    cs.mParam = {pB,
                 pN2,
                 pG,
                 pD,
                 pD2,
                 pDtype,
                 pLayout,
                 pScale,
                 pSparseMode,
                 pSelectedBlockSize,
                 pSelectedBlockCount,
                 pActualSeqQLenList,
                 pActualSeqKVLenList,
                 needAttenMask};
    if (expResult == ge::GRAPH_SUCCESS) {
        cs.mOpInfo.mExp.mTilingKey = tilingKey;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = true;
    } else {
        cs.mOpInfo.mExp.mSuccess = false;
        cs.mOpInfo.mCtr.mRunTiling = true;
        cs.mOpInfo.mCtr.mRunKernel = false;
    }
    cs.Init();
    return cs;
}

void InitAndRunCase(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype,
                    std::string pLayout, float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize,
                    int64_t pSelectedBlockCount, std::vector<int64_t> pActualSeqQLenList,
                    std::vector<int64_t> pActualSeqKVLenList, ge::graphStatus expResult, int64_t tilingKey)
{
    NsaSelectedAttentionCase cs =
        InitCase(pB, pN2, pG, pD, pD2, pDtype, pLayout, pScale, pSparseMode, pSelectedBlockSize, pSelectedBlockCount,
                 pActualSeqQLenList, pActualSeqKVLenList, expResult, tilingKey);
    if (expResult == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

void InitAndRunCaseWithoutAttenMask(int64_t pB, int64_t pN2, int64_t pG, int64_t pD, int64_t pD2, ge::DataType pDtype,
                    std::string pLayout, float pScale, int64_t pSparseMode, int64_t pSelectedBlockSize,
                    int64_t pSelectedBlockCount, std::vector<int64_t> pActualSeqQLenList,
                    std::vector<int64_t> pActualSeqKVLenList, ge::graphStatus expResult, int64_t tilingKey)
{
    NsaSelectedAttentionCase cs =
        InitCase(pB, pN2, pG, pD, pD2, pDtype, pLayout, pScale, pSparseMode, pSelectedBlockSize, pSelectedBlockCount,
                 pActualSeqQLenList, pActualSeqKVLenList, expResult, tilingKey, false);
    if (expResult == ge::GRAPH_SUCCESS) {
        ASSERT_TRUE(cs.Run());
    } else {
        ASSERT_FALSE(cs.Run());
    }
}

/* test cases */

TEST_F(Ts_NsaSelectedAttention, case_fp16_case0)
{
    InitAndRunCase(4, 4, 4, 192, 128,             /* B, N2, G, D, D2 */
                   ge::DT_FLOAT16, "TND", 1.0, 2, /* dtype, layout, scale, sparseMode */
                   64, 16,                        /* selectedBlockSize, selectedBlockCount */
                   {256, 256, 256, 256},          /* ActualSeqQLenList */
                   {256, 256, 256, 256},          /* ActualSeqKvLenList */
                   ge::GRAPH_SUCCESS,             /* expResult*/
                   1                             /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_bf16_case1)
{
    InitAndRunCase(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "TND", 1.0, 2, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_SUCCESS,          /* expResult*/
                   1                          /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_fp16_case2)
{
    InitAndRunCase(4, 4, 4, 192, 128,             /* B, N2, G, D, D2 */
                   ge::DT_FLOAT16, "TND", 1.0, 0, /* dtype, layout, scale, sparseMode */
                   64, 16,                        /* selectedBlockSize, selectedBlockCount */
                   {256, 256, 256, 256},          /* ActualSeqQLenList */
                   {256, 256, 256, 256},          /* ActualSeqKvLenList */
                   ge::GRAPH_SUCCESS,             /* expResult*/
                   0                             /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_bf16_case3)
{
    InitAndRunCase(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "TND", 1.0, 0, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_SUCCESS,          /* expResult*/
                   0                          /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_BNSD_fail)
{
    InitAndRunCase(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "BNSD", 1.0, 0, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_FAILED,          /* expResult*/
                   0                          /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_BSND_fail)
{
    InitAndRunCase(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "BSND", 1.0, 0, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_FAILED,          /* expResult*/
                   0                          /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_SBH_fail)
{
    InitAndRunCase(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "SBH", 1.0, 0, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_FAILED,          /* expResult*/
                   0                          /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_INVALID_fail)
{
    InitAndRunCase(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "INVALID", 1.0, 0, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_FAILED,          /* expResult*/
                   0                          /* tilingKey */
    );
}

TEST_F(Ts_NsaSelectedAttention, case_AttenMask_nullptr)
{
    InitAndRunCaseWithoutAttenMask(2, 8, 4, 192, 128,          /* B, N2, G, D, D2 */
                   ge::DT_BF16, "TND", 1.0, 2, /* dtype, layout, scale, sparseMode */
                   64, 16,                     /* selectedBlockSize, selectedBlockCount */
                   {512, 512},                 /* ActualSeqQLenList */
                   {512, 512},                 /* ActualSeqKvLenList */
                   ge::GRAPH_SUCCESS,          /* expResult*/
                   0                          /* tilingKey */
    );
}