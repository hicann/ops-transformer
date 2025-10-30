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
 * \file ts_nsa_compress_attention.cpp
 * \brief NsaCompressAttention用例.
 */

#include "ts_nsa_compress_attention.h"

using LayoutType = ops::adv::tests::NsaCompressAttention::NsaCompressAttentionCase::LayoutType;
using AttenMaskShapeType = ops::adv::tests::NsaCompressAttention::NsaCompressAttentionCase::AttenMaskShapeType;
using TopkMaskShapeType = ops::adv::tests::NsaCompressAttention::NsaCompressAttentionCase::TopkMaskShapeType;

TEST_F(Ts_NsaCompressAttention, nsa_compress_attention_normal_case)
{
    NsaCompressAttentionCase cs;
    cs.mParam = {1, 4, 16, 65536, 4096, 192, 128, ge::DT_FLOAT16, LayoutType::TND, 1.0f, 1, 32, 16, 64, 16,
                AttenMaskShapeType::S1_S2, TopkMaskShapeType::S1_S2, ge::DT_BOOL, ge::DT_BOOL, {65536}, {4096}, {1024}};
    cs.mOpInfo.mExp.mTilingKey = 10000000000000001124UL;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    cs.Init();
    ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_NsaCompressAttention, nsa_compress_attention_normal_case_1)
{
    NsaCompressAttentionCase cs;
    cs.mParam = {1, 4, 1, 65536, 4096, 192, 128, ge::DT_FLOAT16, LayoutType::TND, 1.0f, 1, 32, 16, 64, 16,
                AttenMaskShapeType::S1_S2, TopkMaskShapeType::S1_S2, ge::DT_BOOL, ge::DT_BOOL, {65536}, {4096}, {1024}};
    cs.mOpInfo.mExp.mTilingKey = 10000000000000001124UL;
    cs.mOpInfo.mCtr.mRunTiling = true;
    cs.mOpInfo.mCtr.mRunKernel = true;
    cs.Init();
    ASSERT_TRUE(cs.Run());
}
