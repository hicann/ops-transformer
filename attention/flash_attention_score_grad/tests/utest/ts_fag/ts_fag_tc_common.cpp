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
 * \file ts_fag_tc_common.cpp
 * \brief FlashAttentionScoreGrad 反向用例.
 */

#include "ts_fag.h"

TEST_F(Ts_Fag_Ascend910B3, Tc_Common_InvalidShape_001)
{
    /**
     * 用例信息
     */
    // 用例 Shape 和 Attrs 信息
    case_->mParam.b = 2;
    case_->mParam.n2 = 16;
    case_->mParam.g = 1;
    case_->mParam.s1 = 2048;
    case_->mParam.s2 = 2048;
    case_->mParam.d = 32;
    case_->mParam.dtype = ge::DataType::DT_FLOAT16;
    case_->mParam.layoutType = LayoutType::BSH;
    case_->mParam.scale = 1.0f;
    case_->mParam.keepProb = 0.9f;
    case_->mParam.preTokens = 65536;
    case_->mParam.nxtTokens = 65536;
    case_->mParam.innerPrecise = 0;
    case_->mParam.sparseMode = 0;
    case_->mParam.pseShapeType = PseShapeType::NONE;
    case_->mParam.dropMaskShapeType = DropMaskShapeType::NONE;
    case_->mParam.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    case_->mParam.attenMaskShapeType = AttenMaskShapeType::NONE;
    case_->mParam.attenMaskDtype = ge::DataType::DT_BOOL;
    case_->mParam.prefixShapeType = PrefixShapeType::NONE;

    // 用例 期望信息
    case_->mReverse.mExp.mSuccess = false;

    // 用例 信息初始化及修正
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenRes = Tensor("atten_res", {case_->mParam.b + 1, case_->mParam.s1, case_->mParam.h1},
                                    case_->mParam.layout.c_str(), case_->mParam.dtype, ge::FORMAT_ND);

    // 用例 执行
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}
