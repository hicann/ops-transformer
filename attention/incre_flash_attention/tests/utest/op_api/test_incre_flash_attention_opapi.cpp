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
 * \file ts_ifa_tc.cpp
 * \brief IncreFlashAttention用例.
 */

#include "ts_ifa.h"
class Ts_Ifa_Ascend910B2_Case : public Ts_Ifa_WithParam_Ascend910B2 {};

TEST_F(Ts_Ifa_Ascend910B2, case_opapi_query)
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
