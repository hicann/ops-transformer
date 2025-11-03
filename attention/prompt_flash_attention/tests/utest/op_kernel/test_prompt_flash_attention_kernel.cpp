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
 * \file test_prompt_flash_attention_kernel.cpp
 * \brief IncreFlashAttention用例.
 */

#include "ts_pfa.h"
class Ts_Pfa_Ascend910B2_Case : public Ts_Pfa_WithParam_Ascend910B2 {};

TEST_P(Ts_Pfa_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Pfa_General_Case =
    ::testing::Values(PfaCase("case_001", true, "",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              PfaCase::Param(1, 4, 1024, 128, "BSH", 4, 4, 1.0f, 0, 1, 0, 524288, 0)));

INSTANTIATE_TEST_SUITE_P(Pfa, Ts_Pfa_Ascend910B2_Case, Tc_Pfa_General_Case);