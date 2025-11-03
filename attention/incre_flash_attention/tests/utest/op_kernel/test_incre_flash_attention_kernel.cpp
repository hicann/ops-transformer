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
 * \file test_incre_flash_attention_kernel.cpp
 * \brief IncreFlashAttention用例.
 */

#include "ts_ifa.h"
class Ts_Ifa_Ascend910B2_Case : public Ts_Ifa_WithParam_Ascend910B2 {};

TEST_P(Ts_Ifa_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Ifa_General_Case = ::testing::Values(
    IfaCase("case_000", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 1, 1024, 128, "BSH", 1, 1, 1.0f, 0, 1, {})),
    IfaCase("case_001", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 4, 1024, 128, "BSH", 4, 4, 1.0f, 0, 1, {})),
    IfaCase("case_002", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 5, 128, 128, "BSH", 5, 5, 1.0f, 0, 1, {})),
    IfaCase("case_003", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 40, 128, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_004", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(13, 20, 2048, 128, "BSH", 20, 20, 1.0f, 0, 1, {})),
    IfaCase("case_005", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(20, 40, 2048, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_006", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(20, 40, 2048, 128, "BSH", 40, 40, 1.0f, 0, 1,
                           {1024, 512,  2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
                            2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048})),
    IfaCase("case_007", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 40, 128, 128, "BNSD", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_008", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 40, 1, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_009", true, "dbginfo",
            OpInfo(ControlInfo(true, false),
                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(2, 40, 4096, 128, "BSND", 40, 40, 1.0f, 0, 1, {}))
    );

INSTANTIATE_TEST_SUITE_P(Ifa, Ts_Ifa_Ascend910B2_Case, Tc_Ifa_General_Case);
