#include "ts_ifa.h"
class Ts_Ifa_Ascend910B2_Case : public Ts_Ifa_WithParam_Ascend910B2 {};
class Ts_Ifa_Ascend310P3_Case : public Ts_Ifa_WithParam_Ascend310P3 {};
class Ts_Ifa_Ascend910_9591_Case : public Ts_Ifa_WithParam_Ascend910_9591 {};

TEST_P(Ts_Ifa_Ascend910B2_Case, general_case)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Ifa_General_Case = ::testing::Values(
    IfaCase("case_000", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000100001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 1, 1024, 128, "BSH", 1, 1, 1.0f, 0, 1, {})),
    IfaCase("case_001", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000100001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 4, 1024, 128, "BSH", 4, 4, 1.0f, 0, 1, {})),
    IfaCase("case_002", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000100001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 5, 128, 128, "BSH", 5, 5, 1.0f, 0, 1, {})),
    IfaCase("case_003", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 40, 128, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_004", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(13, 20, 2048, 128, "BSH", 20, 20, 1.0f, 0, 1, {})),
    IfaCase("case_005", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(20, 40, 2048, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_006", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(20, 40, 2048, 128, "BSH", 40, 40, 1.0f, 0, 1,
                           {1024, 512,  2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
                            2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048})),
    IfaCase("case_007", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000000, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 40, 128, 128, "BNSD", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_008", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(1, 40, 1, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
    IfaCase("case_009", true, "dbginfo",
            OpInfo(ControlInfo(true, true),
                   ExpectInfo(true, 11000000000000001, ExpectInfo::kInvalidTilingBlockDim)),
            IfaCase::Param(2, 40, 4096, 128, "BSND", 40, 40, 1.0f, 0, 1, {}))
    );

INSTANTIATE_TEST_SUITE_P(Ifa, Ts_Ifa_Ascend910B2_Case, Tc_Ifa_General_Case);
