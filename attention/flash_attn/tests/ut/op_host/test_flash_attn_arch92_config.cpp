/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>
#include "../../../op_host/arch92/fa_adjust_sinner_souter_arch92.h"
#include "../../../op_host/flash_attn_kernel_config.h"
#include "../../../op_kernel/utils/flash_attn_common_def.h"

namespace FlashAttnUT {
namespace util = optiling::flash_attn::fa_tiling_util;

TEST(FlashAttnArch92D256, HostAndMetadataSelectSameTile)
{
    for (int32_t mask : {0, 3, 4}) {
        const uint32_t expectedConfig = (mask == 4) ? 5U : 4U;
        for (int32_t maxQ : {-1, 1, 95, 96, 97, 8192}) {
            uint32_t templateId = 0;
            uint32_t m = 0;
            uint32_t n = 0;
            util::GetConfigInfo(256, mask, maxQ, templateId, m, n);
            EXPECT_EQ(templateId, static_cast<uint32_t>(FA_Template_ND));
            EXPECT_EQ(m, static_cast<uint32_t>(ConfigValueArch92[expectedConfig].s1));
            EXPECT_EQ(n, static_cast<uint32_t>(ConfigValueArch92[expectedConfig].s2));
        }
    }
}

TEST(FlashAttnArch92D256, UsedCoreCountIncludesM96AndN256Tails)
{
    EXPECT_EQ(util::GetMaxUsedAicCores(64, 1, 1, 1, 96, 256, 96, 256), 1U);
    EXPECT_EQ(util::GetMaxUsedAicCores(64, 1, 1, 1, 97, 256, 96, 256), 2U);
    EXPECT_EQ(util::GetMaxUsedAicCores(64, 1, 1, 1, 96, 257, 96, 256), 2U);
    EXPECT_EQ(util::GetMaxUsedAicCores(64, 1, 1, 1, 97, 257, 96, 256), 4U);
}

TEST(FlashAttnArch92D256, SharedKvAndSingleUbResultsFitBudgets)
{
    const auto& cfg = ConfigValueArch92[4];
    const uint32_t m = static_cast<uint32_t>(cfg.s1);
    const uint32_t n = static_cast<uint32_t>(cfg.s2);
    const uint32_t d = static_cast<uint32_t>(cfg.d);
    const uint32_t statRows = (m + 63U) / 64U * 64U;
    const uint32_t l1Bytes = (cfg.l1PBufcnt * m * n + cfg.l1QBufcnt * m * d + cfg.l1KvBufcnt * n * d) * 2;
    const uint32_t ubBytes = (cfg.ubMm1ResBufcnt * m * n + cfg.ubMm2ResBufcnt * m * d + m * d) * sizeof(float) +
                             cfg.ubVec1ResBufcnt * (m + 1U) * n * sizeof(uint16_t) + cfg.ubMaskBufcnt * m * n +
                             9U * statRows * sizeof(float) + 2U * statRows * 8U * sizeof(float) +
                             2U * statRows * sizeof(float);

    EXPECT_EQ(m, 96U);
    EXPECT_EQ(statRows, 128U);
    EXPECT_EQ(l1Bytes, 496U * 1024);
    EXPECT_EQ(cfg.ubMm1ResBufcnt, 1U);
    EXPECT_EQ(cfg.ubMm2ResBufcnt, 1U);
    EXPECT_EQ(cfg.ubVec1ResBufcnt, 1U);
    EXPECT_EQ(cfg.ubMaskBufcnt, 1U);
    EXPECT_EQ(cfg.ubMaskBufcnt * m * n, 24U * 1024);
    EXPECT_EQ(ubBytes, 374U * 1024);
    EXPECT_LE(ubBytes, 376U * 1024);
    EXPECT_EQ(m * 256 * 4 / 2, 48U * 1024); // Each N-half fits one 64 KiB L0C.
}

TEST(FlashAttnArch92D256, OtherDimensionsAndLegacyConfigStayUnchanged)
{
    uint32_t templateId = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    util::GetConfigInfo(64, 0, 8192, templateId, m, n);
    EXPECT_EQ(templateId, static_cast<uint32_t>(FA_Template_ND));
    EXPECT_EQ(m, 64U);
    EXPECT_EQ(n, 256U);
    util::GetConfigInfo(128, 0, 8192, templateId, m, n);
    EXPECT_EQ(templateId, static_cast<uint32_t>(FA_Template_DN_Unmerged));
    EXPECT_EQ(m, 128U);
    EXPECT_EQ(n, 256U);
    util::GetConfigInfo(128, 4, 8192, templateId, m, n);
    EXPECT_EQ(templateId, static_cast<uint32_t>(FA_Template_ND));
    EXPECT_EQ(m, 128U);
    EXPECT_EQ(n, 128U);
    EXPECT_EQ(static_cast<uint32_t>(ConfigValueArch92[4].s1), 96U);
    EXPECT_EQ(static_cast<uint32_t>(ConfigValueArch92[5].s1), 64U);
    EXPECT_EQ(static_cast<uint32_t>(ConfigValue[4].s1), 128U);
}
} // namespace FlashAttnUT
