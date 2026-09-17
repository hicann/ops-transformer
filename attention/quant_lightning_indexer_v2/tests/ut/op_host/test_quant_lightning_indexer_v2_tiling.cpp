/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "test_quant_lightning_indexer_v2_utils.h"

// 顶层（soc 无关）用例：executor 的平台信息全 mock，可在任意 soc 构建中执行。
// TilingForQuantLightningIndexerV2 在 DAV_3510 下走 arch35 checker 路径（checkers + tiling_info_parser），
// 本用例以 socVersion="Ascend950" 驱动该路径，保证 arch22 构建的覆盖率报告中
// quant_lightning_indexer_v2 的 checker 体系与 tiling_info_parser 同样有覆盖。
class QuantLightningIndexerV2TilingCommon : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingCommon SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantLightningIndexerV2TilingCommon TearDown" << std::endl;
    }
};

namespace {
// 与 arch35 测试中的 Make950TndPaFp8 参数一致
qliv2_ut::CaseParam MakeMocked950TndPaFp8()
{
    qliv2_ut::CaseParam p;
    p.soc = "Ascend950";
    p.coreNum = 56;
    p.qShape = {78, 64, 128};
    p.kShape = {2, 16, 1, 128};
    p.wShape = {78, 64};
    p.qScaleShape = {78, 64};
    p.kScaleShape = {2, 16, 1};
    p.outShape = {78, 1, 2048};
    p.qType = ge::DT_FLOAT8_E4M3FN;
    p.kType = ge::DT_FLOAT8_E4M3FN;
    p.wType = ge::DT_FLOAT;
    p.qScaleType = ge::DT_FLOAT;
    p.kScaleType = ge::DT_FLOAT;
    p.cuSeqQ = {3};
    p.layoutQ = "TND";
    p.layoutK = "PA_BBND";
    p.quantMode = 1;
    p.maxSeqlenQ = 64;
    p.maskMode = 3;
    return p;
}
} // namespace

// mocked Ascend950 platform, TND/PA_BBND fp8 success: quant_mode=1
TEST_F(QuantLightningIndexerV2TilingCommon, QuantLightningIndexerV2_mocked_950_tiling_fp8_tnd_pa_success)
{
    qliv2_ut::RunTilingCase(MakeMocked950TndPaFp8(), ge::GRAPH_SUCCESS);
}
