/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../stem_oam_prep_varlen_q_param.h"
#include "tiling_case_executor.h"

namespace StemOamPrepVarlenQUT {

class StemPrepQArch35TilingTest : public testing::TestWithParam<StemPrepQTilingUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "StemOamPrepVarlenQ Arch35 TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "StemOamPrepVarlenQ Arch35 TilingTest TearDown" << std::endl; }
};

TEST_P(StemPrepQArch35TilingTest, tiling)
{
    auto param = GetParam();

    gert::TilingContextPara tilingContextPara(
        "StemOamPrepVarlenQ",
        {
            param.q,
            param.qSeqLens,
            param.cuSeqLensQ,
            param.qScale,
        },
        {
            param.qFlat,
        },
        {
            {"stemBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stemBlockSize)},
            {"stemStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.stem_stride)},
        },
        param.inputInstance, param.outputInstance, &param.compileInfo, "Ascend950", 64, 262144, 4096);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(StemOamPrepVarlenQ, StemPrepQArch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<StemPrepQTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<StemPrepQTilingUtParam>);

} // namespace StemOamPrepVarlenQUT
