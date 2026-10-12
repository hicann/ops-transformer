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
#include "scatter_pa_kv_cache_with_k_scale_param.h"
#include "tiling_case_executor.h"

namespace ScatterPaKvCacheWithKScaleUT {

class ScatterPaKvCacheWithKScaleArch35TilingTest
    : public testing::TestWithParam<ScatterPaKvCacheWithKScaleTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterPaKvCacheWithKScale Arch35 TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ScatterPaKvCacheWithKScale Arch35 TilingTest TearDown" << std::endl;
    }
};

TEST_P(ScatterPaKvCacheWithKScaleArch35TilingTest, param)
{
    auto param = GetParam();

    struct EmptyCompileInfo {
    } compileInfo;

    const std::string SocInfo35 = "{\n"
                                  "  \"hardware_info\": {\n"
                                  "    \"BT_SIZE\": 0,\n"
                                  "    \"load3d_constraints\": \"1\",\n"
                                  "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                  "    \"Intrinsic_data_move_l12ub\": true,\n"
                                  "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                  "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                  "    \"UB_SIZE\": 262144,\n"
                                  "    \"L2_SIZE\": 33554432,\n"
                                  "    \"L1_SIZE\": 524288,\n"
                                  "    \"L0A_SIZE\": 65536,\n"
                                  "    \"L0B_SIZE\": 65536,\n"
                                  "    \"L0C_SIZE\": 131072,\n"
                                  "    \"CORE_NUM\": 40,\n"
                                  "    \"socVersion\": \"Ascend950\"\n"
                                  "  }\n"
                                  "}";

    gert::TilingContextPara tilingContextPara(
        "ScatterPaKvCacheWithKScale",
        {param.key, param.value, param.key_cache, param.value_cache, param.slot_mapping, param.key_scale,
         param.key_scale_cache},
        {param.key_cache_out, param.value_cache_out, param.key_scale_cache_out},
        {{"cache_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.cache_layout)}},
        param.inputInstance, param.outputInstance, &compileInfo, "Ascend950", 40, 262144, 4096, SocInfo35);

    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(
    ScatterPaKvCacheWithKScale, ScatterPaKvCacheWithKScaleArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<ScatterPaKvCacheWithKScaleTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ScatterPaKvCacheWithKScaleTilingUtParam>);

} // namespace ScatterPaKvCacheWithKScaleUT
