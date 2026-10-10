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
#include <cstdlib>
#include "../allto_all_matmul_host_ut_param.h"
#include "tiling_case_executor.h"

static std::string GetCsvPath(const char* file)
{
    const char* envPath = std::getenv("CSV_CASE_PATH");
    if (envPath != nullptr && strlen(envPath) > 0) {
        return std::string(envPath);
    }
    return ReplaceFileExtension2Csv(file);
}

namespace AlltoAllMatmulUT {

static const std::string OP_NAME = "AlltoAllMatmul";

struct AlltoAllMatmulCompileInfo {
} compileInfo;

class AlltoAllMatmulArch35TilingTest : public testing::TestWithParam<AlltoAllMatmulTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulArch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AlltoAllMatmulArch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale, param.commScale, param.x1Offset,
         param.x2Offset});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.all2allAxes))},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtypeAttr))},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantMode)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtype)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
         {"alltoallout_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.alltoalloutFlag)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

INSTANTIATE_TEST_SUITE_P(AlltoAllMatmulTilingUT, AlltoAllMatmulArch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulTilingUtParam>(GetCsvPath(__FILE__))),
                         PrintCaseInfoString<AlltoAllMatmulTilingUtParam>);

} // namespace AlltoAllMatmulUT
