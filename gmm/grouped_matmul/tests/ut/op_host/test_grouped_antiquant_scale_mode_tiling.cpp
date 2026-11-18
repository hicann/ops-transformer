/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_antiquant_scale_mode_tiling.cpp
 * \brief CSV-driven unit tests for per-channel antiquantization shape selection.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_matmul_tiling_tool.h"
#include "gmm_csv_ge_parse_utils.h"

namespace {
struct GroupedAntiquantScaleModeTestParam {
    std::string socVersion;
    std::string caseName;
    bool enable = true;
    std::string prefix;
    gert::Shape antiquantScaleShape;
    gert::Shape weightShape;
    uint64_t groupNum = 0;
    bool transposeWeight = false;
    bool result = false;

    void Test() const
    {
        SCOPED_TRACE(testing::Message() << "prefix=" << prefix << ", case=" << caseName);
        EXPECT_EQ(optiling::GMMWeightQuantTilingTool::IsPerChannelAntiquantScale(antiquantScaleShape, weightShape,
                                                                                 groupNum, transposeWeight),
                  result);
    }
};

std::vector<GroupedAntiquantScaleModeTestParam> GetParams(const std::string& socVersion)
{
    std::vector<GroupedAntiquantScaleModeTestParam> params;
    const auto csvPath = ops::ut::ResolveCsvPath("test_grouped_antiquant_scale_mode_tiling.csv",
                                                 "gmm/grouped_matmul/tests/ut/op_host", __FILE__);
    std::ifstream csvData(csvPath);
    if (!csvData.is_open()) {
        ADD_FAILURE() << "cannot open case file " << csvPath;
        return params;
    }
    std::string line;
    size_t lineNo = 0;
    bool hasHeader = false;
    std::set<std::string> paramNames;
    while (std::getline(csvData, line)) {
        ++lineNo;
        line = ops::ut::Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!hasHeader) {
            if (line != "socVersion,caseName,enable,prefix,antiquantScaleShape,weightShape,groupNum,transposeWeight,"
                        "result") {
                ADD_FAILURE() << csvPath << ":" << lineNo << ": unexpected CSV header";
                return {};
            }
            hasHeader = true;
            continue;
        }
        std::vector<std::string> items;
        ops::ut::SplitStr2Vec(line, ",", items);
        constexpr size_t FIELD_COUNT = 9;
        if (items.size() != FIELD_COUNT) {
            ADD_FAILURE() << csvPath << ":" << lineNo << ": expected " << FIELD_COUNT << " columns";
            continue;
        }
        const auto caseName = ops::ut::Trim(items[1]);
        try {
            GroupedAntiquantScaleModeTestParam param;
            size_t idx = 0;
            param.socVersion = ops::ut::Trim(items[idx++]);
            if (param.socVersion != socVersion) {
                continue;
            }
            param.caseName = caseName;
            ++idx;
            param.enable = ops::ut::ParseBool(items[idx++]);
            if (!param.enable) {
                continue;
            }
            param.prefix = ops::ut::Trim(items[idx++]);
            param.antiquantScaleShape = ops::ut::BuildGertShape(ops::ut::ParseDims(items[idx++]));
            param.weightShape = ops::ut::BuildGertShape(ops::ut::ParseDims(items[idx++]));
            param.groupNum = std::stoull(items[idx++]);
            param.transposeWeight = ops::ut::ParseBool(items[idx++]);
            param.result = ops::ut::ParseBool(items[idx++]);
            const auto name = ops::ut::MakeSafeParamName(param.prefix);
            if (name.empty() || !paramNames.insert(name).second) {
                ADD_FAILURE() << csvPath << ":" << lineNo << ": empty or duplicate prefix " << param.prefix;
                continue;
            }
            params.push_back(param);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    if (params.empty()) {
        ADD_FAILURE() << "no enabled cases for " << socVersion << " in " << csvPath;
    }
    return params;
}

const std::vector<GroupedAntiquantScaleModeTestParam>& GetAscend960DTParams()
{
    static const auto params = GetParams("Ascend960DT");
    return params;
}

std::string MakeParamName(const testing::TestParamInfo<GroupedAntiquantScaleModeTestParam>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedAntiquantScaleModeTilingUT {
class TestGroupedAntiquantScaleModeTiling : public testing::TestWithParam<GroupedAntiquantScaleModeTestParam> {};

TEST_P(TestGroupedAntiquantScaleModeTiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_ANTIQUANT_SCALE_MODE_960DT, TestGroupedAntiquantScaleModeTiling,
                         testing::ValuesIn(GetAscend960DTParams()), MakeParamName);
} // namespace GroupedAntiquantScaleModeTilingUT
