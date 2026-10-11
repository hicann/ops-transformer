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
 * \file test_grouped_no_quant_arch35_tiling.cpp
 * \brief CSV-driven unit tests for GroupedMatmul no-quant arch35 tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

DataType ParseDtype(const string& dtype)
{
    const auto parsed = ops::ut::ParseGeDtype(dtype);
    if (parsed == ge::DT_UNDEFINED) {
        cerr << "Unsupported dtype in csv: " << dtype << endl;
    }
    return parsed;
}

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

string TilingData2Str(const void* tilingData, size_t tilingSize)
{
    ostringstream oss;
    const auto* data = reinterpret_cast<const int32_t*>(tilingData);
    const size_t len = tilingSize / sizeof(int32_t);
    for (size_t i = 0; i < len; ++i) {
        if (i != 0) {
            oss << " ";
        }
        oss << data[i];
    }
    return oss.str();
}

vector<gert::TilingContextPara::OpAttr> GetGroupedNoQuantAttrs(int64_t splitItem, bool transposeX, bool transposeWeight,
                                                               int64_t groupType, int64_t groupListType)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

class GroupedNoQuantArch35TilingTestParam {
public:
    void Prepare(optiling::GMMCompileInfo& compileInfo) const
    {
        compileInfo = {
            static_cast<uint32_t>(coreNum > 0 ? coreNum : 32),     // aicNum
            static_cast<uint32_t>(coreNum > 0 ? coreNum * 2 : 64), // aivNum
            262144,                                                // ubSize
            524288,                                                // l1Size
            134217728,                                             // l2Size
            262144,                                                // l0CSize
            65536,                                                 // l0ASize
            65536,                                                 // l0BSize
            platform_ascendc::SocVersion::ASCEND950,               // socVersion
            NpuArch::DAV_3510,
        };
    }

    void InvokeTilingFunc(optiling::GMMCompileInfo& compileInfo) const
    {
        vector<int64_t> xOriginDims = transposeX ? vector<int64_t>{k, m} : vector<int64_t>{m, k};
        gert::StorageShape xShape = ops::ut::MakeGertStorageShape(xOriginDims, xOriginDims);
        gert::StorageShape biasShape =
            hasBias ? ops::ut::MakeGertStorageShape({groupNum, n}, {groupNum, n}) : MakeEmptyShape();
        // groupListType 0(cumsum)/1(count): 1D [E]; groupListType 2(sparse): 2D [E, 2].
        const int64_t groupListDimNum =
            groupListDimNumOverride > 0 ? groupListDimNumOverride : (groupListType == 2 ? 2 : 1);
        gert::StorageShape groupListShape = groupListDimNum == 2 ?
                                                ops::ut::MakeGertStorageShape({groupNum, 2}, {groupNum, 2}) :
                                                ops::ut::MakeGertStorageShape({groupNum}, {groupNum});

        vector<int64_t> weightOriginDims =
            transposeWeight ? vector<int64_t>{groupNum, n, k} : vector<int64_t>{groupNum, k, n};
        vector<int64_t> weightStorageDims;
        if (weightFormat == "NZ") {
            weightStorageDims = transposeWeight ? vector<int64_t>{groupNum, (k + 31) / 32, (n + 15) / 16, 16, 32} :
                                                  vector<int64_t>{groupNum, (n + 31) / 32, (k + 15) / 16, 16, 32};
        } else {
            weightStorageDims = weightOriginDims;
        }
        gert::StorageShape weightShape = ops::ut::MakeGertStorageShape(weightOriginDims, weightStorageDims);

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul",
            {
                {xShape, xDtype, ge::FORMAT_ND},                                                          // x
                {weightShape, weightDtype, weightFormat == "NZ" ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND}, // weight
                {biasShape, biasDtype, ge::FORMAT_ND},                                                    // bias
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND}, // scale (empty for no quant)
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND}, // offset (empty for no quant)
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND}, // antiquantScale (empty for no quant)
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND}, // antiquantOffset (empty for no quant)
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},   // groupList
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND}, // perTokenScale (empty for no quant)
            },
            {{ops::ut::MakeGertStorageShape({m}, {n}), yDtype, ge::FORMAT_ND}},
            GetGroupedNoQuantAttrs(splitItem, transposeX, transposeWeight, groupType, groupListType), &compileInfo,
            "3510", compileInfo.aicNum, compileInfo.ubSize);

        TilingInfo tilingInfo;
        bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_EQ(tilingResult, result) << "prefix=" << prefix << ", case=" << caseName;
        if (!result) {
            return;
        }

        ASSERT_EQ(tilingInfo.blockNum, expectBlockDim) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingKey, expectTilingKey) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMNoQuantTilingData))
            << "prefix=" << prefix;

        string actualTilingData = TilingData2Str(tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
        ASSERT_EQ(actualTilingData, expectTilingData) << "prefix=" << prefix;
    }

    void Test() const
    {
        optiling::GMMCompileInfo compileInfo;
        Prepare(compileInfo);
        InvokeTilingFunc(compileInfo);
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    int64_t coreNum = -1;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t groupNum = 0;
    bool transposeX = false;
    bool transposeWeight = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t groupListDimNumOverride = 0; // 0: derive from groupListType; 1/2: force 1D/2D negative cases
    int64_t splitItem = 0;
    string weightFormat;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType weightDtype = ge::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DT_UNDEFINED;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    bool hasBias = false;
    bool result = false;
    uint64_t expectBlockDim = 0;
    uint64_t expectTilingKey = 0;
    string expectTilingData;
};

vector<GroupedNoQuantArch35TilingTestParam> GetParams(const string& socVersion)
{
    vector<GroupedNoQuantArch35TilingTestParam> params;
    std::string csvPath = ops::ut::ResolveCsvPath("test_grouped_no_quant_arch35_tiling.csv",
                                                  "gmm/grouped_matmul/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << ", maybe not exist" << endl;
        return params;
    }

    string line;
    size_t lineNo = 0U;
    while (getline(csvData, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < 18U) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            GroupedNoQuantArch35TilingTestParam param;
            size_t idx = 0UL;
            param.socVersion = items[idx++];
            if (param.socVersion != socVersion) {
                continue;
            }

            param.caseName = items[idx++];
            param.enable = ParseBool(items[idx++]);
            if (!param.enable) {
                continue;
            }
            param.prefix = items[idx++];
            param.coreNum = items[idx].empty() ? -1 : stoll(items[idx]);
            idx++;
            param.m = stoll(items[idx++]);
            param.k = stoll(items[idx++]);
            param.n = stoll(items[idx++]);
            param.groupNum = stoll(items[idx++]);
            param.transposeX = ParseBool(items[idx++]);
            param.transposeWeight = ParseBool(items[idx++]);
            param.groupType = stoll(items[idx++]);
            param.splitItem = stoll(items[idx++]);
            param.weightFormat = items[idx++];
            param.xDtype = ParseDtype(items[idx++]);
            param.weightDtype = ParseDtype(items[idx++]);
            param.biasDtype = ParseDtype(items[idx++]);
            param.yDtype = ParseDtype(items[idx++]);
            param.hasBias = ParseBool(items[idx++]);
            param.result = ParseBool(items[idx++]);
            param.expectBlockDim = static_cast<uint64_t>(stoull(items[idx++]));
            param.expectTilingKey = static_cast<uint64_t>(stoull(items[idx++]));
            param.expectTilingData = items[idx++];
            // Optional trailing columns for groupListType scenarios; absent columns keep the
            // legacy default (groupListType=0, dimNum derived from the type).
            if (items.size() > idx && !Trim(items[idx]).empty()) {
                param.groupListType = stoll(items[idx]);
            }
            idx++;
            if (items.size() > idx && !Trim(items[idx]).empty()) {
                param.groupListDimNumOverride = stoll(items[idx]);
            }
            idx++;
            params.push_back(param);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return params;
}

string MakeParamName(const testing::TestParamInfo<GroupedNoQuantArch35TilingTestParam>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedNoQuantArch35TilingUT {

const vector<GroupedNoQuantArch35TilingTestParam>& GetAscend950Params()
{
    static const vector<GroupedNoQuantArch35TilingTestParam> params = GetParams("Ascend950");
    return params;
}

class TestGroupedNoQuantArch35Tiling : public testing::TestWithParam<GroupedNoQuantArch35TilingTestParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedNoQuantArch35Tiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_NO_QUANT_950, TestGroupedNoQuantArch35Tiling, testing::ValuesIn(GetAscend950Params()),
                         MakeParamName);

} // namespace GroupedNoQuantArch35TilingUT
