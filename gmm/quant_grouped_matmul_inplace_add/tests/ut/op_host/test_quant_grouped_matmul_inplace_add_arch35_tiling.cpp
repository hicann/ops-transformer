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
 * \file test_quant_grouped_matmul_inplace_add_arch35_tiling.cpp
 * \brief CSV-driven unit tests for QuantGroupedMatmulInplaceAdd arch35 tiling.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/quant_grouped_matmul_inplace_add_tiling.h"
#include "../../../op_host/op_tiling/quant_grouped_matmul_inplace_add_basic_api_tiling.h"
#include "../../../op_kernel/arch35/quant_grouped_matmul_inplace_add_tiling_data.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

gert::StorageShape MakeShape(const vector<int64_t>& dims)
{
    return ops::ut::MakeGertStorageShape(dims);
}

struct QGmmInplaceAddArch35Param {
    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    int64_t groupNum = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    DataType xDtype = ge::DT_UNDEFINED;
    DataType wDtype = ge::DT_UNDEFINED;
    DataType scale2Dtype = ge::DT_UNDEFINED;
    DataType scale1Dtype = ge::DT_UNDEFINED;
    DataType yDtype = ge::DT_UNDEFINED;
    int64_t groupListType = 0;
    bool expectSuccess = false;
    bool checkTilingKey = false;
    uint64_t expectTilingKey = 0;
    vector<int64_t> scale2Shape;
    vector<int64_t> scale1Shape;
    uint64_t tilingDataSize = 0;

    void Test() const
    {
        optiling::GMMCompileInfo compileInfo = {
            32,
            64,
            262144,
            524288,
            134217728,
            262144,
            65536,
            65536,
            platform_ascendc::SocVersion::ASCEND950,
            NpuArch::DAV_3510,
        };

        gert::StorageShape xShape = MakeShape({k, m});
        gert::StorageShape weightShape = MakeShape({k, n});
        gert::StorageShape groupListShape = MakeShape({groupNum});
        gert::StorageShape yShape = MakeShape({groupNum, m, n});

        uint64_t actualTilingDataSize = (tilingDataSize == 0) ? 4096 : tilingDataSize;

        gert::TilingContextPara tilingContextPara(
            "QuantGroupedMatmulInplaceAdd",
            {
                {xShape, xDtype, ge::FORMAT_ND},
                {weightShape, wDtype, ge::FORMAT_ND},
                {MakeShape(scale2Shape), scale2Dtype, ge::FORMAT_ND},
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},
                {yShape, yDtype, ge::FORMAT_ND},
                {MakeShape(scale1Shape), scale1Dtype, ge::FORMAT_ND},
            },
            {{yShape, yDtype, ge::FORMAT_ND}},
            {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
             {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
            &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize, actualTilingDataSize);

        TilingInfo tilingInfo;
        bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_EQ(tilingResult, expectSuccess) << "case=" << caseName;
        if (!expectSuccess) {
            return;
        }

        ASSERT_GT(tilingInfo.blockNum, 0U) << "case=" << caseName;
        ASSERT_TRUE(tilingInfo.tilingDataSize == sizeof(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams) ||
                    tilingInfo.tilingDataSize == sizeof(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData))
            << "case=" << caseName << " tilingDataSize=" << tilingInfo.tilingDataSize;
        if (checkTilingKey) {
            ASSERT_EQ(tilingInfo.tilingKey, expectTilingKey) << "case=" << caseName;
        }
    }
};

vector<QGmmInplaceAddArch35Param> GetParams(const string& socVersion)
{
    vector<QGmmInplaceAddArch35Param> params;
    string csvPath = ops::ut::ResolveCsvPath("test_quant_grouped_matmul_inplace_add_arch35_tiling.csv",
                                             "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host", __FILE__);
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 19U) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            size_t idx = 0;
            QGmmInplaceAddArch35Param p;
            p.socVersion = Trim(items[idx++]);
            if (p.socVersion != socVersion) {
                continue;
            }
            p.caseName = Trim(items[idx++]);
            p.enable = ParseBool(items[idx++]);
            if (!p.enable) {
                continue;
            }
            p.prefix = Trim(items[idx++]);
            p.groupNum = stoll(Trim(items[idx++]));
            p.m = stoll(Trim(items[idx++]));
            p.k = stoll(Trim(items[idx++]));
            p.n = stoll(Trim(items[idx++]));
            p.xDtype = ops::ut::ParseGeDtype(items[idx++]);
            p.wDtype = ops::ut::ParseGeDtype(items[idx++]);
            p.scale2Dtype = ops::ut::ParseGeDtype(items[idx++]);
            p.scale1Dtype = ops::ut::ParseGeDtype(items[idx++]);
            p.yDtype = ops::ut::ParseGeDtype(items[idx++]);
            p.groupListType = stoll(Trim(items[idx++]));
            p.expectSuccess = ParseBool(items[idx++]);
            string keyField = Trim(items[idx++]);
            p.checkTilingKey = !keyField.empty() && keyField != "NONE";
            p.expectTilingKey = p.checkTilingKey ? static_cast<uint64_t>(stoull(keyField)) : 0;
            p.scale2Shape = ops::ut::ParseDims(items[idx++]);
            p.scale1Shape = ops::ut::ParseDims(items[idx++]);
            p.tilingDataSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            params.push_back(p);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return params;
}

string MakeParamName(const testing::TestParamInfo<QGmmInplaceAddArch35Param>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}
} // namespace

namespace QuantGroupedMatmulInplaceAddArch35TilingUT {
const vector<QGmmInplaceAddArch35Param>& GetAscend950Params()
{
    static const vector<QGmmInplaceAddArch35Param> params = GetParams("Ascend950");
    return params;
}

class TestQuantGroupedMatmulInplaceAddArch35Tiling : public testing::TestWithParam<QGmmInplaceAddArch35Param> {};

TEST_P(TestQuantGroupedMatmulInplaceAddArch35Tiling, csvDrivenCase)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(QUANT_GROUPED_MATMUL_INPLACE_ADD_950, TestQuantGroupedMatmulInplaceAddArch35Tiling,
                         testing::ValuesIn(GetAscend950Params()), MakeParamName);
} // namespace QuantGroupedMatmulInplaceAddArch35TilingUT
