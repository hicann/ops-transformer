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
    vector<int64_t> xShapeOverride;
    vector<int64_t> weightShapeOverride;

    void Test() const
    {
        optiling::GMMCompileInfo compileInfo;
        string archStr;
        if (socVersion == "Ascend960DT") {
            compileInfo = {32,
                           32,
                           262144,
                           524288,
                           134217728,
                           262144,
                           65536,
                           65536,
                           platform_ascendc::SocVersion::ASCEND950,
                           NpuArch::DAV_9201,
                           true};
            archStr = "9201";
        } else {
            compileInfo = {32,
                           64,
                           262144,
                           524288,
                           134217728,
                           262144,
                           65536,
                           65536,
                           platform_ascendc::SocVersion::ASCEND950,
                           NpuArch::DAV_3510,
                           false};
            archStr = "3510";
        }

        gert::StorageShape xShape = MakeShape(xShapeOverride.empty() ? vector<int64_t>{k, m} : xShapeOverride);
        gert::StorageShape weightShape =
            MakeShape(weightShapeOverride.empty() ? vector<int64_t>{k, n} : weightShapeOverride);
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
            &compileInfo, archStr.c_str(), compileInfo.aicNum, compileInfo.ubSize, actualTilingDataSize);

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
        // Additional tiling invariants apply only to the new mixed-input cases.
        if (xDtype == ge::DT_FLOAT8_E4M3FN &&
            (wDtype == ge::DT_FLOAT4_E2M1 || wDtype == Ops::Transformer::DT_HIFLOAT4)) {
            CheckPerGroupTiling(tilingInfo, compileInfo.l1Size);
        }
        if (k == 0 && xDtype == ge::DT_HIFLOAT8 &&
            (scale2Shape.size() == 1U || (scale2Shape.size() == 2U && scale2Shape[1] == 1))) {
            // The empty T-T reduction must still serialize the BasicApi layout;
            // the kernel skips the product and leaves the existing yRef intact.
            ASSERT_EQ(tilingInfo.tilingDataSize,
                      sizeof(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData));
            const auto* data = reinterpret_cast<const QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData*>(
                tilingInfo.tilingData.get());
            EXPECT_EQ(data->gmmQuantParams.groupNum, groupNum);
            EXPECT_EQ(data->gmmQuantParams.groupListType, groupListType);
            EXPECT_EQ(data->gmmQuantParams.aQuantMode, static_cast<uint32_t>(optiling::QuantMode::PERTENSOR_MODE));
            EXPECT_EQ(data->gmmQuantParams.bQuantMode, static_cast<uint32_t>(optiling::QuantMode::PERTENSOR_MODE));
            EXPECT_EQ(data->mmTilingData.m, m);
            EXPECT_EQ(data->mmTilingData.n, n);
            EXPECT_GT(data->mmTilingData.baseM, 0U);
            EXPECT_GT(data->mmTilingData.baseN, 0U);
            EXPECT_EQ(data->mmTilingData.k, 0U);
            EXPECT_EQ(data->mmTilingData.baseK, 0U);
            EXPECT_EQ(data->mmTilingData.kAL1, 0U);
            EXPECT_EQ(data->mmTilingData.kBL1, 0U);
        }
    }

    void CheckPerGroupTiling(const TilingInfo& tilingInfo, uint64_t l1Size) const
    {
        SCOPED_TRACE(caseName);
        // TN: atomic uses kernelType=0 (key 4), adaptive UB uses kernelType=3 (key 52).
        EXPECT_EQ(tilingInfo.tilingKey, checkTilingKey ? expectTilingKey : 4U);
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams));
        ASSERT_NE(tilingInfo.tilingData.get(), nullptr);
        const auto* data = reinterpret_cast<const QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams*>(
            tilingInfo.tilingData.get());
        EXPECT_EQ(data->quantGmmInplaceAddParams.groupNum, groupNum);
        EXPECT_EQ(data->quantGmmInplaceAddParams.groupListType, groupListType);
        const auto& mm = data->mmTilingData;
        EXPECT_EQ(mm.M, m);
        EXPECT_EQ(mm.N, n);
        EXPECT_EQ(mm.Ka, k);
        EXPECT_EQ(mm.Kb, k);
        if (k == 0) {
            return;
        }
        ASSERT_GT(mm.baseM, 0);
        ASSERT_GT(mm.baseN, 0);
        ASSERT_GT(mm.baseK, 0);
        const bool isFp4A = xDtype == ge::DT_FLOAT4_E2M1 || xDtype == Ops::Transformer::DT_HIFLOAT4;
        const bool isFp4B = wDtype == ge::DT_FLOAT4_E2M1 || wDtype == Ops::Transformer::DT_HIFLOAT4;
        if (isFp4B) {
            EXPECT_EQ(mm.baseN % 64, 0);
        }
        EXPECT_EQ(mm.baseK % 64, 0);
        ASSERT_GT(mm.stepKa, 0);
        ASSERT_GT(mm.stepKb, 0);
        const uint64_t scaleFactorA = mm.mxTypePara & 0xffU;
        const uint64_t scaleFactorB = (mm.mxTypePara >> 8U) & 0xffU;
        ASSERT_GE(scaleFactorA, 1U);
        ASSERT_LE(scaleFactorA, 127U);
        ASSERT_GE(scaleFactorB, 1U);
        ASSERT_LE(scaleFactorB, 127U);
        const uint64_t kAL1 = static_cast<uint64_t>(mm.stepKa) * mm.baseK;
        const uint64_t kBL1 = static_cast<uint64_t>(mm.stepKb) * mm.baseK;
        // Check the reserved windows, before the adapter possibly shortens the shared scale window.
        // E8M0 stores two one-byte scales per 64 K; HIF4 stores one four-byte scale.
        const uint64_t scaleABytes = ((scaleFactorA * kAL1 + 63U) / 64U) * mm.baseM *
                                     (scale1Dtype == Ops::Transformer::DT_HIFLOAT4_SCALE ? 4U : 2U);
        const uint64_t scaleBBytes = ((scaleFactorB * kBL1 + 63U) / 64U) * mm.baseN *
                                     (scale2Dtype == Ops::Transformer::DT_HIFLOAT4_SCALE ? 4U : 2U);
        const uint64_t dataABytes = mm.baseM * kAL1 / (isFp4A ? 2U : 1U);
        const uint64_t dataBBytes = mm.baseN * kBL1 / (isFp4B ? 2U : 1U);
        const uint64_t oneBufferBytes = dataABytes + dataBBytes + scaleABytes + scaleBBytes;
        EXPECT_LE(oneBufferBytes * 2U, l1Size);
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
            if (idx < items.size()) {
                p.xShapeOverride = ops::ut::ParseDims(items[idx++]);
            }
            if (idx < items.size()) {
                p.weightShapeOverride = ops::ut::ParseDims(items[idx++]);
            }
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

const vector<QGmmInplaceAddArch35Param>& GetAscend960DTParams()
{
    static const vector<QGmmInplaceAddArch35Param> params = GetParams("Ascend960DT");
    return params;
}

class TestQuantGroupedMatmulInplaceAddArch35Tiling : public testing::TestWithParam<QGmmInplaceAddArch35Param> {};

TEST_P(TestQuantGroupedMatmulInplaceAddArch35Tiling, csvDrivenCase)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(QUANT_GROUPED_MATMUL_INPLACE_ADD_950, TestQuantGroupedMatmulInplaceAddArch35Tiling,
                         testing::ValuesIn(GetAscend950Params()), MakeParamName);

INSTANTIATE_TEST_SUITE_P(QUANT_GROUPED_MATMUL_INPLACE_ADD_960DT, TestQuantGroupedMatmulInplaceAddArch35Tiling,
                         testing::ValuesIn(GetAscend960DTParams()), MakeParamName);
} // namespace QuantGroupedMatmulInplaceAddArch35TilingUT
