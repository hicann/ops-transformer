/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul.cpp
 * \brief CSV-driven unit tests for grouped_matmul tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/grouped_matmul_tiling.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kInputTensorCount = 9U;
constexpr size_t kTensorFieldCount = 4U;
constexpr size_t kScalarFieldCount = 22U;
constexpr size_t kCsvColumnCount = kScalarFieldCount + kInputTensorCount * kTensorFieldCount + kTensorFieldCount;
constexpr int64_t kDefaultQuantGroupSize = 256;

struct RawTensorFields {
    string origin;
    string storage;
    string dtype;
    string format;
};

struct TensorDescParam {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
    TensorDescParam() = default;
    TensorDescParam(vector<int64_t> o, vector<int64_t> s, ge::DataType d, ge::Format f)
        : originDims(std::move(o)),
          storageDims(std::move(s)),
          dtype(d),
          format(f)
    {}
};

vector<string> ParseTensorSpecs(const string& value)
{
    const string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    vector<string> specs;
    SplitStr2Vec(trimmed, ";", specs);
    for (auto& spec : specs) {
        spec = Trim(spec);
    }
    return specs;
}

vector<int64_t> ParseDims(const string& value, const map<string, int64_t>& symbols = {})
{
    return ops::ut::ParseDims(value, symbols);
}

vector<int64_t> ParseIntList(const string& value, const map<string, int64_t>& symbols = {})
{
    return ParseDims(value, symbols);
}

ge::DataType ParseDtype(const string& dtype)
{
    const auto parsed = ops::ut::ParseGeDtype(dtype);
    if (parsed == ge::DT_UNDEFINED && Trim(dtype) != "UNDEFINED") {
        cerr << "Unsupported dtype in csv: " << dtype << endl;
    }
    return parsed;
}

ge::Format ParseFormat(const string& format)
{
    return ops::ut::ParseGeFormat(format);
}

map<string, int64_t> BuildSymbols(const RawTensorFields& xFields, const RawTensorFields& weightFields, bool transposeX,
                                  bool transposeWeight)
{
    map<string, int64_t> symbols;

    auto xSpecs = ParseTensorSpecs(xFields.origin);
    string xOrigin = xSpecs.empty() ? Trim(xFields.origin) : xSpecs[0];
    const vector<int64_t> xOriginDims = ParseDims(xOrigin);
    if (xOriginDims.size() >= 2U) {
        const int64_t mValue = transposeX ? xOriginDims[1] : xOriginDims[0];
        const int64_t kValue = transposeX ? xOriginDims[0] : xOriginDims[1];
        symbols["M"] = mValue;
        symbols["m"] = mValue;
        symbols["K"] = kValue;
        symbols["k"] = kValue;
    }

    auto wSpecs = ParseTensorSpecs(weightFields.origin);
    string wOrigin = wSpecs.empty() ? Trim(weightFields.origin) : wSpecs[0];
    const vector<int64_t> weightOriginDims = ParseDims(wOrigin);
    if (weightOriginDims.size() >= 3U) {
        const int64_t groupNum = weightOriginDims[0];
        const int64_t kValue = transposeWeight ? weightOriginDims[2] : weightOriginDims[1];
        const int64_t nValue = transposeWeight ? weightOriginDims[1] : weightOriginDims[2];
        symbols["E"] = groupNum;
        symbols["e"] = groupNum;
        symbols["groupNum"] = groupNum;
        symbols["K"] = kValue;
        symbols["k"] = kValue;
        symbols["N"] = nValue;
        symbols["n"] = nValue;
        if (kValue > 0) {
            symbols["quantGroupNum"] = kValue / kDefaultQuantGroupSize;
        }
    } else if (weightOriginDims.size() >= 2U) {
        const int64_t kValue = transposeWeight ? weightOriginDims[1] : weightOriginDims[0];
        const int64_t nValue = transposeWeight ? weightOriginDims[0] : weightOriginDims[1];
        symbols["K"] = kValue;
        symbols["k"] = kValue;
        symbols["N"] = nValue;
        symbols["n"] = nValue;
        if (kValue > 0) {
            symbols["quantGroupNum"] = kValue / kDefaultQuantGroupSize;
        }
    }

    return symbols;
}

TensorDescParam ParseTensorDesc(const RawTensorFields& rawFields, const map<string, int64_t>& symbols)
{
    return {ParseDims(rawFields.origin, symbols), ParseDims(rawFields.storage, symbols), ParseDtype(rawFields.dtype),
            ParseFormat(rawFields.format)};
}

platform_ascendc::SocVersion ParseSocVersion(const string& socVersion)
{
    const string trimmed = Trim(socVersion);
    if (trimmed == "Ascend950") {
        return platform_ascendc::SocVersion::ASCEND950;
    }
    if (trimmed == "Ascend910_93") {
        return platform_ascendc::SocVersion::ASCEND910_93;
    }
    return platform_ascendc::SocVersion::ASCEND910B;
}

NpuArch ParseNpuArch(const string& socVersion)
{
    const string trimmed = Trim(socVersion);
    return trimmed == "Ascend950" ? NpuArch::DAV_3510 : NpuArch::DAV_2201;
}

struct GroupedMatmulTilingCase {
    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    uint64_t expectTilingKey = 0;
    uint32_t aicNum = 24;
    uint32_t aivNum = 48;
    uint64_t ubSize = 196608;
    uint64_t l1Size = 524288;
    uint64_t l2Size = 196608;
    uint64_t l0CSize = 131072;
    uint64_t l0ASize = 65536;
    uint64_t l0BSize = 65536;
    int64_t splitItem = 3;
    int64_t dtypeAttr = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    vector<int64_t> tuningConfig;
    array<vector<TensorDescParam>, kInputTensorCount> inputs;
    TensorDescParam output;

    void Run() const
    {
        SCOPED_TRACE("socVersion=" + socVersion + ", prefix=" + prefix + ", caseName=" + caseName);

        optiling::GMMCompileInfo compileInfo = {
            aicNum,
            aivNum,
            ubSize,
            l1Size,
            l2Size,
            l0CSize,
            l0ASize,
            l0BSize,
            ParseSocVersion(socVersion),
            ParseNpuArch(socVersion),
        };

        vector<uint32_t> inputInstanceNum(kInputTensorCount);
        bool hasMultiInstance = false;
        for (size_t i = 0; i < kInputTensorCount; ++i) {
            inputInstanceNum[i] = static_cast<uint32_t>(inputs[i].size());
            if (inputInstanceNum[i] > 1U) {
                hasMultiInstance = true;
            }
        }

        auto attrs = vector<gert::TilingContextPara::OpAttr>{
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtypeAttr)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(
                                  tuningConfig.empty() ? vector<int64_t>{0} : tuningConfig)},
        };

        auto runTest = [&](gert::TilingContextPara& para) {
            ExecuteTestCase(para, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                            expectSuccess ? expectTilingKey : 0);
        };

        if (hasMultiInstance) {
            vector<gert::TilingContextPara::TensorDescription> flatInputDescs;
            for (size_t i = 0; i < kInputTensorCount; ++i) {
                for (const auto& tensor : inputs[i]) {
                    flatInputDescs.emplace_back(BuildTensorDesc(tensor));
                }
            }
            vector<uint32_t> outputInstanceNum = {1U};
            gert::TilingContextPara tilingContextPara("GroupedMatmul", flatInputDescs, {BuildTensorDesc(output)}, attrs,
                                                      inputInstanceNum, outputInstanceNum, &compileInfo, socVersion,
                                                      aicNum, ubSize);
            runTest(tilingContextPara);
        } else {
            vector<gert::TilingContextPara::TensorDescription> inputDescs;
            inputDescs.reserve(kInputTensorCount);
            for (size_t i = 0; i < kInputTensorCount; ++i) {
                if (inputs[i].empty()) {
                    inputDescs.emplace_back(ops::ut::MakeGertStorageShape({}, {}), ge::DT_UNDEFINED, ge::FORMAT_ND);
                } else {
                    inputDescs.emplace_back(BuildTensorDesc(inputs[i][0]));
                }
            }
            gert::TilingContextPara tilingContextPara("GroupedMatmul", inputDescs, {BuildTensorDesc(output)}, attrs,
                                                      &compileInfo, socVersion, aicNum, ubSize);
            runTest(tilingContextPara);
        }
    }

private:
    static gert::TilingContextPara::TensorDescription BuildTensorDesc(const TensorDescParam& tensor)
    {
        return {ops::ut::MakeGertStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype, tensor.format};
    }
};

vector<GroupedMatmulTilingCase> LoadCases(const string& socVersion)
{
    vector<GroupedMatmulTilingCase> cases;

    const string csvPath =
        ops::ut::ResolveCsvPath("test_grouped_matmul_tiling.csv", "gmm/grouped_matmul/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
        return cases;
    }

    string line;
    size_t lineNo = 0U;
    while (getline(csvData, line)) {
        ++lineNo;
        const string trimmedLine = Trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue;
        }

        vector<string> items;
        SplitStr2Vec(trimmedLine, ",", items);
        if (items.empty() || items[0] == "socVersion") {
            continue;
        }
        if (items.size() < kCsvColumnCount) {
            cerr << "Skip invalid csv line with column count " << items.size() << ", expect at least "
                 << kCsvColumnCount << endl;
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            size_t idx = 0U;
            GroupedMatmulTilingCase tc;
            tc.socVersion = Trim(items[idx++]);
            if (tc.socVersion != socVersion) {
                continue;
            }

            tc.caseName = Trim(items[idx++]);
            tc.enable = ParseBool(items[idx++]);
            if (!tc.enable) {
                continue;
            }

            tc.prefix = Trim(items[idx++]);
            tc.expectSuccess = ParseBool(items[idx++]);
            tc.expectTilingKey = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.aicNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
            tc.aivNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
            tc.ubSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.l1Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.l2Size = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.l0CSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.l0ASize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.l0BSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.splitItem = stoll(Trim(items[idx++]));
            tc.dtypeAttr = stoll(Trim(items[idx++]));
            tc.transposeWeight = ParseBool(items[idx++]);
            tc.transposeX = ParseBool(items[idx++]);
            tc.groupType = stoll(Trim(items[idx++]));
            tc.groupListType = stoll(Trim(items[idx++]));
            tc.actType = stoll(Trim(items[idx++]));
            const string rawTuningConfig = Trim(items[idx++]);

            array<RawTensorFields, kInputTensorCount> rawInputs;
            for (size_t tensorIdx = 0; tensorIdx < kInputTensorCount; ++tensorIdx) {
                rawInputs[tensorIdx] = {Trim(items[idx++]), Trim(items[idx++]), Trim(items[idx++]), Trim(items[idx++])};
            }
            const RawTensorFields rawOutput = {Trim(items[idx++]), Trim(items[idx++]), Trim(items[idx++]),
                                               Trim(items[idx++])};

            const map<string, int64_t> symbols =
                BuildSymbols(rawInputs[0], rawInputs[1], tc.transposeX, tc.transposeWeight);
            tc.tuningConfig = ParseIntList(rawTuningConfig, symbols);
            for (size_t tensorIdx = 0; tensorIdx < kInputTensorCount; ++tensorIdx) {
                auto originSpecs = ParseTensorSpecs(rawInputs[tensorIdx].origin);
                auto storageSpecs = ParseTensorSpecs(rawInputs[tensorIdx].storage);
                if (originSpecs.empty()) {
                    tc.inputs[tensorIdx] = {};
                } else {
                    for (size_t specIdx = 0; specIdx < originSpecs.size(); ++specIdx) {
                        const string& originSpec = originSpecs[specIdx];
                        const string& storageSpec = specIdx < storageSpecs.size() ? storageSpecs[specIdx] : originSpec;
                        tc.inputs[tensorIdx].emplace_back(
                            ParseDims(originSpec, symbols), ParseDims(storageSpec, symbols),
                            ParseDtype(rawInputs[tensorIdx].dtype), ParseFormat(rawInputs[tensorIdx].format));
                    }
                }
            }
            tc.output = ParseTensorDesc(rawOutput, symbols);
            cases.push_back(tc);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }

    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulTilingCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.socVersion + "_" + info.param.prefix);
}

} // namespace

namespace GroupedMatmulTilingUT {

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestGroupedMatmulAscend950Tiling);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestGroupedMatmulAscend910BTiling);

const vector<GroupedMatmulTilingCase>& GetAscend950Cases()
{
    static const vector<GroupedMatmulTilingCase> cases = LoadCases("Ascend950");
    return cases;
}

const vector<GroupedMatmulTilingCase>& GetAscend910BCases()
{
    static const vector<GroupedMatmulTilingCase> cases = LoadCases("Ascend910B");
    return cases;
}

class TestGroupedMatmulAscend950Tiling : public testing::TestWithParam<GroupedMatmulTilingCase> {};

TEST_P(TestGroupedMatmulAscend950Tiling, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_TILING_950, TestGroupedMatmulAscend950Tiling, testing::ValuesIn(GetAscend950Cases()),
                         MakeParamName);

class TestGroupedMatmulAscend910BTiling : public testing::TestWithParam<GroupedMatmulTilingCase> {};

TEST_P(TestGroupedMatmulAscend910BTiling, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_TILING_910B, TestGroupedMatmulAscend910BTiling, testing::ValuesIn(GetAscend910BCases()),
                         MakeParamName);

} // namespace GroupedMatmulTilingUT
