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
 * \file test_grouped_matmul_activation_quant_tiling.cpp
 * \brief CSV-driven unit tests for GroupedMatmulActivationQuant tiling.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_matmul_activation_quant_tiling.h"
#include "gmm_csv_ge_parse_utils.h"
#include "platform/platform_infos_def.h"
#include "tiling_case_executor.h"

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

struct TensorDescParam {
    vector<int64_t> originDims;
    vector<int64_t> storageDims;
    ge::DataType dtype = ge::DT_UNDEFINED;
    ge::Format format = ge::FORMAT_ND;
};

constexpr size_t INPUT_TENSOR_COUNT = 6U;
constexpr size_t OUTPUT_TENSOR_COUNT = 2U;
constexpr size_t TENSOR_FIELD_COUNT = 4U;
constexpr size_t SCALAR_FIELD_COUNT = 19U;
constexpr size_t CSV_COLUMN_COUNT =
    SCALAR_FIELD_COUNT + (INPUT_TENSOR_COUNT + OUTPUT_TENSOR_COUNT) * TENSOR_FIELD_COUNT;
constexpr uint32_t DEFAULT_L1_SIZE = 512288U;
constexpr uint32_t DEFAULT_L2_SIZE = 134217728U;
constexpr uint32_t DEFAULT_L0C_SIZE = 262144U;
constexpr uint32_t DEFAULT_L0A_SIZE = 65536U;
constexpr uint32_t DEFAULT_L0B_SIZE = 65536U;
constexpr uint32_t DEFAULT_AIC_NUM = 32U;
constexpr uint32_t DEFAULT_AIV_NUM = 64U;
constexpr uint64_t DEFAULT_UB_SIZE = 262144UL;
constexpr uint64_t MX_EPILOGUE_MAX_ELEMENTS_PER_AIV = 128UL * 256UL;
constexpr uint64_t MX_EPILOGUE_AIV_COUNT = 2UL;

struct GroupedMatmulActivationQuantTilingCase {
    void Run() const
    {
        if (!enable) {
            return;
        }
        optiling::GMMCompileInfo compileInfo = BuildCompileInfo();
        vector<int64_t> tuningConfigVec = ops::ut::ParseDims(tuningConfig, {});
        if (tuningConfigVec.empty()) {
            tuningConfigVec = {0};
        }
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulActivationQuant", BuildTensorDescs(inputs), BuildTensorDescs(outputs),
            {
                {"activation_type", Ops::Transformer::AnyValue::CreateFrom<string>(activationType)},
                {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
                {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tuningConfigVec)},
                {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<string>(quantMode)},
                {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yDtype)},
                {"round_mode", Ops::Transformer::AnyValue::CreateFrom<string>(roundMode)},
                {"scale_alg", Ops::Transformer::AnyValue::CreateFrom<int64_t>(scaleAlg)},
                {"dst_type_max", Ops::Transformer::AnyValue::CreateFrom<float>(dstTypeMax)},
            },
            &compileInfo, socVersion, compileInfo.aicNum, compileInfo.ubSize);

        if (!expectSuccess) {
            ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
            return;
        }

        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "prefix=" << prefix;
        EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey) << "prefix=" << prefix;
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(GroupedMatmulActivationQuant::GMMActivationQuantTilingDataParams));
        const auto *tilingData =
            reinterpret_cast<const GroupedMatmulActivationQuant::GMMActivationQuantTilingDataParams *>(
                tilingInfo.tilingData.get());
        ASSERT_NE(tilingData, nullptr);
        EXPECT_LE(((static_cast<uint64_t>(tilingData->mmTilingData.baseM) + MX_EPILOGUE_AIV_COUNT - 1) /
                   MX_EPILOGUE_AIV_COUNT) *
                      tilingData->mmTilingData.baseN,
                  MX_EPILOGUE_MAX_ELEMENTS_PER_AIV)
            << "prefix=" << prefix;
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    int64_t expectTilingKey = 0;
    uint32_t aicNum = DEFAULT_AIC_NUM;
    uint32_t aivNum = DEFAULT_AIV_NUM;
    uint64_t ubSize = DEFAULT_UB_SIZE;
    bool transposeWeight = false;
    int64_t groupListType = 0;
    string quantMode;
    int64_t yDtype = 0;
    string roundMode;
    int64_t scaleAlg = 0;
    float dstTypeMax = 0.0F;
    string tuningConfig;
    string activationType;
    vector<TensorDescParam> inputs;
    vector<TensorDescParam> outputs;

private:
    optiling::GMMCompileInfo BuildCompileInfo() const
    {
        return {
            aicNum,
            aivNum,
            ubSize,
            DEFAULT_L1_SIZE,
            DEFAULT_L2_SIZE,
            DEFAULT_L0C_SIZE,
            DEFAULT_L0A_SIZE,
            DEFAULT_L0B_SIZE,
            platform_ascendc::SocVersion::ASCEND950,
            NpuArch::DAV_3510,
        };
    }

    static vector<gert::TilingContextPara::TensorDescription> BuildTensorDescs(const vector<TensorDescParam> &tensors)
    {
        vector<gert::TilingContextPara::TensorDescription> descs;
        descs.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            descs.emplace_back(ops::ut::MakeGertStorageShape(tensor.originDims, tensor.storageDims), tensor.dtype,
                               tensor.format);
        }
        return descs;
    }
};

vector<GroupedMatmulActivationQuantTilingCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulActivationQuantTilingCase> cases;
    const string csvPath = ops::ut::ResolveCsvPath("test_grouped_matmul_activation_quant_tiling.csv",
                                                   "gmm/grouped_matmul_activation_quant/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
        return cases;
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
        if (items.empty() || items[0] == "socVersion" || items.size() < CSV_COLUMN_COUNT) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            size_t idx = 0U;
            GroupedMatmulActivationQuantTilingCase tc;
            tc.socVersion = Trim(items[idx++]);
            if (tc.socVersion != socVersion) {
                continue;
            }
            tc.caseName = Trim(items[idx++]);
            tc.enable = ParseBool(Trim(items[idx++]));
            if (!tc.enable) {
                continue;
            }
            tc.prefix = Trim(items[idx++]);
            tc.expectSuccess = ParseBool(Trim(items[idx++]));
            tc.expectTilingKey = stoll(Trim(items[idx++]));
            tc.aicNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
            tc.aivNum = static_cast<uint32_t>(stoul(Trim(items[idx++])));
            tc.ubSize = static_cast<uint64_t>(stoull(Trim(items[idx++])));
            tc.activationType = Trim(items[idx++]);
            tc.transposeWeight = ParseBool(Trim(items[idx++]));
            tc.groupListType = stoll(Trim(items[idx++]));
            tc.tuningConfig = Trim(items[idx++]);
            tc.quantMode = Trim(items[idx++]);
            tc.yDtype = stoll(Trim(items[idx++]));
            tc.roundMode = Trim(items[idx++]);
            tc.scaleAlg = stoll(Trim(items[idx++]));
            tc.dstTypeMax = stof(Trim(items[idx++]));

            for (size_t tensorIdx = 0U; tensorIdx < INPUT_TENSOR_COUNT; ++tensorIdx) {
                tc.inputs.push_back({ops::ut::ParseDims(Trim(items[idx++])), ops::ut::ParseDims(Trim(items[idx++])),
                                     ops::ut::ParseGeDtype(Trim(items[idx++])),
                                     ops::ut::ParseGeFormat(Trim(items[idx++]))});
            }
            for (size_t tensorIdx = 0U; tensorIdx < OUTPUT_TENSOR_COUNT; ++tensorIdx) {
                tc.outputs.push_back({ops::ut::ParseDims(Trim(items[idx++])), ops::ut::ParseDims(Trim(items[idx++])),
                                      ops::ut::ParseGeDtype(Trim(items[idx++])),
                                      ops::ut::ParseGeFormat(Trim(items[idx++]))});
            }
            cases.push_back(tc);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulActivationQuantTilingCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedMatmulActivationQuantTilingUT {

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TestGroupedMatmulActivationQuantTiling950);

const vector<GroupedMatmulActivationQuantTilingCase> &GetAscend950Cases()
{
    static const vector<GroupedMatmulActivationQuantTilingCase> cases = LoadCases("Ascend950");
    return cases;
}

class TestGroupedMatmulActivationQuantTiling950
    : public testing::TestWithParam<GroupedMatmulActivationQuantTilingCase> {};

TEST_P(TestGroupedMatmulActivationQuantTiling950, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMM_ACTIVATION_QUANT_TILING_950, TestGroupedMatmulActivationQuantTiling950,
                         testing::ValuesIn(GetAscend950Cases()), MakeParamName);

} // namespace GroupedMatmulActivationQuantTilingUT
