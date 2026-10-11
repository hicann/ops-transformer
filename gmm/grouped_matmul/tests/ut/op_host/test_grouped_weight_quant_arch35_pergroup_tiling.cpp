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
 * \file test_grouped_weight_quant_arch35_pergroup_tiling.cpp
 * \brief CSV-driven unit tests for GroupedWeightQuant PerGroup arch35 tiling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr uint32_t DEFAULT_AIC_NUM = 32;

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

vector<gert::TilingContextPara::OpAttr> GetGroupedWeightQuantPergroupAttrs(bool transposeX, bool transposeWeight,
                                                                           int64_t groupType)
{
    int64_t splitItem = (groupType == -1) ? 0 : 3;
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

optiling::GMMCompileInfo MakeAscend950CompileInfo()
{
    return {
        DEFAULT_AIC_NUM,                                // aicNum
        DEFAULT_AIC_NUM * optiling::AIC_AIV_CORE_RATIO, // aivNum
        262144,                                         // ubSize
        524288,                                         // l1Size
        134217728,                                      // l2Size
        262144,                                         // l0CSize
        65536,                                          // l0ASize
        65536,                                          // l0BSize
        platform_ascendc::SocVersion::ASCEND950,        // socVersion
        NpuArch::DAV_3510,
    };
}

gert::TilingContextPara::TensorDescription MakeTensorDesc(const vector<int64_t>& dims, ge::DataType dtype,
                                                          ge::Format format = ge::FORMAT_ND)
{
    return {ops::ut::MakeGertStorageShape(dims, dims), dtype, format};
}

class GroupedWeightQuantPergroupArch35TilingTestParam {
public:
    void Prepare(optiling::GMMCompileInfo& compileInfo) const
    {
        compileInfo = {
            static_cast<uint32_t>(coreNum > 0 ? coreNum : DEFAULT_AIC_NUM),                                  // aicNum
            static_cast<uint32_t>((coreNum > 0 ? coreNum : DEFAULT_AIC_NUM) * optiling::AIC_AIV_CORE_RATIO), // aivNum
            262144,                                                                                          // ubSize
            524288,                                                                                          // l1Size
            134217728,                                                                                       // l2Size
            262144,                                                                                          // l0CSize
            65536,                                                                                           // l0ASize
            65536,                                                                                           // l0BSize
            platform_ascendc::SocVersion::ASCEND950, // socVersion
            NpuArch::DAV_3510,
        };
    }

    void InvokeTilingFunc(optiling::GMMCompileInfo& compileInfo) const
    {
        int64_t G = (groupSize > 0) ? (k / groupSize) : 0;

        // x shape
        vector<int64_t> xOriginDims = transposeX ? vector<int64_t>{k, m} : vector<int64_t>{m, k};
        gert::StorageShape xShape = ops::ut::MakeGertStorageShape(xOriginDims, xOriginDims);

        // weight shape
        ge::Format weightFmt = (weightFormat == "NZ") ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND;
        gert::StorageShape weightShape;
        if (groupType == -1) {
            // 多多多: weight is 2D [K, N] or [N, K]
            vector<int64_t> wDims = transposeWeight ? vector<int64_t>{n, k} : vector<int64_t>{k, n};
            weightShape = ops::ut::MakeGertStorageShape(wDims, wDims);
        } else {
            // 单单单 (or error cases): weight is 3D [E, K, N] or [E, N, K]
            vector<int64_t> wDims = transposeWeight ? vector<int64_t>{groupNum, n, k} : vector<int64_t>{groupNum, k, n};
            weightShape = ops::ut::MakeGertStorageShape(wDims, wDims);
        }

        // bias shape
        gert::StorageShape biasShape;
        if (hasBias) {
            if (groupType == -1) {
                biasShape = ops::ut::MakeGertStorageShape({n}, {n});
            } else {
                biasShape = ops::ut::MakeGertStorageShape({groupNum, n}, {groupNum, n});
            }
        } else {
            biasShape = MakeEmptyShape();
        }

        // antiquantScale shape (pergroup)
        gert::StorageShape antiquantScaleShape;
        if (groupType == -1) {
            // 多多多: [G, N]
            antiquantScaleShape = ops::ut::MakeGertStorageShape({G, n}, {G, n});
        } else {
            // 单单单: [E, G, N]
            antiquantScaleShape = ops::ut::MakeGertStorageShape({groupNum, G, n}, {groupNum, G, n});
        }

        // antiquantOffset shape
        gert::StorageShape antiquantOffsetShape;
        if (hasAntiquantOffset) {
            if (antiquantOffsetShapeMode == "bad_group") {
                if (groupType == -1) {
                    antiquantOffsetShape = ops::ut::MakeGertStorageShape({G + 1, n}, {G + 1, n});
                } else {
                    antiquantOffsetShape = ops::ut::MakeGertStorageShape({groupNum, G + 1, n}, {groupNum, G + 1, n});
                }
            } else if (antiquantOffsetShapeMode == "bad_rank") {
                antiquantOffsetShape = groupType == -1 ? ops::ut::MakeGertStorageShape({n}, {n}) :
                                                         ops::ut::MakeGertStorageShape({groupNum, n}, {groupNum, n});
            } else {
                antiquantOffsetShape = antiquantScaleShape;
            }
        } else {
            antiquantOffsetShape = MakeEmptyShape();
        }

        // groupList shape
        gert::StorageShape groupListShape;
        if (groupType == -1) {
            groupListShape = MakeEmptyShape();
        } else {
            groupListShape = ops::ut::MakeGertStorageShape({groupNum}, {groupNum});
        }

        gert::StorageShape yShape = ops::ut::MakeGertStorageShape({m, n}, {m, n});

        gert::TilingContextPara tilingContextPara(
            "GroupedMatmul",
            {
                {xShape, xDtype, ge::FORMAT_ND},                             // x
                {weightShape, weightDtype, weightFmt},                       // weight
                {biasShape, biasDtype, ge::FORMAT_ND},                       // bias
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},             // scale
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},             // offset
                {antiquantScaleShape, antiquantScaleDtype, ge::FORMAT_ND},   // antiquantScale
                {antiquantOffsetShape, antiquantOffsetDtype, ge::FORMAT_ND}, // antiquantOffset
                {groupListShape, ge::DT_INT64, ge::FORMAT_ND},               // groupList
                {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},             // perTokenScale
            },
            {{yShape, yDtype, ge::FORMAT_ND}},
            GetGroupedWeightQuantPergroupAttrs(transposeX, transposeWeight, groupType), &compileInfo, "3510",
            compileInfo.aicNum, compileInfo.ubSize);

        TilingInfo tilingInfo;
        bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
        ASSERT_EQ(tilingResult, result) << "prefix=" << prefix << ", case=" << caseName;
        if (!result) {
            return;
        }
        ASSERT_EQ(tilingInfo.blockNum, expectBlockDim) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingKey, expectTilingKey) << "prefix=" << prefix;
        ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMWeightQuantTilingData))
            << "prefix=" << prefix;
        const auto* tilingData =
            reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(tilingInfo.tilingData.get());
        const uint32_t expectedSingleFlag = groupType == -1 ? 0U : 1U;
        const uint32_t expectedGroupNum = groupType == -1 ? 1U : static_cast<uint32_t>(groupNum);
        EXPECT_EQ(tilingData->gmmWeightQuantParam.groupNum, expectedGroupNum) << "prefix=" << prefix;
        EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, static_cast<uint32_t>(groupSize)) << "prefix=" << prefix;
        EXPECT_EQ(static_cast<uint32_t>(tilingData->gmmWeightQuantParam.hasBias), static_cast<uint32_t>(hasBias))
            << "prefix=" << prefix;
        EXPECT_EQ(static_cast<uint32_t>(tilingData->gmmWeightQuantParam.singleX), expectedSingleFlag)
            << "prefix=" << prefix;
        EXPECT_EQ(static_cast<uint32_t>(tilingData->gmmWeightQuantParam.singleWeight), expectedSingleFlag)
            << "prefix=" << prefix;
        EXPECT_EQ(static_cast<uint32_t>(tilingData->gmmWeightQuantParam.singleY), expectedSingleFlag)
            << "prefix=" << prefix;
        EXPECT_EQ(static_cast<int32_t>(tilingData->gmmWeightQuantParam.groupType), static_cast<int32_t>(groupType))
            << "prefix=" << prefix;
        EXPECT_EQ(static_cast<uint32_t>(tilingData->gmmWeightQuantParam.groupListType), 0U) << "prefix=" << prefix;
        EXPECT_EQ(tilingData->gmmWeightQuantParam.kSize, static_cast<uint64_t>(k)) << "prefix=" << prefix;
        EXPECT_EQ(tilingData->gmmWeightQuantParam.nSize, static_cast<uint64_t>(n)) << "prefix=" << prefix;

        if (expectTilingData.empty()) {
            return;
        }

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
    string weightFormat;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType weightDtype = ge::DT_UNDEFINED;
    ge::DataType antiquantScaleDtype = ge::DT_UNDEFINED;
    ge::DataType antiquantOffsetDtype = ge::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DT_UNDEFINED;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    bool hasBias = false;
    bool hasAntiquantOffset = false;
    int64_t groupSize = 0;
    bool result = false;
    uint64_t expectBlockDim = 0;
    uint64_t expectTilingKey = 0;
    string expectTilingData;
    string antiquantOffsetShapeMode = "match";
};

vector<GroupedWeightQuantPergroupArch35TilingTestParam> GetParams(const string& socVersion)
{
    vector<GroupedWeightQuantPergroupArch35TilingTestParam> params;
    std::string csvPath = ops::ut::ResolveCsvPath("test_grouped_weight_quant_arch35_pergroup_tiling.csv",
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 26U) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            GroupedWeightQuantPergroupArch35TilingTestParam param;
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
            param.weightFormat = items[idx++];
            param.xDtype = ParseDtype(items[idx++]);
            param.weightDtype = ParseDtype(items[idx++]);
            param.antiquantScaleDtype = ParseDtype(items[idx++]);
            param.antiquantOffsetDtype = ParseDtype(items[idx++]);
            param.biasDtype = ParseDtype(items[idx++]);
            param.yDtype = ParseDtype(items[idx++]);
            param.hasBias = ParseBool(items[idx++]);
            param.hasAntiquantOffset = ParseBool(items[idx++]);
            param.groupSize = stoll(items[idx++]);
            param.result = ParseBool(items[idx++]);
            param.expectBlockDim = static_cast<uint64_t>(stoull(items[idx++]));
            param.expectTilingKey = static_cast<uint64_t>(stoull(items[idx++]));
            param.expectTilingData = items[idx++];
            if (idx < items.size()) {
                param.antiquantOffsetShapeMode = Trim(items[idx++]);
            }
            params.push_back(param);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return params;
}

string MakeParamName(const testing::TestParamInfo<GroupedWeightQuantPergroupArch35TilingTestParam>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedWeightQuantPergroupArch35TilingUT {

const vector<GroupedWeightQuantPergroupArch35TilingTestParam>& GetAscend950Params()
{
    static const vector<GroupedWeightQuantPergroupArch35TilingTestParam> params = GetParams("Ascend950");
    return params;
}

class TestGroupedWeightQuantPergroupArch35Tiling
    : public testing::TestWithParam<GroupedWeightQuantPergroupArch35TilingTestParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedWeightQuantPergroupArch35Tiling, generalTest)
{
    GetParam().Test();
}

// These regression cases use multi-instance inputs or cross-case comparisons that the scalar CSV rows cannot express.
TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiDifferentNkWritesEachMknToTilingData)
{
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 512}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 1024}, ge::DT_FLOAT16), // x1
        MakeTensorDesc({512, 256}, ge::DT_INT4),    // weight0
        MakeTensorDesc({1024, 384}, ge::DT_INT4),   // weight1
        MakeTensorDesc({256}, ge::DT_FLOAT16),      // bias0
        MakeTensorDesc({384}, ge::DT_FLOAT16),      // bias1
        MakeTensorDesc({4, 256}, ge::DT_FLOAT16),   // antiquantScale0, groupSize=512/4=128
        MakeTensorDesc({8, 384}, ge::DT_FLOAT16),   // antiquantScale1, groupSize=1024/8=128
        MakeTensorDesc({4, 256}, ge::DT_FLOAT16),   // antiquantOffset0
        MakeTensorDesc({8, 384}, ge::DT_FLOAT16),   // antiquantOffset1
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 384}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 2, 0, 0, 2, 2, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMWeightQuantTilingData));
    const auto* tilingData =
        reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(tilingInfo.tilingData.get());

    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupNum, 2U);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, 128U);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.kSize, 1024U);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.nSize, 384U);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleX, 0U);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleWeight, 0U);
    EXPECT_EQ(tilingData->gmmWeightQuantParam.singleY, 0U);
    EXPECT_EQ(tilingData->gmmArray.mList[0], 128);
    EXPECT_EQ(tilingData->gmmArray.kList[0], 512);
    EXPECT_EQ(tilingData->gmmArray.nList[0], 256);
    EXPECT_EQ(tilingData->gmmArray.mList[1], 64);
    EXPECT_EQ(tilingData->gmmArray.kList[1], 1024);
    EXPECT_EQ(tilingData->gmmArray.nList[1], 384);
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiRejectsFirstOffsetShapeMismatch)
{
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 512}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 1024}, ge::DT_FLOAT16), // x1
        MakeTensorDesc({512, 256}, ge::DT_INT4),    // weight0
        MakeTensorDesc({1024, 384}, ge::DT_INT4),   // weight1
        MakeTensorDesc({4, 256}, ge::DT_FLOAT16),   // antiquantScale0
        MakeTensorDesc({8, 384}, ge::DT_FLOAT16),   // antiquantScale1
        MakeTensorDesc({5, 256}, ge::DT_FLOAT16),   // antiquantOffset0, wrong G
        MakeTensorDesc({8, 384}, ge::DT_FLOAT16),   // antiquantOffset1
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 384}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 2, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiKEqualsGroupSizeStillUsesPergroup)
{
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 128}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 128}, ge::DT_FLOAT16),  // x1
        MakeTensorDesc({128, 256}, ge::DT_INT4),    // weight0
        MakeTensorDesc({128, 384}, ge::DT_INT4),    // weight1
        MakeTensorDesc({1, 256}, ge::DT_FLOAT16),   // antiquantScale0, groupSize=128/1=128
        MakeTensorDesc({1, 384}, ge::DT_FLOAT16),   // antiquantScale1, groupSize=128/1=128
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 384}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 0, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMWeightQuantTilingData));
    const auto* tilingData =
        reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, 128U);
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiRejectsPerChannelThenPerGroupScale)
{
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 128}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 128}, ge::DT_FLOAT16),  // x1
        MakeTensorDesc({128, 256}, ge::DT_INT4),    // weight0
        MakeTensorDesc({128, 384}, ge::DT_INT4),    // weight1
        MakeTensorDesc({256}, ge::DT_FLOAT16),      // antiquantScale0 perchannel
        MakeTensorDesc({1, 384}, ge::DT_FLOAT16),   // antiquantScale1 pergroup
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 384}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 0, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiRejectsPerGroupThenPerChannelScale)
{
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 128}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 128}, ge::DT_FLOAT16),  // x1
        MakeTensorDesc({128, 256}, ge::DT_INT4),    // weight0
        MakeTensorDesc({128, 384}, ge::DT_INT4),    // weight1
        MakeTensorDesc({1, 256}, ge::DT_FLOAT16),   // antiquantScale0 pergroup
        MakeTensorDesc({384}, ge::DT_FLOAT16),      // antiquantScale1 perchannel
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 384}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 0, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiPerChannelSucceedsWithGroupSizeZero)
{
    // 1D antiquantScale [N_i] -> perchannel mode in multi-multi-multi bf16/fp16-int4/int32 ND.
    // Verifies the new CheckMultiA16W4PerChannelShape path succeeds and groupSize stays 0.
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 512}, ge::DT_FLOAT16),  // x1
        MakeTensorDesc({256, 128}, ge::DT_INT4),    // weight0 [K,N]
        MakeTensorDesc({512, 256}, ge::DT_INT4),    // weight1 [K,N]
        MakeTensorDesc({128}, ge::DT_FLOAT16),      // antiquantScale0 [N] perchannel
        MakeTensorDesc({256}, ge::DT_FLOAT16),      // antiquantScale1 [N] perchannel
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 128}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 256}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 0, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMWeightQuantTilingData));
    const auto* tilingData =
        reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, 0U);
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiRejectsInconsistentGroupSize)
{
    // tensor0: K=256, G=2 -> groupSize=128; tensor1: K=512, G=8 -> groupSize=64.
    // The two tensors derive different groupSize values and must be rejected.
    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 512}, ge::DT_FLOAT16),  // x1
        MakeTensorDesc({256, 128}, ge::DT_INT4),    // weight0 [K,N]
        MakeTensorDesc({512, 256}, ge::DT_INT4),    // weight1 [K,N]
        MakeTensorDesc({2, 128}, ge::DT_FLOAT16),   // antiquantScale0, groupSize=256/2=128
        MakeTensorDesc({8, 256}, ge::DT_FLOAT16),   // antiquantScale1, groupSize=512/8=64
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 128}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 256}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 0, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmul", inputDescs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, multiMultiMultiKEqualsGroupSizePergroupKeyDiffersFromPerChannel)
{
    // Regression for the K==groupSize ambiguity bug: with identical M/N/K shapes,
    // a 2D antiquantScale [1, N] (pergroup, G=1, K==gs) must produce a different
    // tilingKey from a 1D antiquantScale [N] (perchannel). Before the fix, the
    // K==gs pergroup case was misclassified as perchannel (groupSize stayed 0),
    // producing the same PER_CHANNEL tilingKey for both.
    auto compileInfo = MakeAscend950CompileInfo();

    // Common tensor shapes for both modes.
    vector<gert::TilingContextPara::TensorDescription> commonInputPrefix = {
        MakeTensorDesc({128, 128}, ge::DT_FLOAT16), // x0
        MakeTensorDesc({64, 128}, ge::DT_FLOAT16),  // x1
        MakeTensorDesc({128, 256}, ge::DT_INT4),    // weight0 [K,N]
        MakeTensorDesc({128, 384}, ge::DT_INT4),    // weight1 [K,N]
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({128, 256}, ge::DT_FLOAT16),
        MakeTensorDesc({64, 384}, ge::DT_FLOAT16),
    };
    vector<uint32_t> inputInstanceNum = {2, 2, 0, 0, 0, 2, 0, 0, 0};
    vector<uint32_t> outputInstanceNum = {2};

    // Pergroup: 2D scale [1, N], K=128, G=1, groupSize=128/1=128.
    vector<gert::TilingContextPara::TensorDescription> pergroupInputs(commonInputPrefix);
    pergroupInputs.push_back(MakeTensorDesc({1, 256}, ge::DT_FLOAT16)); // antiquantScale0
    pergroupInputs.push_back(MakeTensorDesc({1, 384}, ge::DT_FLOAT16)); // antiquantScale1
    gert::TilingContextPara pergroupCtx(
        "GroupedMatmul", pergroupInputs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);
    TilingInfo pergroupInfo;
    ASSERT_TRUE(ExecuteTiling(pergroupCtx, pergroupInfo));
    ASSERT_EQ(pergroupInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMWeightQuantTilingData));
    const auto* pergroupData =
        reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(pergroupInfo.tilingData.get());
    ASSERT_EQ(pergroupData->gmmWeightQuantParam.groupSize, 128U);

    // Perchannel: 1D scale [N], same M/N/K.
    vector<gert::TilingContextPara::TensorDescription> perchannelInputs(commonInputPrefix);
    perchannelInputs.push_back(MakeTensorDesc({256}, ge::DT_FLOAT16)); // antiquantScale0
    perchannelInputs.push_back(MakeTensorDesc({384}, ge::DT_FLOAT16)); // antiquantScale1
    gert::TilingContextPara perchannelCtx(
        "GroupedMatmul", perchannelInputs, outputDescs, GetGroupedWeightQuantPergroupAttrs(false, false, -1),
        inputInstanceNum, outputInstanceNum, &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);
    TilingInfo perchannelInfo;
    ASSERT_TRUE(ExecuteTiling(perchannelCtx, perchannelInfo));
    const auto* perchannelData =
        reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(perchannelInfo.tilingData.get());
    ASSERT_EQ(perchannelData->gmmWeightQuantParam.groupSize, 0U);

    // The core assertion: pergroup tilingKey must differ from perchannel tilingKey.
    EXPECT_NE(pergroupInfo.tilingKey, perchannelInfo.tilingKey);
}

TEST_F(TestGroupedWeightQuantPergroupArch35Tiling, singleSingleSingleInt32PackedWeightOutputUsesLogicalN)
{
    constexpr int64_t m = 256;
    constexpr int64_t k = 1024;
    constexpr int64_t logicalN = 512;
    constexpr int64_t packedN = logicalN / optiling::FP4_PER_FP32;
    constexpr int64_t groupNum = 2;
    constexpr int64_t groupSize = 128;
    constexpr int64_t scaleGroupNum = k / groupSize;

    auto compileInfo = MakeAscend950CompileInfo();
    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({m, k}, ge::DT_FLOAT16), // x
        {ops::ut::MakeGertStorageShape({groupNum, k, logicalN}, {groupNum, k, packedN}), ge::DT_INT32,
         ge::FORMAT_ND},                                                     // packed int4 weight
        MakeTensorDesc({groupNum, logicalN}, ge::DT_FLOAT16),                // bias
        {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},                     // scale
        {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},                     // offset
        MakeTensorDesc({groupNum, scaleGroupNum, logicalN}, ge::DT_FLOAT16), // antiquantScale
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},                   // antiquantOffset
        MakeTensorDesc({groupNum}, ge::DT_INT64),                            // groupList
        {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},                     // perTokenScale
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({m, logicalN}, ge::DT_FLOAT16),
    };
    gert::TilingContextPara tilingContextPara("GroupedMatmul", inputDescs, outputDescs,
                                              GetGroupedWeightQuantPergroupAttrs(false, false, 0), &compileInfo, "3510",
                                              compileInfo.aicNum, compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMWeightQuantTilingData));
    const auto* tilingData =
        reinterpret_cast<const GroupedMatmulTilingData::GMMWeightQuantTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->gmmWeightQuantParam.nSize, static_cast<uint64_t>(logicalN));
    EXPECT_EQ(tilingData->gmmWeightQuantParam.groupSize, static_cast<uint32_t>(groupSize));
}

INSTANTIATE_TEST_SUITE_P(GROUPED_WQ_PERGROUP_950, TestGroupedWeightQuantPergroupArch35Tiling,
                         testing::ValuesIn(GetAscend950Params()), MakeParamName);

} // namespace GroupedWeightQuantPergroupArch35TilingUT
