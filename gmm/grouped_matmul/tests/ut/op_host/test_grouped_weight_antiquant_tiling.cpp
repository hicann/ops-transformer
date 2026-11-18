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
 * \file test_grouped_weight_antiquant_tiling.cpp
 * \brief CSV-driven unit tests for grouped weight antiquantization tiling.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_matmul_asw_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using Attr = gert::TilingContextPara::OpAttr;
struct GroupedWeightAntiquantTilingTestParam {
    std::string socVersion;
    std::string caseName;
    bool enable = true;
    std::string prefix;
    uint32_t coreNum = 64;
    int64_t m = 257;
    int64_t groupNum = 4;
    int64_t splitItem = 3;
    ge::DataType yDtype = ge::DT_FLOAT16;

    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_INT8;
    ge::DataType antiquantScaleDtype = ge::DT_FLOAT16;
    ge::DataType antiquantOffsetDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType groupListDtype = ge::DT_INT64;
    ge::Format weightFormat = ge::FORMAT_ND;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format antiquantScaleFormat = ge::FORMAT_ND;
    bool hasQuantScale = false;
    bool hasQuantOffset = false;
    bool hasPerTokenScale = false;
    bool transposeWeight = false;
    bool transposeX = false;
    bool hasAntiquantOffset = false;
    bool hasBias = false;
    int64_t groupType = 0;
    int64_t groupListType = 1;
    int64_t k = 529;
    int64_t n = 273;
    int64_t antiquantScaleN = 273;
    int64_t antiquantOffsetN = 273;
    int64_t groupListLen = 4;
    int64_t yN = 273;
    int64_t actType = 0;
    bool result = false;
    bool compareNd = false;
    uint64_t expectBlockDim = 0;
    uint64_t expectTilingKey = 0;
    int64_t expectBaseM = 0;
    int64_t expectBaseN = 0;
    int64_t expectBaseK = 0;
    int64_t expectStepKa = 0;
    int64_t expectStepKb = 0;
    uint64_t expectL1UsedBytes = 0;

    void Test() const;
};

TensorDesc Desc(const std::vector<int64_t>& shape, ge::DataType type, ge::Format format = ge::FORMAT_ND)
{
    return {ops::ut::MakeGertStorageShape(shape), type, format};
}
bool RunGroupedWeightAntiquantTiling(const GroupedWeightAntiquantTilingTestParam& c, TilingInfo& out)
{
    // Route through the real architecture dispatcher, not just the ASW helper.
    constexpr uint64_t KIB = 1024;
    optiling::GMMCompileInfo compileInfo = {c.coreNum,        c.coreNum, 376 * KIB,
                                            512 * KIB,        196608,    256 * KIB,
                                            64 * KIB,         64 * KIB,  platform_ascendc::SocVersion::ASCEND960DT,
                                            NpuArch::DAV_9201};
    std::vector<TensorDesc> inputs{Desc({c.m, c.k}, c.xDtype, c.xFormat),
                                   Desc(c.transposeWeight ? std::vector<int64_t>{c.groupNum, c.n, c.k} :
                                                            std::vector<int64_t>{c.groupNum, c.k, c.n},
                                        c.weightDtype, c.weightFormat)};
    if (c.hasBias) {
        inputs.push_back(Desc({c.groupNum, c.n}, c.biasDtype));
    }
    if (c.hasQuantScale) {
        inputs.push_back(Desc({c.groupNum, c.n}, ge::DT_FLOAT));
    }
    if (c.hasQuantOffset) {
        inputs.push_back(Desc({c.groupNum, c.n}, ge::DT_FLOAT));
    }
    inputs.push_back(Desc({c.groupNum, c.antiquantScaleN}, c.antiquantScaleDtype, c.antiquantScaleFormat));
    if (c.hasAntiquantOffset) {
        inputs.push_back(Desc({c.groupNum, c.antiquantOffsetN}, c.antiquantOffsetDtype));
    }
    inputs.push_back(Desc({c.groupListLen}, c.groupListDtype));
    if (c.hasPerTokenScale) {
        inputs.push_back(Desc({c.m}, ge::DT_FLOAT));
    }
    std::vector<Attr> attrs{{"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.splitItem)},
                            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(c.transposeWeight)},
                            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(c.transposeX)},
                            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.groupType)},
                            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.groupListType)},
                            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.actType)},
                            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})}};
    gert::TilingContextPara context("GroupedMatmul", inputs, {Desc({c.m, c.yN}, c.yDtype)}, attrs,
                                    {1U, 1U, c.hasBias ? 1U : 0U, c.hasQuantScale ? 1U : 0U, c.hasQuantOffset ? 1U : 0U,
                                     1U, c.hasAntiquantOffset ? 1U : 0U, 1U, c.hasPerTokenScale ? 1U : 0U},
                                    {1U}, &compileInfo, "9201", compileInfo.aicNum, compileInfo.ubSize);
    // ExecuteTiling supplies a non-null PlatformInfo; keep its resources consistent with CompileInfo.
    context.socInfoString_ =
        std::string(R"({"hardware_info":{"CORE_NUM":)") + std::to_string(compileInfo.aicNum) + R"(,"cube_core_cnt":)" +
        std::to_string(compileInfo.aicNum) + R"(,"vector_core_cnt":)" + std::to_string(compileInfo.aivNum) +
        R"(,"UB_SIZE":)" + std::to_string(compileInfo.ubSize) + R"(,"L1_SIZE":)" + std::to_string(compileInfo.l1Size) +
        R"(,"L2_SIZE":)" + std::to_string(compileInfo.l2Size) + R"(,"L0A_SIZE":)" +
        std::to_string(compileInfo.l0ASize) + R"(,"L0B_SIZE":)" + std::to_string(compileInfo.l0BSize) +
        R"(,"L0C_SIZE":)" + std::to_string(compileInfo.l0CSize) + "}}";
    return ExecuteTiling(context, out);
}
void GroupedWeightAntiquantTilingTestParam::Test() const
{
    SCOPED_TRACE(testing::Message() << "prefix=" << prefix << ", case=" << caseName << ", K=" << k << ", N=" << n
                                    << ", trans=" << transposeWeight << ", bias=" << hasBias
                                    << ", offset=" << hasAntiquantOffset << ", groupListType=" << groupListType);
    TilingInfo info;
    ASSERT_EQ(RunGroupedWeightAntiquantTiling(*this, info), result);
    if (!result) {
        return;
    }
    EXPECT_EQ(info.blockNum, expectBlockDim);
    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), expectTilingKey);
    using Tiling = GroupedMatmulTilingData::GMMWeightQuantAswTilingData;
    ASSERT_EQ(info.tilingDataSize, sizeof(Tiling));
    const auto* data = reinterpret_cast<const Tiling*>(info.tilingData.get());
    EXPECT_EQ(data->gmmWeightQuantAswParams.groupNum, groupNum);
    EXPECT_EQ(data->gmmWeightQuantAswParams.groupListType, groupListType);
    EXPECT_EQ(data->gmmWeightQuantAswParams.hasBias, hasBias);
    const auto& mm = data->mmTilingData;
    EXPECT_EQ(mm.usedCoreNum, coreNum);
    EXPECT_EQ(mm.baseM, expectBaseM);
    EXPECT_EQ(mm.baseN, expectBaseN);
    EXPECT_EQ(mm.baseK, expectBaseK);
    EXPECT_EQ(mm.stepKa, expectStepKa);
    EXPECT_EQ(mm.stepKb, expectStepKb);
    constexpr uint64_t KIB = 1024;
    constexpr uint64_t BUFFER_COUNT = 2;
    constexpr uint64_t ELEMENT_BYTES = 2;
    constexpr uint64_t ACCUMULATOR_BYTES = 4;
    constexpr uint64_t BIAS_BUFFER_BYTES = 8 * KIB;
    const uint64_t l1Bytes =
        BUFFER_COUNT * (mm.baseM + mm.baseN) * mm.baseK * mm.stepKb * ELEMENT_BYTES + (hasBias ? BIAS_BUFFER_BYTES : 0);
    EXPECT_EQ(l1Bytes, expectL1UsedBytes);
    EXPECT_LE(l1Bytes, 512 * KIB);
    EXPECT_LE(BUFFER_COUNT * mm.baseM * mm.baseK * ELEMENT_BYTES, 64 * KIB);
    EXPECT_LE(BUFFER_COUNT * mm.baseN * mm.baseK * ELEMENT_BYTES, 64 * KIB);
    EXPECT_LE(uint64_t(mm.baseM) * mm.baseN * ACCUMULATOR_BYTES, 256 * KIB);
    if (compareNd) {
        auto ndCase = *this;
        ndCase.weightFormat = ge::FORMAT_ND;
        TilingInfo ndInfo;
        ASSERT_TRUE(RunGroupedWeightAntiquantTiling(ndCase, ndInfo));
        EXPECT_EQ(info.tilingKey, ndInfo.tilingKey);
        EXPECT_EQ(info.blockNum, ndInfo.blockNum);
        EXPECT_EQ(info.tilingDataSize, ndInfo.tilingDataSize);
    }
}

std::vector<GroupedWeightAntiquantTilingTestParam> GetParams(const std::string& socVersion)
{
    std::vector<GroupedWeightAntiquantTilingTestParam> params;
    const auto csvPath = ops::ut::ResolveCsvPath("test_grouped_weight_antiquant_tiling.csv",
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
            if (line !=
                "socVersion,caseName,enable,prefix,coreNum,m,k,n,groupNum,transposeX,transposeWeight,groupType,"
                "splitItem,weightFormat,xDtype,weightDtype,antiquantScaleDtype,antiquantOffsetDtype,biasDtype,"
                "yDtype,hasBias,hasAntiquantOffset,result,blockDim,tilingKey,baseM,baseN,baseK,stepKa,stepKb,"
                "l1UsedBytes,groupListType,groupListLen,groupListDtype,antiquantScaleN,antiquantOffsetN,yN,"
                "actType,xFormat,antiquantScaleFormat,hasQuantScale,hasQuantOffset,hasPerTokenScale,compareNd") {
                ADD_FAILURE() << csvPath << ":" << lineNo << ": unexpected CSV header";
                return {};
            }
            hasHeader = true;
            continue;
        }
        std::vector<std::string> items;
        ops::ut::SplitStr2Vec(line, ",", items);
        constexpr size_t FIELD_COUNT = 44;
        if (items.size() != FIELD_COUNT) {
            ADD_FAILURE() << csvPath << ":" << lineNo << ": expected " << FIELD_COUNT << " columns";
            continue;
        }
        const auto caseName = ops::ut::Trim(items[1]);
        try {
            GroupedWeightAntiquantTilingTestParam param;
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
            param.coreNum = static_cast<uint32_t>(std::stoul(items[idx++]));
            param.m = std::stoll(items[idx++]);
            param.k = std::stoll(items[idx++]);
            param.n = std::stoll(items[idx++]);
            param.groupNum = std::stoll(items[idx++]);
            param.transposeX = ops::ut::ParseBool(items[idx++]);
            param.transposeWeight = ops::ut::ParseBool(items[idx++]);
            param.groupType = std::stoll(items[idx++]);
            param.splitItem = std::stoll(items[idx++]);
            param.weightFormat = ops::ut::ParseGeFormat(items[idx++]);
            param.xDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.weightDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.antiquantScaleDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.antiquantOffsetDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.biasDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.yDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.hasBias = ops::ut::ParseBool(items[idx++]);
            param.hasAntiquantOffset = ops::ut::ParseBool(items[idx++]);
            param.result = ops::ut::ParseBool(items[idx++]);
            param.expectBlockDim = param.result ? std::stoull(items[idx]) : 0;
            ++idx;
            param.expectTilingKey = param.result ? std::stoull(items[idx]) : 0;
            ++idx;
            param.expectBaseM = param.result ? std::stoll(items[idx]) : 0;
            ++idx;
            param.expectBaseN = param.result ? std::stoll(items[idx]) : 0;
            ++idx;
            param.expectBaseK = param.result ? std::stoll(items[idx]) : 0;
            ++idx;
            param.expectStepKa = param.result ? std::stoll(items[idx]) : 0;
            ++idx;
            param.expectStepKb = param.result ? std::stoll(items[idx]) : 0;
            ++idx;
            param.expectL1UsedBytes = param.result ? std::stoull(items[idx]) : 0;
            ++idx;
            param.groupListType = std::stoll(items[idx++]);
            param.groupListLen = std::stoll(items[idx++]);
            param.groupListDtype = ops::ut::ParseGeDtype(items[idx++]);
            param.antiquantScaleN = std::stoll(items[idx++]);
            param.antiquantOffsetN = std::stoll(items[idx++]);
            param.yN = std::stoll(items[idx++]);
            param.actType = std::stoll(items[idx++]);
            param.xFormat = ops::ut::ParseGeFormat(items[idx++]);
            param.antiquantScaleFormat = ops::ut::ParseGeFormat(items[idx++]);
            param.hasQuantScale = ops::ut::ParseBool(items[idx++]);
            param.hasQuantOffset = ops::ut::ParseBool(items[idx++]);
            param.hasPerTokenScale = ops::ut::ParseBool(items[idx++]);
            param.compareNd = ops::ut::ParseBool(items[idx++]);
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

const std::vector<GroupedWeightAntiquantTilingTestParam>& GetAscend960DTParams()
{
    static const auto params = GetParams("Ascend960DT");
    return params;
}

std::string MakeParamName(const testing::TestParamInfo<GroupedWeightAntiquantTilingTestParam>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedWeightAntiquantTilingUT {
class TestGroupedWeightAntiquantTiling : public testing::TestWithParam<GroupedWeightAntiquantTilingTestParam> {};

TEST_P(TestGroupedWeightAntiquantTiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_WEIGHT_ANTIQUANT_960DT, TestGroupedWeightAntiquantTiling,
                         testing::ValuesIn(GetAscend960DTParams()), MakeParamName);
} // namespace GroupedWeightAntiquantTilingUT
