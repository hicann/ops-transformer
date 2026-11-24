// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "tiling_case_executor.h"
#include "dual_level_grouped_matmul_ut_param.h"

namespace DualLevelGmmUT {

class DualLevelGmmTilingTest : public testing::TestWithParam<DualLevelGmmTilingUtParam> {};

// CSV-driven batch tiling cases; add rows to test_dual_level_grouped_matmul_tiling.csv to extend coverage.
TEST_P(DualLevelGmmTilingTest, csvDrivenCase)
{
    const auto& param = GetParam();
    static optiling::DualLevelGroupedMatmulCompileInfo compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs{
        param.x,     param.xLevel0Scale, param.perTokenScale, param.weight, param.weightLevel0Scale,
        param.scale, param.groupList,    param.BuildBias()};
    std::vector<gert::TilingContextPara::TensorDescription> outputs{param.y};
    const uint64_t tilingDataSize = param.shortTilingBuffer ? sizeof(DualLevelGmmTilingData) - 1 : 4096;
    std::string socInfo = GetDlgmmSocInfoString();
    const auto replaceCoreCount = [&socInfo](const std::string& key, uint32_t count) {
        const auto start = socInfo.find(key) + key.size();
        const auto end = socInfo.find_first_not_of("0123456789", start);
        socInfo.replace(start, end - start, std::to_string(count));
    };
    replaceCoreCount("\"CORE_NUM\":", param.aicNum);
    replaceCoreCount("\"cube_core_cnt\":", param.aicNum);
    replaceCoreCount("\"vector_core_cnt\":", param.aivNum);
    auto attrs = param.BuildAttrs<gert::TilingContextPara>();
    if (!param.level0GroupSize.empty()) {
        attrs.push_back(
            {"level0_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(std::stoll(param.level0GroupSize))});
    }
    if (!param.level1GroupSize.empty()) {
        attrs.push_back(
            {"level1_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(std::stoll(param.level1GroupSize))});
    }
    gert::TilingContextPara para("DualLevelGroupedMatmul", inputs, outputs, attrs, &compileInfo, param.socVersion,
                                 socInfo, tilingDataSize);

    TilingInfo info;
    if (param.expectResult != ge::GRAPH_SUCCESS) {
        EXPECT_FALSE(ExecuteTiling(para, info));
        return;
    }
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(param.expectTilingKey));
    ASSERT_EQ(info.tilingDataSize, sizeof(DualLevelGmmTilingData));
    DualLevelGmmTilingData data;
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    ASSERT_GE(param.x.shape_.GetOriginShape().GetDimNum(), 2U);
    EXPECT_EQ(data.m, param.x.shape_.GetOriginShape().GetDim(0));
    EXPECT_EQ(data.k, param.x.shape_.GetOriginShape().GetDim(1));
    EXPECT_EQ(data.level0Size, 512U);
    EXPECT_EQ(data.groupListType, static_cast<uint64_t>(param.groupListType));
    EXPECT_EQ(data.hasBias, param.hasBias ? 1U : 0U);
    EXPECT_EQ(data.coreNum, param.expectCoreNum);
}

INSTANTIATE_TEST_SUITE_P(
    DUAL_LEVEL_GMM_TILING_CSV, DualLevelGmmTilingTest,
    testing::ValuesIn(GetCasesFromCsv<DualLevelGmmTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<DualLevelGmmTilingUtParam>);

} // namespace DualLevelGmmUT
