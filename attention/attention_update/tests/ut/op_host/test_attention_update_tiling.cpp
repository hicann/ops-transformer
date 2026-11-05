/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstring>

#include "../../../op_host/decode_update_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024UL;

gert::TilingContextPara BuildCase(optiling::DecodeUpdateCompileInfo& compileInfo, ge::DataType dtype,
                                  int64_t updateType, int64_t sp, const gert::StorageShape& lseShape,
                                  const gert::StorageShape& localOutShape, const gert::StorageShape& outShape,
                                  const gert::StorageShape& lseOutShape = {{}, {}}, uint64_t rowStride = 0)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outShape, dtype, ge::FORMAT_ND},
    };

    for (int64_t i = 0; i < sp; ++i) {
        inputs.push_back({lseShape, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    for (int64_t i = 0; i < sp; ++i) {
        gert::TilingContextPara::TensorDescription go(localOutShape, dtype, ge::FORMAT_ND);
        if (rowStride != 0) {
            go.stride_ = gert::Stride({static_cast<int64_t>(rowStride), 1});
            go.hasStride_ = true;
        }
        inputs.push_back(go);
    }
    if (updateType == 1) {
        outputs.push_back({lseOutShape, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    return gert::TilingContextPara(
        "AttentionUpdate", inputs, outputs,
        {
            {"update_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(updateType)},
            {"sp", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sp)},
        },
        {static_cast<uint32_t>(sp), static_cast<uint32_t>(sp)}, {1, updateType == 1 ? 1U : 0U}, &compileInfo,
        compileInfo.is_ascendc ? "Ascend950" : "Ascend910B", compileInfo.coreNum, compileInfo.ubSize);
}

optiling::DecodeUpdateCompileInfo LegacyCompileInfo(uint32_t coreNum = 40, uint64_t ubSize = 196608)
{
    optiling::DecodeUpdateCompileInfo compileInfo{};
    compileInfo.coreNum = coreNum;
    compileInfo.ubSize = ubSize;
    compileInfo.is_ascendc = false;
    return compileInfo;
}

optiling::DecodeUpdateCompileInfo AscendCCompileInfo()
{
    optiling::DecodeUpdateCompileInfo compileInfo{};
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.is_ascendc = true;
    return compileInfo;
}
} // namespace

class AttentionUpdateTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionUpdateTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionUpdateTiling TearDown" << std::endl;
    }
};

TEST_F(AttentionUpdateTilingTest, LegacyFloatUpdateType0Sp1)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara =
        BuildCase(compileInfo, ge::DT_FLOAT, 0, 1, {{128}, {128}}, {{128, 256}, {128, 256}}, {{128, 256}, {128, 256}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010, "12884901892 137438953480 256 549755813889 ",
                    {SYS_WORKSPACE_SIZE});
}

TEST_F(AttentionUpdateTilingTest, AscendCFloatUpdateType1Sp1)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT, 1, 1, {{128}, {128}}, {{128, 8}, {128, 8}},
                                       {{128, 8}, {128, 8}}, {{128}, {128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20001);
}

TEST_F(AttentionUpdateTilingTest, AscendCFloat16UpdateType1Sp1)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT16, 1, 1, {{128}, {128}}, {{128, 8}, {128, 8}},
                                       {{128, 8}, {128, 8}}, {{128}, {128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20001);
}

TEST_F(AttentionUpdateTilingTest, AscendCFloat16UpdateType1Sp2)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT16, 1, 2, {{128}, {128}}, {{128, 8}, {128, 8}},
                                       {{128, 8}, {128, 8}}, {{128}, {128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20001);
}

TEST_F(AttentionUpdateTilingTest, AscendCInt32Rejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_INT32, 1, 2, {{128}, {128}}, {{128, 8}, {128, 8}},
                                       {{128, 8}, {128, 8}}, {{128}, {128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AttentionUpdateTilingTest, AscendCHDimTooLargeRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT16, 1, 2, {{128}, {128}}, {{128, 1024}, {128, 1024}},
                                       {{128, 1024}, {128, 1024}}, {{128}, {128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AttentionUpdateTilingTest, AscendCLargeTotalLength)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCase(compileInfo, ge::DT_FLOAT16, 1, 2, {{12800000}, {12800000}}, {{12800000, 8}, {12800000, 8}},
                  {{12800000, 8}, {1280000, 8}}, {{12800000}, {12800000}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 20001);
}

// 测试A2A3环境下sp为16的场景，不输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyFloatUpdateType0Sp16)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara =
        BuildCase(compileInfo, ge::DT_FLOAT, 0, 16, {{128}, {128}}, {{128, 128}, {128, 128}}, {{128, 128}, {128, 128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010);
}

// 测试A2A3环境下sp为16的场景且fp16，输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyFloat16UpdateType1Sp16)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT16, 1, 16, {{128}, {128}}, {{128, 128}, {128, 128}},
                                       {{128, 128}, {128, 128}}, {{128}, {128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10020);
}

// 测试A2A3环境下sp大于16的场景，输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyFloatUpdateType1Sp32)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT, 1, 32, {{64}, {64}}, {{64, 128}, {64, 128}},
                                       {{64, 128}, {64, 128}}, {{64}, {64}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010);
}

// 测试A2A3环境下sp大于16的场景且fp16，输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyFloat16UpdateType1Sp64)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT16, 1, 64, {{64}, {64}}, {{64, 128}, {64, 128}},
                                       {{64, 128}, {64, 128}}, {{64}, {64}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10020);
}

// 测试A2A3环境下sp为64的场景且bf16，输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyBf16UpdateType1Sp64)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_BF16, 1, 64, {{64}, {64}}, {{64, 128}, {64, 128}},
                                       {{64, 128}, {64, 128}}, {{64}, {64}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10030);
}

// 测试A2A3环境下sp为128的场景且fp16，输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyFloat16UpdateType1Sp128)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT16, 1, 128, {{64}, {64}}, {{64, 128}, {64, 128}},
                                       {{64, 128}, {64, 128}}, {{64}, {64}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10020);
}

// 测试A2A3环境下sp为最大值128而且totallength超大，不输出lse的tiling能否正常运行
TEST_F(AttentionUpdateTilingTest, LegacyFloatUpdateType0Sp128LargeTotalLength)
{
    auto compileInfo = LegacyCompileInfo();
    auto tilingContextPara = BuildCase(compileInfo, ge::DT_FLOAT, 0, 128, {{128000}, {128000}},
                                       {{128000, 128}, {128000, 128}}, {{128000, 128}, {128000, 128}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 10010);
}

// ==================== OP_LOGE 分支覆盖用例（错误码整改配套） ====================

namespace {
using TensorSpec = std::pair<ge::DataType, gert::StorageShape>;

gert::TilingContextPara BuildCustomCase(optiling::DecodeUpdateCompileInfo& compileInfo,
                                        const std::vector<TensorSpec>& lseTensors,
                                        const std::vector<TensorSpec>& goTensors,
                                        const std::vector<TensorSpec>& outTensors,
                                        std::vector<gert::TilingContextPara::OpAttr> attrs,
                                        uint64_t coreNumOverride = 0, uint64_t ubSizeOverride = 0)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (auto& t : lseTensors) {
        inputs.push_back({t.second, t.first, ge::FORMAT_ND});
    }
    for (auto& t : goTensors) {
        inputs.push_back({t.second, t.first, ge::FORMAT_ND});
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    for (auto& t : outTensors) {
        outputs.push_back({t.second, t.first, ge::FORMAT_ND});
    }
    uint64_t coreNum = coreNumOverride > 0 ? coreNumOverride : compileInfo.coreNum;
    uint64_t ubSize = ubSizeOverride > 0 ? ubSizeOverride : compileInfo.ubSize;
    return gert::TilingContextPara("AttentionUpdate", inputs, outputs, attrs, &compileInfo, "Ascend910B", coreNum,
                                   ubSize);
}

std::vector<gert::TilingContextPara::OpAttr> DefaultAttrs(int64_t updateType, int64_t sp)
{
    return {{"update_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(updateType)},
            {"sp", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sp)}};
}
} // namespace

// 分支: CheckInputDtype - lse dtype必须为FLOAT (OP_LOGE_FOR_INVALID_DTYPE "lse")
TEST_F(AttentionUpdateTilingTest, AscendCLseDtypeFp16Rejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT16, {{128}, {128}}}}, {{ge::DT_FLOAT16, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT16, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 1));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputDtype - go各tensor dtype必须一致 (OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON "go[i]")
TEST_F(AttentionUpdateTilingTest, AscendCGoDtypeInconsistentRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}, {ge::DT_FLOAT, {{128}, {128}}}},
                        {{ge::DT_FLOAT16, {{128, 8}, {128, 8}}}, {ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT16, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 2));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputDtype - lse各tensor dtype必须一致 (OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON "lse[i]")
TEST_F(AttentionUpdateTilingTest, AscendCLseDtypeInconsistentRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}, {ge::DT_FLOAT16, {{128}, {128}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}, {ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 2));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputDim - go维度数必须为2D (OP_LOGE_FOR_INVALID_SHAPEDIM "go[i]")
TEST_F(AttentionUpdateTilingTest, AscendCGoDimNumNot2DRejected)
{
    auto compileInfo = AscendCCompileInfo();
    // go[0]为2D保证d_合法，go[1]为1D触发go维度数校验
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}, {ge::DT_FLOAT, {{128}, {128}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}, {ge::DT_FLOAT, {{128}, {128}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 2));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputDim - lse维度数必须为1D (OP_LOGE_FOR_INVALID_SHAPEDIM "lse[i]")
TEST_F(AttentionUpdateTilingTest, AscendCLseDimNumNot1DRejected)
{
    auto compileInfo = AscendCCompileInfo();
    // sp=1且tensor数量匹配，lse[0]为2D触发lse维度数校验
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 1));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputDim - go各tensor的H维必须一致 (OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON "go[i]")
TEST_F(AttentionUpdateTilingTest, AscendCGoHDimInconsistentRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}, {ge::DT_FLOAT, {{128}, {128}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}, {ge::DT_FLOAT, {{128, 16}, {128, 16}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 2));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputDim - 所有tensor的batch维(第0维)必须一致 (OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON batch)
TEST_F(AttentionUpdateTilingTest, AscendCBatchDimInconsistentRejected)
{
    auto compileInfo = AscendCCompileInfo();
    // sp=2且tensor数量匹配，lse的第0维(64)与go的第0维(128)不一致触发batch轴校验
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{64}, {64}}}, {ge::DT_FLOAT, {{64}, {64}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}, {ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 2));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputParams - update_type必须为0或1 (OP_LOGE_FOR_INVALID_VALUE "update_type")
TEST_F(AttentionUpdateTilingTest, AscendCUpdateTypeInvalidRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}}, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(2, 1));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckInputParams - sp必须在[1,16] (OP_LOGE_FOR_INVALID_VALUE "sp")
TEST_F(AttentionUpdateTilingTest, AscendCSpOutOfRangeRejected)
{
    auto compileInfo = AscendCCompileInfo();
    std::vector<TensorSpec> lseTensors;
    std::vector<TensorSpec> goTensors;
    for (int64_t i = 0; i < 17; ++i) {
        lseTensors.push_back({ge::DT_FLOAT, {{128}, {128}}});
        goTensors.push_back({ge::DT_FLOAT, {{128, 8}, {128, 8}}});
    }
    auto tilingContextPara = BuildCustomCase(compileInfo, lseTensors, goTensors, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                                             DefaultAttrs(0, 17));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: GetShapeAttrsInfo - lse/go tensor数量必须为sp的2倍 (OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON)
TEST_F(AttentionUpdateTilingTest, AscendCTensorNumsMismatchRejected)
{
    auto compileInfo = AscendCCompileInfo();
    // sp属性为2，但仅提供1个lse和1个go输入，触发tensor数量校验失败
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}}, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 2));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: GetShapeAttrsInfo - 缺失sp属性，GetAttrPointer返回nullptr（OP_LOGE_WITHOUT_REPORT "spPtr is null"）
TEST_F(AttentionUpdateTilingTest, AscendCMissingSpAttrRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara = BuildCustomCase(
        compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}}, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}}, {{"update_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckOutputParams - output与go的dtype必须一致 (OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON "go and output")
TEST_F(AttentionUpdateTilingTest, AscendCOutputDtypeMismatchRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}}, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT16, {{128, 8}, {128, 8}}}}, DefaultAttrs(0, 1));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: CheckOutputParams - update_type=1时lse_m必须为FLOAT (OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON "lse_m")
TEST_F(AttentionUpdateTilingTest, AscendCOutputLseMDtypeMismatchRejected)
{
    auto compileInfo = AscendCCompileInfo();
    auto tilingContextPara =
        BuildCustomCase(compileInfo, {{ge::DT_FLOAT, {{128}, {128}}}}, {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}},
                        {{ge::DT_FLOAT, {{128, 8}, {128, 8}}}, {ge::DT_FLOAT16, {{128}, {128}}}}, DefaultAttrs(1, 1));
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 分支: GetPlatformInfo - 平台Vector核数为0 / UB大小为0（OP_LOGE_WITHOUT_REPORT）
// 说明：平台coreNum/ubSize由硬件信息保证非0，UT假平台不随TilingContextPara参数变化，该分支UT不可达，不构造用例。

TEST_F(AttentionUpdateTilingTest, LegacyRowStridedFloat)
{
    auto compileInfo = LegacyCompileInfo();
    auto context = BuildCase(compileInfo, ge::DT_FLOAT, 0, 5, {{33}, {33}}, {{33, 72}, {33, 76}}, {{33, 72}, {33, 72}},
                             {{}, {}}, 76);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 11010);
    ASSERT_EQ(info.tilingDataSize, 40U);
    uint64_t rowStride = 0;
    std::memcpy(&rowStride, info.tilingData.get() + info.tilingDataSize - sizeof(rowStride), sizeof(rowStride));
    EXPECT_EQ(rowStride, 76U);
}

TEST_F(AttentionUpdateTilingTest, AscendCRowStridedBf16WithLseOut)
{
    auto compileInfo = AscendCCompileInfo();
    auto context = BuildCase(compileInfo, ge::DT_BF16, 1, 9, {{33}, {33}}, {{33, 72}, {33, 76}}, {{33, 72}, {33, 72}},
                             {{33}, {33}}, 76);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 21001);
    ASSERT_EQ(info.tilingDataSize, 96U);
    uint64_t rowStride = 0;
    std::memcpy(&rowStride, info.tilingData.get() + info.tilingDataSize - sizeof(rowStride), sizeof(rowStride));
    EXPECT_EQ(rowStride, 76U);
}
