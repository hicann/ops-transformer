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

#include "../../../op_host/decode_update_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr uint64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024UL;

gert::TilingContextPara BuildCase(optiling::DecodeUpdateCompileInfo& compileInfo, ge::DataType dtype,
                                  int64_t updateType, int64_t sp, const gert::StorageShape& lseShape,
                                  const gert::StorageShape& localOutShape, const gert::StorageShape& outShape,
                                  const gert::StorageShape& lseOutShape = {{}, {}})
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {outShape, dtype, ge::FORMAT_ND},
    };

    for (int64_t i = 0; i < sp; ++i) {
        inputs.push_back({lseShape, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    for (int64_t i = 0; i < sp; ++i) {
        inputs.push_back({localOutShape, dtype, ge::FORMAT_ND});
    }
    if (updateType == 1) {
        outputs.push_back({lseOutShape, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    return gert::TilingContextPara("AttentionUpdate", inputs, outputs,
                                   {
                                       {"update_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(updateType)},
                                       {"sp", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sp)},
                                   },
                                   &compileInfo, "Ascend910B", compileInfo.coreNum, compileInfo.ubSize);
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
