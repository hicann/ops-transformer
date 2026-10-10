/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file test_grouped_matmul_quant_tiling.cpp
 *  \brief Unit tests for GroupedMatmulQuant tiling.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/grouped_matmul_quant_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace ge;

class GroupedMatmulQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulQuantTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulQuantTilingTest TearDown" << std::endl;
    }
};

namespace {
using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr uint64_t TILING_KEY_NORMAL = 10000001UL;
constexpr uint64_t PLATFORM_CORE_NUM = 40UL;
constexpr uint64_t PLATFORM_UB_SIZE = 196608UL;
constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

struct CaseParam {
    int64_t m = 17;
    int64_t k = 64;
    int64_t n = 80;
    int64_t groupNum = 1;
    int64_t scaleGroupSize = 32;
    ge::DataType dtype = ge::DT_FLOAT16;
    bool withGroupList = false;
};

struct CaseShapes {
    std::vector<int64_t> x;
    std::vector<int64_t> weight;
    std::vector<int64_t> scale;
    std::vector<int64_t> offset;
    std::vector<int64_t> groupList;
    std::vector<int64_t> y;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }

    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return storageShape;
}

CaseShapes MakeValidShapes(const CaseParam& param)
{
    return {
        {param.m, param.k},
        {param.groupNum, param.k / 16, param.n / 16, 16, 2},
        {param.groupNum, param.k / param.scaleGroupSize, param.n},
        {param.groupNum, param.k / param.scaleGroupSize, param.n},
        {param.groupNum},
        {param.m, param.n},
    };
}

gert::TilingContextPara MakeContext(const CaseParam& param, const CaseShapes& shapes,
                                    optiling::GroupedMatmulQuantCompileInfo* compileInfo)
{
    std::vector<TensorDescription> inputs = {
        {MakeStorageShape(shapes.x), param.dtype, ge::FORMAT_ND},
        {MakeStorageShape(shapes.weight), ge::DT_INT32, ge::FORMAT_ND},
        {MakeStorageShape(shapes.scale), param.dtype, ge::FORMAT_ND},
        {MakeStorageShape(shapes.offset), ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    if (param.withGroupList) {
        inputs.emplace_back(MakeStorageShape(shapes.groupList), ge::DT_INT64, ge::FORMAT_ND);
    }

    const std::vector<TensorDescription> outputs = {
        {MakeStorageShape(shapes.y), param.dtype, ge::FORMAT_ND},
    };
    const std::vector<OpAttr> attrs = {
        {"scale_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.scaleGroupSize)},
    };
    const std::vector<uint32_t> inputInstanceNum = {
        1U, 1U, 1U, 1U, static_cast<uint32_t>(param.withGroupList ? 1U : 0U),
    };

    return gert::TilingContextPara("GroupedMatmulQuant", inputs, outputs, attrs, inputInstanceNum, {1U}, compileInfo,
                                   "Ascend910B", PLATFORM_CORE_NUM, PLATFORM_UB_SIZE);
}

bool RunTiling(const CaseParam& param, const CaseShapes& shapes, TilingInfo& tilingInfo)
{
    optiling::GroupedMatmulQuantCompileInfo compileInfo;
    auto context = MakeContext(param, shapes, &compileInfo);
    return ExecuteTiling(context, tilingInfo);
}

template <typename T>
T ReadTilingField(const uint8_t* buffer, size_t& offset)
{
    T value = 0;
    std::memcpy(&value, buffer + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

void DeserializeTilingData(const TilingInfo& info, optiling::GroupedMatmulQuantTilingData& data)
{
    size_t offset = 0;
#define READ_TILING_FIELD(field) data.set_##field(ReadTilingField<uint32_t>(info.tilingData.get(), offset))
    READ_TILING_FIELD(CoreNum);
    READ_TILING_FIELD(dataType);
    READ_TILING_FIELD(UBSize);
    READ_TILING_FIELD(L1Size);
    READ_TILING_FIELD(L0ASize);
    READ_TILING_FIELD(L0BSize);
    READ_TILING_FIELD(L0CSize);
    READ_TILING_FIELD(noGroup);
    READ_TILING_FIELD(originE);
    READ_TILING_FIELD(originM);
    READ_TILING_FIELD(originN);
    READ_TILING_FIELD(originK);
    READ_TILING_FIELD(scaleK);
    READ_TILING_FIELD(scaleGroupSize);
    READ_TILING_FIELD(fracN);
    READ_TILING_FIELD(fracK);
    READ_TILING_FIELD(splitK);
    READ_TILING_FIELD(clearBaseN);
    READ_TILING_FIELD(clearOutLoop);
    READ_TILING_FIELD(clearOutTailN);
    READ_TILING_FIELD(clearOutTailCoreNum);
    READ_TILING_FIELD(castBaseN);
    READ_TILING_FIELD(castOutLoop);
    READ_TILING_FIELD(castOutTailN);
    READ_TILING_FIELD(castOutTailCoreNum);
#undef READ_TILING_FIELD
    constexpr size_t TILING_DATA_ALIGNMENT = sizeof(uint64_t);
    const size_t alignedSize = (offset + TILING_DATA_ALIGNMENT - 1U) / TILING_DATA_ALIGNMENT * TILING_DATA_ALIGNMENT;
    EXPECT_EQ(alignedSize, info.tilingDataSize)
        << "Tiling data schema changed or has unexpected trailing data; update the deserializer";
}

void ExpectValidCase(const CaseParam& param, uint32_t expectedSplitK)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(param, MakeValidShapes(param), info));
    ASSERT_NE(info.tilingData, nullptr);
    ASSERT_EQ(info.tilingKey, TILING_KEY_NORMAL);

    optiling::GroupedMatmulQuantTilingData data;
    ASSERT_EQ(info.tilingDataSize, data.GetDataSize());
    DeserializeTilingData(info, data);
    EXPECT_EQ(data.get_dataType(), static_cast<uint32_t>(param.dtype));
    EXPECT_EQ(data.get_noGroup(), static_cast<uint32_t>(param.withGroupList ? 0U : 1U));
    EXPECT_EQ(data.get_originE(), static_cast<uint32_t>(param.groupNum));
    EXPECT_EQ(data.get_originM(), static_cast<uint32_t>(param.m));
    EXPECT_EQ(data.get_originN(), static_cast<uint32_t>(param.n));
    EXPECT_EQ(data.get_originK(), static_cast<uint32_t>(param.k));
    EXPECT_EQ(data.get_scaleK(), static_cast<uint32_t>(param.k / param.scaleGroupSize));
    EXPECT_EQ(data.get_scaleGroupSize(), static_cast<uint32_t>(param.scaleGroupSize));
    EXPECT_EQ(data.get_fracN(), static_cast<uint32_t>(param.n / 16));
    EXPECT_EQ(data.get_fracK(), static_cast<uint32_t>(param.k / 16));
    EXPECT_EQ(data.get_splitK(), expectedSplitK);
    EXPECT_EQ(info.blockNum, data.get_CoreNum());

    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    const uint64_t expectedWorkspace =
        SYSTEM_WORKSPACE_SIZE + static_cast<uint64_t>(data.get_L0ASize()) * 8UL * data.get_CoreNum() +
        (expectedSplitK == 0U ? 0UL : static_cast<uint64_t>(param.m) * param.n * sizeof(float));
    EXPECT_EQ(info.workspaceSizes[0], expectedWorkspace);
}

void ExpectInvalidCase(const CaseParam& param, const CaseShapes& shapes)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling(param, shapes, info));
}
} // namespace

TEST_F(GroupedMatmulQuantTilingTest, fp16_single_group_without_group_list_tail_shape)
{
    const CaseParam param{17, 64, 80, 1, 32, ge::DT_FLOAT16, false};
    ExpectValidCase(param, 1U);
}

TEST_F(GroupedMatmulQuantTilingTest, bf16_multiple_groups_scale_group_64)
{
    const CaseParam param{257, 128, 256, 4, 64, ge::DT_BF16, true};
    ExpectValidCase(param, 1U);
}

TEST_F(GroupedMatmulQuantTilingTest, large_shape_disables_split_k)
{
    const CaseParam param{12288, 64, 1024, 2, 32, ge::DT_FLOAT16, true};
    ExpectValidCase(param, 0U);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_x_rank)
{
    CaseParam param;
    auto shapes = MakeValidShapes(param);
    shapes.x = {1, param.m, param.k};
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_k_not_divisible_by_scale_group)
{
    CaseParam param;
    param.k = 80;
    auto shapes = MakeValidShapes(param);
    shapes.weight = {param.groupNum, 5, param.n / 16, 16, 2};
    shapes.scale = {param.groupNum, 2, param.n};
    shapes.offset = shapes.scale;
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_scale_group_not_aligned_to_32)
{
    CaseParam param;
    param.scaleGroupSize = 48;
    param.k = 96;
    ExpectInvalidCase(param, MakeValidShapes(param));
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_weight_rank)
{
    CaseParam param;
    auto shapes = MakeValidShapes(param);
    shapes.weight = {param.groupNum, param.k / 16, param.n / 16, 32};
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_weight_inner_shape)
{
    CaseParam param;
    auto shapes = MakeValidShapes(param);
    shapes.weight = {param.groupNum, param.k / 16, param.n / 16, 8, 4};
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_scale_shape)
{
    CaseParam param;
    auto shapes = MakeValidShapes(param);
    shapes.scale = {param.groupNum, param.k / param.scaleGroupSize, param.n + 16};
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_offset_shape)
{
    CaseParam param;
    auto shapes = MakeValidShapes(param);
    shapes.offset = {param.groupNum, param.k / param.scaleGroupSize + 1, param.n};
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_missing_group_list_for_multiple_groups)
{
    CaseParam param;
    param.groupNum = 2;
    param.withGroupList = false;
    ExpectInvalidCase(param, MakeValidShapes(param));
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_group_list_shape)
{
    CaseParam param;
    param.groupNum = 3;
    param.withGroupList = true;
    auto shapes = MakeValidShapes(param);
    shapes.groupList = {param.groupNum + 1};
    ExpectInvalidCase(param, shapes);
}

TEST_F(GroupedMatmulQuantTilingTest, rejects_output_shape)
{
    CaseParam param;
    auto shapes = MakeValidShapes(param);
    shapes.y = {param.m, param.n + 16};
    ExpectInvalidCase(param, shapes);
}
