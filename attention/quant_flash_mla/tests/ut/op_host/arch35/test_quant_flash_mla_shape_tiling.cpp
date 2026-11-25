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
 * \file test_quant_flash_mla_shape_tiling.cpp
 * \brief QuantFlashMla tiling 依赖的共享 shape 工具 UT
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
// 该实现只以隐藏符号编入 libophost_transformer_ut.so, 测试 TU 内直接包含源码以便直接调用
#include "../../../../../common/op_host/fia_tiling_shape.cpp"

namespace QuantFlashMlaUT {
namespace {

using optiling::FiaAxis;
using optiling::FiaCompareType;
using optiling::FiaLayout;
using optiling::FiaTilingShape;
using optiling::FiaTilingShapeCompare;
using optiling::FiaTilingShapeCompareParam;

constexpr int64_t kNotSetN = gert::Shape::kInvalidDimValue;
constexpr int64_t kInvalid = gert::Shape::kInvalidDimValue;

gert::Shape MakeShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

struct AccessorCase {
    FiaLayout layout;
    std::vector<int64_t> dims;
    int64_t nParam;
    int64_t b;
    int64_t s;
    int64_t n;
    int64_t h;
    int64_t t;
    int64_t d;
    int64_t bn;
    int64_t bs;
};

// 各 layout 下 axis 解析与取值
const std::vector<AccessorCase> ACCESSOR_CASES = {
    {FiaLayout::BSH, {2, 3, 32}, 4, 2, 3, kInvalid, 32, kInvalid, 8, kInvalid, kInvalid},
    {FiaLayout::BSND, {2, 3, 4, 8}, kNotSetN, 2, 3, 4, kInvalid, kInvalid, 8, kInvalid, kInvalid},
    {FiaLayout::BNSD, {2, 4, 3, 8}, kNotSetN, 2, 3, 4, kInvalid, kInvalid, 8, kInvalid, kInvalid},
    {FiaLayout::TND, {6, 4, 8}, kNotSetN, kInvalid, kInvalid, 4, kInvalid, 6, 8, kInvalid, kInvalid},
    {FiaLayout::NBSD, {4, 2, 3, 8}, kNotSetN, 2, 3, 4, kInvalid, kInvalid, 8, kInvalid, kInvalid},
    {FiaLayout::NTD, {4, 6, 8}, kNotSetN, kInvalid, kInvalid, 4, kInvalid, 6, 8, kInvalid, kInvalid},
    {FiaLayout::PA_BBND, {5, 128, 4, 8}, kNotSetN, kInvalid, kInvalid, 4, kInvalid, kInvalid, 8, 5, 128},
    {FiaLayout::PA_BNBD, {5, 4, 128, 8}, kNotSetN, kInvalid, kInvalid, 4, kInvalid, kInvalid, 8, 5, 128},
    {FiaLayout::PA_NZ, {5, 4, 2, 128, 8}, kNotSetN, kInvalid, kInvalid, 4, kInvalid, kInvalid, 16, 5, 128},
    {FiaLayout::BnBsH, {5, 128, 32}, 4, kInvalid, kInvalid, kInvalid, 32, kInvalid, 8, 5, 128},
};

} // namespace

TEST(FiaTilingShapeTest, AccessorsAcrossLayouts)
{
    for (const auto& item : ACCESSOR_CASES) {
        auto shape = MakeShape(item.dims);
        FiaTilingShape tilingShape(shape, item.layout, "shape", "QuantFlashMla", item.nParam);
        EXPECT_EQ(tilingShape.GetDimNum(), item.dims.size());
        EXPECT_EQ(tilingShape.GetShapeB(), item.b);
        EXPECT_EQ(tilingShape.GetShapeS(), item.s);
        EXPECT_EQ(tilingShape.GetShapeN(), item.n);
        EXPECT_EQ(tilingShape.GetShapeH(), item.h);
        EXPECT_EQ(tilingShape.GetShapeT(), item.t);
        EXPECT_EQ(tilingShape.GetShapeD(), item.d);
        EXPECT_EQ(tilingShape.GetShapeBlockNum(), item.bn);
        EXPECT_EQ(tilingShape.GetShapeBlockSize(), item.bs);
    }
}

TEST(FiaTilingShapeTest, DrillIntoD1AndD0)
{
    auto shape = MakeShape({5, 4, 2, 128, 8});
    FiaTilingShape tilingShape(shape, FiaLayout::PA_NZ, "shape", "QuantFlashMla");
    EXPECT_TRUE(tilingShape.HasShapeD1());
    EXPECT_TRUE(tilingShape.HasShapeD0());
    EXPECT_TRUE(tilingShape.HasShapeD());
    EXPECT_EQ(tilingShape.GetShapeD1(), 2);
    EXPECT_EQ(tilingShape.GetShapeD0(), 8);
    EXPECT_EQ(ge::GRAPH_SUCCESS, tilingShape.CheckHasShapeD("CheckHasShapeD"));
}

TEST(FiaTilingShapeTest, DeriveDFromH)
{
    auto shape = MakeShape({2, 3, 32});
    FiaTilingShape withN(shape, FiaLayout::BSH, "shape", "QuantFlashMla", 4);
    EXPECT_TRUE(withN.HasShapeH());
    EXPECT_TRUE(withN.HasShapeD());
    EXPECT_EQ(withN.GetShapeD(), 8);
    EXPECT_EQ(ge::GRAPH_SUCCESS, withN.CheckHasShapeD("CheckHasShapeD"));

    FiaTilingShape withoutN(shape, FiaLayout::BSH, "shape", "QuantFlashMla");
    EXPECT_FALSE(withoutN.HasShapeD());
    EXPECT_EQ(withoutN.GetShapeD(), kInvalid);
    EXPECT_EQ(ge::GRAPH_FAILED, withoutN.CheckHasShapeD("CheckHasShapeD"));

    FiaTilingShape zeroN(shape, FiaLayout::BSH, "shape", "QuantFlashMla", 0);
    EXPECT_EQ(zeroN.GetShapeD(), kInvalid);
    EXPECT_EQ(ge::GRAPH_FAILED, zeroN.CheckHasShapeD("CheckHasShapeD"));

    auto notAlignedShape = MakeShape({2, 3, 30});
    FiaTilingShape notAligned(notAlignedShape, FiaLayout::BSH, "shape", "QuantFlashMla", 4);
    EXPECT_FALSE(notAligned.HasShapeD());
    EXPECT_EQ(ge::GRAPH_FAILED, notAligned.CheckHasShapeD("CheckHasShapeD"));
}

TEST(FiaTilingShapeTest, CheckHasShapeFailures)
{
    auto shape = MakeShape({2, 3, 4, 8});
    FiaTilingShape bsnd(shape, FiaLayout::BSND, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_SUCCESS, bsnd.CheckHasShapeB("CheckHasShapeB"));
    EXPECT_EQ(ge::GRAPH_SUCCESS, bsnd.CheckHasShapeS("CheckHasShapeS"));
    EXPECT_EQ(ge::GRAPH_SUCCESS, bsnd.CheckHasShapeN("CheckHasShapeN"));
    EXPECT_EQ(ge::GRAPH_SUCCESS, bsnd.CheckHasShapeD("CheckHasShapeD"));

    auto threeDimShape = MakeShape({2, 3, 32});
    FiaTilingShape bsndWrongDim(threeDimShape, FiaLayout::BSND, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_FAILED, bsndWrongDim.CheckHasShapeB("CheckHasShapeB"));

    auto emptyShape = MakeShape({});
    FiaTilingShape empty(emptyShape, FiaLayout::BSND, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_FAILED, empty.CheckHasShapeB("CheckHasShapeB"));

    // 布局不含 H, 校验 H 走 axis 缺失分支
    auto lseShape = MakeShape({4, 6});
    FiaTilingShape lse(lseShape, FiaLayout::LSE_NT, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_FAILED, lse.CheckHasShapeH("CheckHasShapeH"));
    EXPECT_EQ(ge::GRAPH_FAILED, lse.CheckHasShapeD("CheckHasShapeD"));

    // 布局不含 N/T/Bs/Bn, 覆盖对应校验分支
    FiaTilingShape bsh(threeDimShape, FiaLayout::BSH, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_FAILED, bsh.CheckHasShapeN("CheckHasShapeN"));
    EXPECT_EQ(ge::GRAPH_FAILED, bsh.CheckHasShapeT("CheckHasShapeT"));
    EXPECT_EQ(ge::GRAPH_FAILED, bsh.CheckHasShapeBlockNum("CheckHasShapeBlockNum"));
    EXPECT_EQ(ge::GRAPH_FAILED, bsh.CheckHasShapeBlockSize("CheckHasShapeBlockSize"));

    // FiaTilingShape 以 const gert::Shape& 保存 shape, 临时对象必须落到具名局部变量, 否则析构后使用
    gert::Shape tndShape = MakeShape({6, 4, 8});
    FiaTilingShape tnd(tndShape, FiaLayout::TND, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_SUCCESS, tnd.CheckHasShapeT("CheckHasShapeT"));

    gert::Shape paBnbdShape = MakeShape({5, 4, 128, 8});
    FiaTilingShape paBnbd(paBnbdShape, FiaLayout::PA_BNBD, "shape", "QuantFlashMla");
    EXPECT_EQ(ge::GRAPH_SUCCESS, paBnbd.CheckHasShapeBlockNum("CheckHasShapeBlockNum"));
    EXPECT_EQ(ge::GRAPH_SUCCESS, paBnbd.CheckHasShapeBlockSize("CheckHasShapeBlockSize"));

    // 不支持的 layout
    FiaTilingShape unknown(shape, static_cast<FiaLayout>(100U), "shape", "QuantFlashMla");
    EXPECT_FALSE(unknown.HasShapeB());
    EXPECT_EQ(ge::GRAPH_FAILED, unknown.CheckHasShapeB("CheckHasShapeB"));
}

TEST(FiaTilingShapeTest, CompareTypeSerialStrings)
{
    // 同上: 先落到具名局部变量, 避免 FiaTilingShapeCompare::shape_ 悬空
    gert::Shape compareShape = MakeShape({2, 3, 4, 8});
    FiaTilingShapeCompare compare(compareShape, FiaLayout::BSND, "shape", "QuantFlashMla");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::EQUAL), "EQUAL");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::GREATER), "GREATER");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::GREATER_EQUAL), "GREATER_EQUAL");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::LESS), "LESS");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::LESS_EQUAL), "LESS_EQUAL");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::NOT_EQUAL), "NOT_EQUAL");
    EXPECT_EQ(compare.CompareTypeToSerialString(FiaCompareType::IGNORE_INPUT), "UNKNOWN");

    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::EQUAL), "==");
    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::GREATER), ">");
    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::GREATER_EQUAL), ">=");
    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::LESS), "<");
    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::LESS_EQUAL), "<=");
    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::NOT_EQUAL), "!=");
    EXPECT_EQ(compare.CompareTypeToSerialSymbolString(FiaCompareType::IGNORE_INPUT), "UNKNOWN");
}

namespace {

struct CompareCase {
    FiaLayout layout;
    std::vector<int64_t> dims;
    std::vector<int64_t> expectDims;
};

// 各 layout 的期望 shape 推导
const std::vector<CompareCase> COMPARE_CASES = {
    {FiaLayout::BSH, {2, 3, 32}, {2, 3, 32}},
    {FiaLayout::BSND, {2, 3, 4, 8}, {2, 3, 4, 8}},
    {FiaLayout::BNSD, {2, 4, 3, 8}, {2, 4, 3, 8}},
    {FiaLayout::NZ, {5, 4, 1, 128, 8}, {5, 4, 1, 128, 8}},
    {FiaLayout::TND, {6, 4, 8}, {6, 4, 8}},
    {FiaLayout::NBSD, {4, 2, 3, 8}, {4, 2, 3, 8}},
    {FiaLayout::NTD, {4, 6, 8}, {4, 6, 8}},
    {FiaLayout::BnBsH, {5, 128, 32}, {5, 128, 32}},
    {FiaLayout::BnNBsD, {5, 4, 128, 8}, {5, 4, 128, 8}},
    {FiaLayout::PA_BBND, {5, 128, 4, 8}, {5, 128, 4, 8}},
    {FiaLayout::PA_BNBD, {5, 4, 128, 8}, {5, 4, 128, 8}},
    {FiaLayout::PA_NZ, {5, 4, 1, 128, 8}, {5, 4, 1, 128, 8}},
    {FiaLayout::LSE_BNS, {2, 4, 3}, {2, 4, 3}},
    {FiaLayout::LSE_NT, {4, 6}, {4, 6}},
    {FiaLayout::BNS1S2, {2, 4, 3, 5}, {2, 4, 3, 5}},
    {FiaLayout::INS1S2, {1, 4, 3, 5}, {1, 4, 3, 5}},
    {FiaLayout::BNS11, {2, 4, 3, 1}, {2, 4, 3, 1}},
    {FiaLayout::TN1, {6, 4, 1}, {6, 4, 1}},
    {FiaLayout::BS2, {2, 5}, {2, 5}},
    {FiaLayout::S1S2, {3, 5}, {3, 5}},
    {FiaLayout::BS1S2, {2, 3, 5}, {2, 3, 5}},
    {FiaLayout::B1S1S2, {2, 1, 3, 5}, {2, 1, 3, 5}},
    {FiaLayout::IS1S2, {1, 3, 5}, {1, 3, 5}},
    {FiaLayout::I1S1S2, {1, 1, 3, 5}, {1, 1, 3, 5}},
    {FiaLayout::S1S1, {3}, {3}},
};

FiaTilingShapeCompareParam MakeCompareParam()
{
    FiaTilingShapeCompareParam param;
    param.B = 2;
    param.S = 3;
    param.N = 4;
    param.D = 8;
    param.H = param.N * param.D;
    param.T = 6;
    param.Bn = 5;
    param.Bs = 128;
    param.D0 = 8;
    param.S1 = 3;
    param.S2 = 5;
    param.CONST = 1;
    return param;
}

} // namespace

TEST(FiaTilingShapeCompareTest, ExpectedShapeAllLayouts)
{
    for (const auto& item : COMPARE_CASES) {
        auto shape = MakeShape(item.dims);
        FiaTilingShapeCompare compare(shape, item.layout, "shape", "QuantFlashMla");
        auto param = MakeCompareParam();
        gert::Shape expected;
        EXPECT_EQ(ge::GRAPH_SUCCESS, compare.GetExpectedShape(expected, param, "GetExpectedShape"));
        ASSERT_EQ(expected.GetDimNum(), item.expectDims.size());
        for (size_t i = 0; i < item.expectDims.size(); ++i) {
            EXPECT_EQ(expected.GetDim(i), item.expectDims[i]);
        }
        EXPECT_EQ(ge::GRAPH_SUCCESS, compare.CompareShape(param, "CompareShape"));
    }
}

TEST(FiaTilingShapeCompareTest, CompareWithCustomCompareTypes)
{
    struct Case {
        FiaCompareType type;
        int64_t shapeB;
    };
    const std::vector<Case> cases = {
        {FiaCompareType::EQUAL, 2},         {FiaCompareType::GREATER, 3},    {FiaCompareType::GREATER_EQUAL, 2},
        {FiaCompareType::LESS, 1},          {FiaCompareType::LESS_EQUAL, 2}, {FiaCompareType::NOT_EQUAL, 1},
        {FiaCompareType::IGNORE_INPUT, 99},
    };
    for (const auto& item : cases) {
        auto shape = MakeShape({item.shapeB, 3, 4, 8});
        FiaTilingShapeCompare compare(shape, FiaLayout::BSND, "shape", "QuantFlashMla");
        auto param = MakeCompareParam();
        param.compareTypeMap[FiaAxis::B] = item.type;
        EXPECT_EQ(ge::GRAPH_SUCCESS, compare.CompareShape(param, "CompareShape"));
        EXPECT_EQ(compare.GetCompareType(param.compareTypeMap, FiaAxis::B), item.type);
        EXPECT_EQ(compare.GetCompareType(param.compareTypeMap, FiaAxis::S), FiaCompareType::EQUAL);
    }
}

TEST(FiaTilingShapeCompareTest, CompareFailures)
{
    auto param = MakeCompareParam();

    // 未指定比较方式, B 不相等
    {
        auto shape = MakeShape({3, 3, 4, 8});
        FiaTilingShapeCompare compare(shape, FiaLayout::BSND, "shape", "QuantFlashMla");
        EXPECT_EQ(ge::GRAPH_FAILED, compare.CompareShape(param, "CompareShape"));
    }

    // 指定比较方式但不满足
    {
        auto shape = MakeShape({1, 3, 4, 8});
        FiaTilingShapeCompare compare(shape, FiaLayout::BSND, "shape", "QuantFlashMla");
        auto customParam = MakeCompareParam();
        customParam.compareTypeMap[FiaAxis::B] = FiaCompareType::GREATER;
        EXPECT_EQ(ge::GRAPH_FAILED, compare.CompareShape(customParam, "CompareShape"));
    }

    // shape 维度与 layout 不符
    {
        auto shape = MakeShape({2, 3, 4});
        FiaTilingShapeCompare compare(shape, FiaLayout::BSND, "shape", "QuantFlashMla");
        EXPECT_EQ(ge::GRAPH_FAILED, compare.CompareShape(param, "CompareShape"));
    }

    // 不支持的 layout
    {
        auto shape = MakeShape({2, 3, 4, 8});
        FiaTilingShapeCompare compare(shape, static_cast<FiaLayout>(100U), "shape", "QuantFlashMla");
        gert::Shape expected;
        EXPECT_EQ(ge::GRAPH_FAILED, compare.GetExpectedShape(expected, param, "GetExpectedShape"));
        EXPECT_EQ(ge::GRAPH_FAILED, compare.CompareShape(param, "CompareShape"));
    }

    // 不支持的比较类型
    {
        auto shape = MakeShape({2, 3, 4, 8});
        FiaTilingShapeCompare compare(shape, FiaLayout::BSND, "shape", "QuantFlashMla");
        auto customParam = MakeCompareParam();
        customParam.compareTypeMap[FiaAxis::B] = static_cast<FiaCompareType>(99);
        EXPECT_EQ(ge::GRAPH_FAILED, compare.CompareShape(customParam, "CompareShape"));
    }
}

} // namespace QuantFlashMlaUT
