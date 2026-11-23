/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_flash_attn_shape_tiling.cpp
 * \brief FlashAttn tiling 依赖的共享 shape 工具 UT（照 quant_flash_mla 同名先例）
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "experiment/metadef/common/util/error_manager/error_manager.h"
// 该实现只以隐藏符号编入 libophost_transformer_ut.so, 测试 TU 内直接包含源码以便直接调用
#include "../../../../../common/op_host/fia_tiling_shape.cpp"

namespace FlashAttnUT {
namespace {

using optiling::FiaAxis;
using optiling::FiaLayout;
using optiling::FiaTilingShape;

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

// 断言 CheckHasShapeD 失败时首报的正式码
void ExpectCheckDFailsWith(const FiaTilingShape& s, const std::string& expectCode)
{
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    EXPECT_EQ(ge::GRAPH_FAILED, s.CheckHasShapeD("CheckHasShapeD"));
    const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, expectCode);
}

// layout 既无 D 轴也无 H 轴(BS2 = {B, S2}) -> EZ0026
TEST(FiaTilingShapeErrorCodeTest, ReportsEz0026WhenLayoutHasNeitherDNorH)
{
    auto shape = MakeShape({2, 128});
    FiaTilingShape s(shape, FiaLayout::BS2, "shape", "FlashAttn");
    EXPECT_FALSE(s.HasShapeD());
    EXPECT_FALSE(s.HasShapeH());
    ExpectCheckDFailsWith(s, "EZ0026");
}

// layout 有 H 但未指定 N(BSH = {B, S, H}, N 用默认值) -> EZ0037
TEST(FiaTilingShapeErrorCodeTest, ReportsEz0037WhenNNotSpecified)
{
    auto shape = MakeShape({2, 3, 32});
    FiaTilingShape s(shape, FiaLayout::BSH, "shape", "FlashAttn");
    EXPECT_TRUE(s.HasShapeH());
    EXPECT_FALSE(s.HasShapeD());
    EXPECT_EQ(s.GetShapeD(), kInvalid);
    ExpectCheckDFailsWith(s, "EZ0037");
}

// N 显式传 0 -> EZ0026
TEST(FiaTilingShapeErrorCodeTest, ReportsEz0026WhenNIsZero)
{
    auto shape = MakeShape({2, 3, 32});
    FiaTilingShape s(shape, FiaLayout::BSH, "shape", "FlashAttn", 0);
    EXPECT_EQ(s.GetShapeD(), kInvalid);
    ExpectCheckDFailsWith(s, "EZ0026");
}

// H 不是 N 的整数倍(30 % 4 != 0) -> EZ0012
TEST(FiaTilingShapeErrorCodeTest, ReportsEz0012WhenHNotMultipleOfN)
{
    auto shape = MakeShape({2, 3, 30});
    FiaTilingShape s(shape, FiaLayout::BSH, "shape", "FlashAttn", 4);
    EXPECT_FALSE(s.HasShapeD());
    ExpectCheckDFailsWith(s, "EZ0012");
}

// 正向对照: H 能被 N 整除时应成功且零上报
TEST(FiaTilingShapeErrorCodeTest, DerivesDFromHWhenAligned)
{
    auto shape = MakeShape({2, 3, 32});
    FiaTilingShape s(shape, FiaLayout::BSH, "shape", "FlashAttn", 4);
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    EXPECT_TRUE(s.HasShapeD());
    EXPECT_EQ(s.GetShapeD(), 8);
    EXPECT_EQ(ge::GRAPH_SUCCESS, s.CheckHasShapeD("CheckHasShapeD"));
    EXPECT_EQ(ErrorManager::GetInstance().GetRawErrorMessages().size(), 0U);
}

} // namespace
} // namespace FlashAttnUT
