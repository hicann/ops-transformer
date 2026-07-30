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
 * \file fa_tiling_shape.cpp
 * \brief Flash Attention Tiling Shape 检查实现
 */

#include <vector>
#include <map>
#include <algorithm>
#include "fa_tiling_shape.h"

namespace optiling {
namespace flash_attn {

// ============================================================================
// 静态常量定义
// ============================================================================

static const std::map<FaLayout, std::vector<FaAxis>> FA_LAYOUT_AXIS_MAP = {
    {FaLayout::BSND, {FaAxis::B, FaAxis::S, FaAxis::N, FaAxis::D}},
    {FaLayout::BNSD, {FaAxis::B, FaAxis::N, FaAxis::S, FaAxis::D}},
    {FaLayout::TND, {FaAxis::T, FaAxis::N, FaAxis::D}},
    {FaLayout::PA_BBND, {FaAxis::Bn, FaAxis::Bs, FaAxis::N, FaAxis::D}},
    {FaLayout::PA_BNBD, {FaAxis::Bn, FaAxis::N, FaAxis::Bs, FaAxis::D}},
    {FaLayout::PA_NZ, {FaAxis::Bn, FaAxis::N, FaAxis::D1, FaAxis::Bs, FaAxis::D0}},
    {FaLayout::LSE_BNS, {FaAxis::B, FaAxis::N, FaAxis::S1}},
    {FaLayout::LSE_NT, {FaAxis::N, FaAxis::T}}};

// ============================================================================
// 比较函数工具类
// ============================================================================

namespace {

struct CompareFuncs {
    static bool Equal(const int64_t &a, const int64_t &b)
    {
        return a == b;
    }
    static bool Greater(const int64_t &a, const int64_t &b)
    {
        return a > b;
    }
    static bool GreaterEqual(const int64_t &a, const int64_t &b)
    {
        return a >= b;
    }
    static bool Less(const int64_t &a, const int64_t &b)
    {
        return a < b;
    }
    static bool LessEqual(const int64_t &a, const int64_t &b)
    {
        return a <= b;
    }
    static bool NotEqual(const int64_t &a, const int64_t &b)
    {
        return a != b;
    }
    static bool IgnoreInput(const int64_t &a, const int64_t &b)
    {
        (void)a;
        (void)b;
        return true;
    }
};

} // namespace

const std::map<FaCompareType, CompareFunc<int64_t>> FaTilingShapeCompare::compareFuncMap_ = {
    {FaCompareType::EQUAL, CompareFuncs::Equal},
    {FaCompareType::GREATER, CompareFuncs::Greater},
    {FaCompareType::GREATER_EQUAL, CompareFuncs::GreaterEqual},
    {FaCompareType::LESS, CompareFuncs::Less},
    {FaCompareType::LESS_EQUAL, CompareFuncs::LessEqual},
    {FaCompareType::NOT_EQUAL, CompareFuncs::NotEqual},
    {FaCompareType::IGNORE_INPUT, CompareFuncs::IgnoreInput}};

// ============================================================================
// 辅助函数
// ============================================================================

static ge::graphStatus GetLayoutAxes(std::vector<FaAxis> &layoutAxes, const FaLayout &layout, const std::string &opName,
                                     const std::string &funcName)
{
    auto it = FA_LAYOUT_AXIS_MAP.find(layout);
    if (it == FA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(opName, "[%s] Layout %s is unsupported.", funcName.c_str(), LayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    layoutAxes = it->second;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// FaTilingShape 类实现
// ============================================================================

bool FaTilingShape::HasAxis(const FaAxis &axis) const
{
    auto layoutIt = FA_LAYOUT_AXIS_MAP.find(layout_);
    if (layoutIt == FA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }
    const auto &axes = layoutIt->second;
    return std::find(axes.begin(), axes.end(), axis) != axes.end();
}

size_t FaTilingShape::GetAxisIdx(const FaAxis &axis) const
{
    if (!HasAxis(axis)) {
        return 0;
    }
    const auto &axes = FA_LAYOUT_AXIS_MAP.find(layout_)->second;
    auto axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

int64_t FaTilingShape::GetAxisNum(const FaAxis &axis) const
{
    return HasAxis(axis) ? shape_.GetDim(GetAxisIdx(axis)) : invalidDimValue_;
}

ge::graphStatus FaTilingShape::CheckHasAxis(const FaAxis &axis, const std::string &funcName) const
{
    if (shape_.GetDimNum() == 0) {
        std::string reason = "[" + funcName + "] The shape dim of " + name_ + " cannot be 0";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, name_.c_str(),
            std::to_string(0).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    std::vector<FaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape_.GetDimNum() != layoutAxes.size()) {
        std::string dimStr = std::to_string(shape_.GetDimNum());
        std::string reason = "[" + funcName + "] The shape dim of " + name_ +
            " must be equal to the axes size of layout(" + LayoutToSerialString(layout_) + "): " +
            std::to_string(layoutAxes.size());
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, name_.c_str(), dimStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if ((axis == FaAxis::D)) {
        if (HasShapeD()) {
            return ge::GRAPH_SUCCESS;
        }
        std::string reason = "[" + funcName + "] Axis D or (D1, D0) cannot be empty when layout of " +
            name_ + " is " + LayoutToSerialString(layout_);
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, name_.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    } else if (HasAxis(axis)) {
        return ge::GRAPH_SUCCESS;
    }
    std::string reasonStr = "[" + funcName + "] " + AxisToSerialString(axis) +
        " cannot be empty when layout of " + name_ + " is " + LayoutToSerialString(layout_);
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, name_.c_str(), reasonStr.c_str());
    return ge::GRAPH_FAILED;
}

// ============================================================================
// FaTilingShapeCompare 类实现
// ============================================================================

std::string FaTilingShapeCompare::CompareTypeToSerialString(const FaCompareType compareType) const
{
    static const std::map<FaCompareType, std::string> typeStrMap = {{FaCompareType::EQUAL, "EQUAL"},
                                                                    {FaCompareType::GREATER, "GREATER"},
                                                                    {FaCompareType::GREATER_EQUAL, "GREATER_EQUAL"},
                                                                    {FaCompareType::LESS, "LESS"},
                                                                    {FaCompareType::LESS_EQUAL, "LESS_EQUAL"},
                                                                    {FaCompareType::NOT_EQUAL, "NOT_EQUAL"}};
    auto it = typeStrMap.find(compareType);
    return (it != typeStrMap.end()) ? it->second : "UNKNOWN";
}

std::string FaTilingShapeCompare::CompareTypeToSerialSymbolString(const FaCompareType &compareType) const
{
    static const std::map<FaCompareType, std::string> symbolMap = {
        {FaCompareType::EQUAL, "=="}, {FaCompareType::GREATER, ">"},     {FaCompareType::GREATER_EQUAL, ">="},
        {FaCompareType::LESS, "<"},   {FaCompareType::LESS_EQUAL, "<="}, {FaCompareType::NOT_EQUAL, "!="}};
    auto it = symbolMap.find(compareType);
    return (it != symbolMap.end()) ? it->second : "UNKNOWN";
}

ge::graphStatus FaTilingShapeCompare::GetExpectedShape(gert::Shape &shapeExpected,
                                                       const FaTilingShapeCompareParam &param,
                                                       const std::string &funcName) const
{
    switch (layout_) {
        case FaLayout::BSND:
            shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
            break;
        case FaLayout::BNSD:
            shapeExpected = gert::Shape({param.B, param.N, param.S, param.D});
            break;
        case FaLayout::TND:
            shapeExpected = gert::Shape({param.T, param.N, param.D});
            break;
        case FaLayout::PA_BBND:
            shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
            break;
        case FaLayout::PA_BNBD:
            shapeExpected = gert::Shape({param.Bn, param.N, param.Bs, param.D});
            break;
        case FaLayout::PA_NZ:
            shapeExpected = gert::Shape({param.Bn, param.N, param.D / param.D0, param.Bs, param.D0});
            break;
        case FaLayout::LSE_BNS:
            shapeExpected = gert::Shape({param.B, param.N, param.S1});
            break;
        case FaLayout::LSE_NT:
            shapeExpected = gert::Shape({param.N, param.T});
            break;
        default:
            OP_LOGE(opName_, "[%s] Layout %s is unsupported.", funcName.c_str(), LayoutToSerialString(layout_).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

FaCompareType FaTilingShapeCompare::GetCompareType(const std::map<FaAxis, FaCompareType> &compareTypeMap,
                                                   const FaAxis &axis) const
{
    auto it = compareTypeMap.find(axis);
    return (it != compareTypeMap.end()) ? it->second : FaCompareType::EQUAL;
}

ge::graphStatus FaTilingShapeCompare::GetCompareFunc(const FaCompareType &compareType,
                                                     CompareFunc<int64_t> &compareFunc,
                                                     const std::string &funcName) const
{
    auto it = compareFuncMap_.find(compareType);
    if (it == compareFuncMap_.end()) {
        OP_LOGE(opName_, "[%s] Compare type %s is unsupported.", funcName.c_str(),
                CompareTypeToSerialString(compareType).c_str());
        return ge::GRAPH_FAILED;
    }
    compareFunc = it->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FaTilingShapeCompare::CompareShape(FaTilingShapeCompareParam &param, const std::string &funcName) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::vector<FaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape_.GetDimNum() != shapeExpected.GetDimNum() || shape_.GetDimNum() != layoutAxes.size()) {
        std::string reason = "[" + funcName + "] The shape dim of " + name_ + " must be " +
            std::to_string(shapeExpected.GetDimNum()) + " when layout is " + LayoutToSerialString(layout_);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, name_.c_str(),
            std::to_string(shape_.GetDimNum()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape_.GetDimNum(); i++) {
        FaAxis axis = layoutAxes[i];
        FaCompareType compareType = GetCompareType(param.compareTypeMap, axis);
        CompareFunc<int64_t> compareFunc;
        if (GetCompareFunc(compareType, compareFunc, funcName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (!compareFunc(shape_.GetDim(i), shapeExpected.GetDim(i))) {
            if (param.compareTypeMap.empty()) {
                std::string reason = "[" + funcName + "] The shape of " + name_ + " must be equal to " +
                    GetShapeStr(shapeExpected) + " when layout of " + name_ + " is " + LayoutToSerialString(layout_);
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, name_.c_str(),
                    GetShapeStr(shape_).c_str(), reason.c_str());
            } else {
                std::string paramStr = name_ + " and expected";
                std::string shapeMsg = GetShapeStr(shape_) + " and " + GetShapeStr(shapeExpected);
                std::string reason = "[" + funcName + "] The axis " + AxisToSerialString(axis) + " of " +
                    name_ + " must " + CompareTypeToSerialSymbolString(compareType) + "expected: " +
                    std::to_string(shapeExpected.GetDim(i)) + " when layout of " + name_ + " is " +
                    LayoutToSerialString(layout_);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, paramStr.c_str(), shapeMsg.c_str(), reason.c_str());
            }
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling