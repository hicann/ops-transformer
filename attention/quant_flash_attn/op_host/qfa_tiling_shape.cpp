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
 * \file qfa_tiling_shape.cpp
 * \brief QuantFlashAttn Tiling Shape implementation
 */

#include <vector>
#include <map>
#include <algorithm>
#include "qfa_tiling_shape.h"

namespace optiling {
namespace quant_flash_attn {

static const std::map<QfaLayout, std::vector<QfaAxis>> QFA_LAYOUT_AXIS_MAP = {
    {QfaLayout::BSND, {QfaAxis::B, QfaAxis::S, QfaAxis::N, QfaAxis::D}},
    {QfaLayout::BNSD, {QfaAxis::B, QfaAxis::N, QfaAxis::S, QfaAxis::D}},
    {QfaLayout::TND, {QfaAxis::T, QfaAxis::N, QfaAxis::D}},
    {QfaLayout::NTD, {QfaAxis::N, QfaAxis::T, QfaAxis::D}},
    {QfaLayout::PA_BBND, {QfaAxis::Bn, QfaAxis::Bs, QfaAxis::N, QfaAxis::D}},
    {QfaLayout::PA_BNBD, {QfaAxis::Bn, QfaAxis::N, QfaAxis::Bs, QfaAxis::D}},
    {QfaLayout::PA_NZ, {QfaAxis::Bn, QfaAxis::N, QfaAxis::D1, QfaAxis::Bs, QfaAxis::D0}},
    {QfaLayout::LSE_BNS, {QfaAxis::B, QfaAxis::N, QfaAxis::S1}},
    {QfaLayout::LSE_NT, {QfaAxis::N, QfaAxis::T}}};

namespace {

struct CompareFuncs {
    static bool Equal(const int64_t &a, const int64_t &b) { return a == b; }
    static bool Greater(const int64_t &a, const int64_t &b) { return a > b; }
    static bool GreaterEqual(const int64_t &a, const int64_t &b) { return a >= b; }
    static bool Less(const int64_t &a, const int64_t &b) { return a < b; }
    static bool LessEqual(const int64_t &a, const int64_t &b) { return a <= b; }
    static bool NotEqual(const int64_t &a, const int64_t &b) { return a != b; }
    static bool IgnoreInput(const int64_t &a, const int64_t &b)
    {
        (void)a;
        (void)b;
        return true;
    }
};

} // namespace

const std::map<QfaCompareType, QfaCompareFunc<int64_t>> QfaTilingShapeCompare::compareFuncMap_ = {
    {QfaCompareType::EQUAL, CompareFuncs::Equal},
    {QfaCompareType::GREATER, CompareFuncs::Greater},
    {QfaCompareType::GREATER_EQUAL, CompareFuncs::GreaterEqual},
    {QfaCompareType::LESS, CompareFuncs::Less},
    {QfaCompareType::LESS_EQUAL, CompareFuncs::LessEqual},
    {QfaCompareType::NOT_EQUAL, CompareFuncs::NotEqual},
    {QfaCompareType::IGNORE_INPUT, CompareFuncs::IgnoreInput}};

static ge::graphStatus GetLayoutAxes(std::vector<QfaAxis> &layoutAxes, const QfaLayout &layout,
                                     const std::string &opName, const std::string &funcName)
{
    auto it = QFA_LAYOUT_AXIS_MAP.find(layout);
    if (it == QFA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(opName, "[%s] Layout %s is unsupported.", funcName.c_str(), QfaLayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    layoutAxes = it->second;
    return ge::GRAPH_SUCCESS;
}

bool QfaTilingShape::HasAxis(const QfaAxis &axis) const
{
    auto layoutIt = QFA_LAYOUT_AXIS_MAP.find(layout_);
    if (layoutIt == QFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }
    const auto &axes = layoutIt->second;
    return std::find(axes.begin(), axes.end(), axis) != axes.end();
}

size_t QfaTilingShape::GetAxisIdx(const QfaAxis &axis) const
{
    if (!HasAxis(axis)) {
        return 0;
    }
    const auto &axes = QFA_LAYOUT_AXIS_MAP.find(layout_)->second;
    auto axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

int64_t QfaTilingShape::GetAxisNum(const QfaAxis &axis) const
{
    return HasAxis(axis) ? shape_.GetDim(GetAxisIdx(axis)) : invalidDimValue_;
}

ge::graphStatus QfaTilingShape::CheckHasAxis(const QfaAxis &axis, const std::string &funcName) const
{
    if (shape_.GetDimNum() == 0) {
        OP_LOGE(opName_, "[%s] The dim number of %s is 0.", funcName.c_str(), name_.c_str());
        return ge::GRAPH_FAILED;
    }

    std::vector<QfaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape_.GetDimNum() != layoutAxes.size()) {
        OP_LOGE(opName_, "[%s] %s shape dimension is %zu, expected is %zu (layout %s).", funcName.c_str(),
                name_.c_str(), shape_.GetDimNum(), layoutAxes.size(), QfaLayoutToSerialString(layout_).c_str());
        return ge::GRAPH_FAILED;
    }

    if (axis == QfaAxis::D) {
        if (HasShapeD()) {
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGE(opName_, "[%s] %s's layout is %s, axis D or (D1, D0) does not exist.", funcName.c_str(), name_.c_str(),
                QfaLayoutToSerialString(layout_).c_str());
        return ge::GRAPH_FAILED;
    } else if (HasAxis(axis)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(opName_, "[%s] %s's layout is %s, %s is not exists.", funcName.c_str(), name_.c_str(),
            QfaLayoutToSerialString(layout_).c_str(), QfaAxisToSerialString(axis).c_str());
    return ge::GRAPH_FAILED;
}

std::string QfaTilingShapeCompare::CompareTypeToSerialString(const QfaCompareType compareType) const
{
    static const std::map<QfaCompareType, std::string> typeStrMap = {{QfaCompareType::EQUAL, "EQUAL"},
                                                                     {QfaCompareType::GREATER, "GREATER"},
                                                                     {QfaCompareType::GREATER_EQUAL, "GREATER_EQUAL"},
                                                                     {QfaCompareType::LESS, "LESS"},
                                                                     {QfaCompareType::LESS_EQUAL, "LESS_EQUAL"},
                                                                     {QfaCompareType::NOT_EQUAL, "NOT_EQUAL"}};
    auto it = typeStrMap.find(compareType);
    return (it != typeStrMap.end()) ? it->second : "UNKNOWN";
}

std::string QfaTilingShapeCompare::CompareTypeToSerialSymbolString(const QfaCompareType &compareType) const
{
    static const std::map<QfaCompareType, std::string> symbolMap = {
        {QfaCompareType::EQUAL, "=="}, {QfaCompareType::GREATER, ">"},     {QfaCompareType::GREATER_EQUAL, ">="},
        {QfaCompareType::LESS, "<"},   {QfaCompareType::LESS_EQUAL, "<="}, {QfaCompareType::NOT_EQUAL, "!="}};
    auto it = symbolMap.find(compareType);
    return (it != symbolMap.end()) ? it->second : "UNKNOWN";
}

ge::graphStatus QfaTilingShapeCompare::GetExpectedShape(gert::Shape &shapeExpected,
                                                        const QfaTilingShapeCompareParam &param,
                                                        const std::string &funcName) const
{
    switch (layout_) {
        case QfaLayout::BSND:
            shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
            break;
        case QfaLayout::BNSD:
            shapeExpected = gert::Shape({param.B, param.N, param.S, param.D});
            break;
        case QfaLayout::TND:
            shapeExpected = gert::Shape({param.T, param.N, param.D});
            break;
        case QfaLayout::NTD:
            shapeExpected = gert::Shape({param.N, param.T, param.D});
            break;
        case QfaLayout::PA_BBND:
            shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
            break;
        case QfaLayout::PA_BNBD:
            shapeExpected = gert::Shape({param.Bn, param.N, param.Bs, param.D});
            break;
        case QfaLayout::PA_NZ:
            shapeExpected = gert::Shape({param.Bn, param.N, param.D / param.D0, param.Bs, param.D0});
            break;
        case QfaLayout::LSE_BNS:
            shapeExpected = gert::Shape({param.B, param.N, param.S1});
            break;
        case QfaLayout::LSE_NT:
            shapeExpected = gert::Shape({param.N, param.T});
            break;
        default:
            OP_LOGE(opName_, "[%s] Layout %s is unsupported.", funcName.c_str(),
                    QfaLayoutToSerialString(layout_).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

QfaCompareType QfaTilingShapeCompare::GetCompareType(const std::map<QfaAxis, QfaCompareType> &compareTypeMap,
                                                     const QfaAxis &axis) const
{
    auto it = compareTypeMap.find(axis);
    return (it != compareTypeMap.end()) ? it->second : QfaCompareType::EQUAL;
}

ge::graphStatus QfaTilingShapeCompare::GetCompareFunc(const QfaCompareType &compareType,
                                                      QfaCompareFunc<int64_t> &compareFunc,
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

ge::graphStatus QfaTilingShapeCompare::CompareShape(QfaTilingShapeCompareParam &param,
                                                    const std::string &funcName) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::vector<QfaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape_.GetDimNum() != shapeExpected.GetDimNum() || shape_.GetDimNum() != layoutAxes.size()) {
        OP_LOGE(opName_, "[%s] %s shape dimension is %zu, expected is %zu (layout %s).", funcName.c_str(),
                name_.c_str(), shape_.GetDimNum(), shapeExpected.GetDimNum(), QfaLayoutToSerialString(layout_).c_str());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape_.GetDimNum(); i++) {
        QfaAxis axis = layoutAxes[i];
        QfaCompareType compareType = GetCompareType(param.compareTypeMap, axis);
        QfaCompareFunc<int64_t> compareFunc;
        if (GetCompareFunc(compareType, compareFunc, funcName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (!compareFunc(shape_.GetDim(i), shapeExpected.GetDim(i))) {
            if (param.compareTypeMap.empty()) {
                OP_LOGE(opName_, "[%s] %s layout is %s, shape %s should be equal to %s.", funcName.c_str(),
                        name_.c_str(), QfaLayoutToSerialString(layout_).c_str(), GetQfaShapeStr(shape_).c_str(),
                        GetQfaShapeStr(shapeExpected).c_str());
            } else {
                OP_LOGE(opName_,
                        "[%s] %s layout is %s, shape is %s, expected is %s, "
                        "axis %s(%ld) should %s expected %ld.",
                        funcName.c_str(), name_.c_str(), QfaLayoutToSerialString(layout_).c_str(),
                        GetQfaShapeStr(shape_).c_str(), GetQfaShapeStr(shapeExpected).c_str(),
                        QfaAxisToSerialString(axis).c_str(), shape_.GetDim(i),
                        CompareTypeToSerialSymbolString(compareType).c_str(), shapeExpected.GetDim(i));
            }
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
