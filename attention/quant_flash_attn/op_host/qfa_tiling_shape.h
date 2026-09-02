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
 * \file qfa_tiling_shape.h
 * \brief QuantFlashAttn Tiling Shape
 */

#ifndef QUANT_FLASH_ATTN_QFA_TILING_SHAPE_H
#define QUANT_FLASH_ATTN_QFA_TILING_SHAPE_H

#include "qfa_tiling_info.h"

namespace optiling {
namespace quant_flash_attn {

template <typename T>
using QfaCompareFunc = bool (*)(const T &, const T &);

enum class QfaCompareType : uint32_t {
    EQUAL = 0,
    GREATER = 1,
    GREATER_EQUAL = 2,
    LESS = 3,
    LESS_EQUAL = 4,
    NOT_EQUAL = 5,
    IGNORE_INPUT = 6
};

struct QfaTilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t H = 1;
    int64_t T = 1;
    int64_t Bn = 1;
    int64_t Bs = 1;
    int64_t D0 = 16;
    int64_t S1 = 1;
    int64_t S2 = 1;
    int64_t CONST = 1;
    std::map<QfaAxis, QfaCompareType> compareTypeMap = {};
};

[[maybe_unused]] static std::string GetQfaShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

class QfaTilingShape {
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

public:
    QfaTilingShape(const gert::Shape &shape, QfaLayout layout, std::string name, std::string opName)
        : shape_(shape),
          layout_(layout),
          name_(name),
          opName_(opName) {};

public:
    const gert::Shape &shape_;
    QfaLayout layout_;
    std::string name_;
    std::string opName_;

    size_t GetDimNum() const
    {
        return shape_.GetDimNum();
    }

    bool HasShapeB() const
    {
        return HasAxis(QfaAxis::B);
    }
    bool HasShapeS() const
    {
        return HasAxis(QfaAxis::S);
    }
    bool HasShapeN() const
    {
        return HasAxis(QfaAxis::N);
    }
    bool HasShapeT() const
    {
        return HasAxis(QfaAxis::T);
    }
    bool HasShapeD1() const
    {
        return HasAxis(QfaAxis::D1);
    }
    bool HasShapeD0() const
    {
        return HasAxis(QfaAxis::D0);
    }
    bool HasShapeD() const
    {
        if (HasAxis(QfaAxis::D)) {
            return true;
        }
        if (HasShapeD1() && HasShapeD0()) {
            return true;
        }
        return false;
    }

    int64_t GetShapeB() const
    {
        return GetAxisNum(QfaAxis::B);
    }
    int64_t GetShapeS() const
    {
        return GetAxisNum(QfaAxis::S);
    }
    int64_t GetShapeN() const
    {
        return GetAxisNum(QfaAxis::N);
    }
    int64_t GetShapeBlockSize() const
    {
        return GetAxisNum(QfaAxis::Bs);
    }
    int64_t GetShapeBlockNum() const
    {
        return GetAxisNum(QfaAxis::Bn);
    }
    int64_t GetShapeT() const
    {
        return GetAxisNum(QfaAxis::T);
    }
    int64_t GetShapeD1() const
    {
        return GetAxisNum(QfaAxis::D1);
    }
    int64_t GetShapeD0() const
    {
        return GetAxisNum(QfaAxis::D0);
    }
    int64_t GetShapeD() const
    {
        if (HasAxis(QfaAxis::D)) {
            return shape_.GetDim(GetAxisIdx(QfaAxis::D));
        }
        if (HasShapeD1() && HasShapeD0()) {
            return GetShapeD1() * GetShapeD0();
        }
        return invalidDimValue_;
    }

    ge::graphStatus CheckHasShapeB(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::B, funcName);
    }
    ge::graphStatus CheckHasShapeS(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::S, funcName);
    }
    ge::graphStatus CheckHasShapeD(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::D, funcName);
    }
    ge::graphStatus CheckHasShapeN(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::N, funcName);
    }
    ge::graphStatus CheckHasShapeT(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::T, funcName);
    }
    ge::graphStatus CheckHasShapeBlockSize(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::Bs, funcName);
    }
    ge::graphStatus CheckHasShapeBlockNum(const std::string &funcName) const
    {
        return CheckHasAxis(QfaAxis::Bn, funcName);
    }

private:
    bool HasAxis(const QfaAxis &axis) const;
    size_t GetAxisIdx(const QfaAxis &axis) const;
    int64_t GetAxisNum(const QfaAxis &axis) const;
    ge::graphStatus CheckHasAxis(const QfaAxis &axis, const std::string &funcName) const;
};

class QfaTilingShapeCompare {
    static const std::map<QfaCompareType, QfaCompareFunc<int64_t>> compareFuncMap_;

public:
    QfaTilingShapeCompare(const gert::Shape &shape, QfaLayout layout, std::string name, std::string opName)
        : shape_(shape),
          layout_(layout),
          name_(name),
          opName_(opName) {};

public:
    const gert::Shape &shape_;
    QfaLayout layout_;
    std::string name_;
    std::string opName_;

    std::string CompareTypeToSerialString(const QfaCompareType compareType) const;
    std::string CompareTypeToSerialSymbolString(const QfaCompareType &compareType) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected, const QfaTilingShapeCompareParam &param,
                                     const std::string &funcName) const;
    QfaCompareType GetCompareType(const std::map<QfaAxis, QfaCompareType> &compareTypeMap, const QfaAxis &axis) const;
    ge::graphStatus GetCompareFunc(const QfaCompareType &compareType, QfaCompareFunc<int64_t> &compareFunc,
                                   const std::string &funcName) const;
    ge::graphStatus CompareShape(QfaTilingShapeCompareParam &param, const std::string &funcName) const;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QFA_TILING_SHAPE_H
