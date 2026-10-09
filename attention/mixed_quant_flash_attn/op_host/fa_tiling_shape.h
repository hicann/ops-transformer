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
 * \file fa_tiling_shape.h
 * \brief
 */
#ifndef MIXED_QUANT_FLASH_ATTN_FA_TILING_SHAPE_H
#define MIXED_QUANT_FLASH_ATTN_FA_TILING_SHAPE_H

#include "mqfa_fa_tiling_info.h"

namespace optiling {
namespace mixed_quant_flash_attn {
template <typename T>
using CompareFunc = bool (*)(const T&, const T&);

enum class FaCompareType : uint32_t {
    EQUAL = 0,
    GREATER = 1,
    GREATER_EQUAL = 2,
    LESS = 3,
    LESS_EQUAL = 4,
    NOT_EQUAL = 5,
    IGNORE_INPUT = 6
};

struct FaTilingShapeCompareParam {
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
    std::map<FaAxis, FaCompareType> compareTypeMap = {};
};

[[maybe_unused]] static std::string GetShapeStr(gert::Shape shape)
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

class FaTilingShape {
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

public:
    FaTilingShape(const gert::Shape& shape, FaLayout layout, std::string name, std::string opName)
        : shape_(shape),
          layout_(layout),
          name_(name),
          opName_(opName) {};

public:
    const gert::Shape& shape_;
    FaLayout layout_;
    std::string name_;
    std::string opName_;

    size_t GetDimNum() const
    {
        return shape_.GetDimNum();
    }

    bool HasShapeB() const
    {
        return HasAxis(FaAxis::B);
    }
    bool HasShapeS() const
    {
        return HasAxis(FaAxis::S);
    }

    bool HasShapeN() const
    {
        return HasAxis(FaAxis::N);
    }
    bool HasShapeT() const
    {
        return HasAxis(FaAxis::T);
    }
    bool HasShapeD1() const
    {
        return HasAxis(FaAxis::D1);
    }
    bool HasShapeD0() const
    {
        return HasAxis(FaAxis::D0);
    }
    bool HasShapeD() const
    {
        if (HasAxis(FaAxis::D)) {
            return true;
        }
        if (HasShapeD1() && HasShapeD0()) {
            return true;
        }
        return false;
    }

    int64_t GetShapeB() const
    {
        return GetAxisNum(FaAxis::B);
    }
    int64_t GetShapeS() const
    {
        return GetAxisNum(FaAxis::S);
    }
    int64_t GetShapeN() const
    {
        return GetAxisNum(FaAxis::N);
    }
    int64_t GetShapeBlockSize() const
    {
        return GetAxisNum(FaAxis::Bs);
    }
    int64_t GetShapeBlockNum() const
    {
        return GetAxisNum(FaAxis::Bn);
    }

    int64_t GetShapeT() const
    {
        return GetAxisNum(FaAxis::T);
    }
    int64_t GetShapeD1() const
    {
        return GetAxisNum(FaAxis::D1);
    }
    int64_t GetShapeD0() const
    {
        return GetAxisNum(FaAxis::D0);
    }
    int64_t GetShapeD() const
    {
        if (HasAxis(FaAxis::D)) {
            return shape_.GetDim(GetAxisIdx(FaAxis::D));
        }
        if (HasShapeD1() && HasShapeD0()) {
            return GetShapeD1() * GetShapeD0();
        }
        return invalidDimValue_;
    }

    ge::graphStatus CheckHasShapeB(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::B, funcName);
    }
    ge::graphStatus CheckHasShapeS(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::S, funcName);
    }
    ge::graphStatus CheckHasShapeD(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::D, funcName);
    }
    ge::graphStatus CheckHasShapeN(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::N, funcName);
    }

    ge::graphStatus CheckHasShapeT(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::T, funcName);
    }

    ge::graphStatus CheckHasShapeBlockSize(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::Bs, funcName);
    }
    ge::graphStatus CheckHasShapeBlockNum(const std::string& funcName) const
    {
        return CheckHasAxis(FaAxis::Bn, funcName);
    }

private:
    bool HasAxis(const FaAxis& axis) const;
    size_t GetAxisIdx(const FaAxis& axis) const;
    int64_t GetAxisNum(const FaAxis& axis) const;
    ge::graphStatus CheckHasAxis(const FaAxis& axis, const std::string& funcName) const;
};

class FaTilingShapeCompare {
    static const std::map<FaCompareType, CompareFunc<int64_t>> compareFuncMap_;

public:
    FaTilingShapeCompare(const gert::Shape& shape, FaLayout layout, std::string name, std::string opName)
        : shape_(shape),
          layout_(layout),
          name_(name),
          opName_(opName) {};

public:
    const gert::Shape& shape_;
    FaLayout layout_;
    std::string name_;
    std::string opName_;

    std::string CompareTypeToSerialString(const FaCompareType compareType) const;
    std::string CompareTypeToSerialSymbolString(const FaCompareType& compareType) const;
    ge::graphStatus GetExpectedShape(gert::Shape& shapeExpected, const FaTilingShapeCompareParam& param,
                                     const std::string& funcName) const;
    FaCompareType GetCompareType(const std::map<FaAxis, FaCompareType>& compareTypeMap, const FaAxis& axis) const;
    ge::graphStatus GetCompareFunc(const FaCompareType& compareType, CompareFunc<int64_t>& compareFunc,
                                   const std::string& funcName) const;
    ge::graphStatus CompareShape(FaTilingShapeCompareParam& param, const std::string& funcName) const;
};
} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MIXED_QUANT_FLASH_ATTN_FA_TILING_SHAPE_H
