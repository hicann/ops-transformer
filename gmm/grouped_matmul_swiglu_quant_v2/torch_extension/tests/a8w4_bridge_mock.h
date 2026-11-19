// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

constexpr int64_t MAX_DIM_NUM = 5;
constexpr int64_t FP4_IN_INT8 = 2;
constexpr int64_t PENULTIMATE_DIM = 2;
enum aclDataType {
    ACL_FLOAT = 0,
    ACL_FLOAT8_E4M3FN = 36,
    ACL_FLOAT8_E8M0 = 37,
    ACL_FLOAT4_E2M1 = 40
};
namespace at {
enum class ScalarType {
    Byte,
    Char,
    Long = 4,
    Float = 6,
    Float8_e4m3fn = 24,
    Float4_e2m1fn_x2 = 42,
    Float8_e8m0fnu = 43
};
struct Options {
    ScalarType type;
    Options dtype(ScalarType value) const
    {
        return {value};
    }
};
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    ScalarType type = ScalarType::Byte;
    bool valid = false;
    bool zeroed = false;
    bool materialized = false;
    int identity = 0;
    int dev = 0;
    Tensor() = default;
    Tensor(std::initializer_list<int64_t> dims, ScalarType dtype, int id = 0)
        : shape(dims),
          type(dtype),
          valid(true),
          identity(id)
    {
        ResetStrides();
    }
    void ResetStrides()
    {
        strides.resize(shape.size());
        int64_t step = 1;
        for (int64_t index = dim() - 1; index >= 0; --index) {
            strides[index] = step;
            step *= std::max<int64_t>(shape[index], 1);
        }
    }
    bool defined() const
    {
        return valid;
    }
    int64_t dim() const
    {
        return shape.size();
    }
    int64_t size(int64_t index) const
    {
        return shape.at(index < 0 ? dim() + index : index);
    }
    int64_t numel() const
    {
        int64_t count = 1;
        for (int64_t extent : shape) {
            count *= extent;
        }
        return count;
    }
    int64_t stride(int64_t index) const
    {
        return strides.at(index < 0 ? dim() + index : index);
    }
    Tensor transpose(int64_t first, int64_t second) const
    {
        Tensor result = *this;
        first = first < 0 ? dim() + first : first;
        second = second < 0 ? dim() + second : second;
        std::swap(result.shape.at(first), result.shape.at(second));
        std::swap(result.strides.at(first), result.strides.at(second));
        return result;
    }
    const std::vector<int64_t>& sizes() const
    {
        return shape;
    }
    ScalarType scalar_type() const
    {
        return type;
    }
    int device() const
    {
        return dev;
    }
    Options options() const
    {
        return {type};
    }
    Tensor view(ScalarType value) const
    {
        Tensor result = *this;
        result.type = value;
        return result;
    }
    Tensor contiguous() const
    {
        Tensor result = *this;
        result.materialized = true;
        result.ResetStrides();
        return result;
    }
};
using TensorList = std::vector<Tensor>;
using IntArrayRef = std::vector<int64_t>;
inline Tensor empty(std::initializer_list<int64_t> shape, Options options)
{
    return {shape, options.type};
}
inline Tensor zeros(std::initializer_list<int64_t> shape, Options options)
{
    auto result = empty(shape, options);
    result.zeroed = true;
    return result;
}
} // namespace at
namespace c10 {
template <class T>
using optional = std::optional<T>;
template <class T, int64_t N>
using SmallVector = std::vector<T>;
constexpr auto nullopt = std::nullopt;
struct Device {
    explicit Device(int value) {}
};
struct OptionalDeviceGuard {
    explicit OptionalDeviceGuard(Device value) {}
};
} // namespace c10
inline aclDataType ConvertToAclDataType(at::ScalarType type)
{
    if (type == at::ScalarType::Float8_e4m3fn)
        return ACL_FLOAT8_E4M3FN;
    if (type == at::ScalarType::Float4_e2m1fn_x2)
        return ACL_FLOAT4_E2M1;
    if (type == at::ScalarType::Float8_e8m0fnu)
        return ACL_FLOAT8_E8M0;
    return ACL_FLOAT;
}
inline aclDataType GetAclDataType(int64_t value)
{
    constexpr int64_t ACL_OFFSET = 256;
    return value >= ACL_OFFSET ? static_cast<aclDataType>(value - ACL_OFFSET) :
                                 ConvertToAclDataType(static_cast<at::ScalarType>(value));
}
struct TensorWrapper {
    const at::Tensor& tensor_;
    aclDataType dtype;
};
struct TensorListWrapper {
    const at::TensorList& tensor_list_;
    aclDataType dtype;
};
struct Record {
    bool x_scale_materialized = false;
    int calls = 0;
    int smooth = 0;
    int groups = 0;
    int64_t dequant_dtype = -1;
    int64_t dequant_mode = -1;
    int64_t quant_mode = -1;
    double clamp = 0;
    double alpha = 0;
    double bias = 0;
    aclDataType weight_type = ACL_FLOAT;
    std::vector<int64_t> weight_shape;
} record;
template <class... Args>
void record_call(Args&... args)
{
    constexpr size_t WEIGHT = 1, X_SCALE = 5, SMOOTH = 6, GROUPS = 7, DEQUANT = 8, DEQUANT_TYPE = 9;
    constexpr size_t QUANT = 10, CLAMP = 14, ALPHA = 15, BIAS = 16;
    auto tuple = std::tie(args...);
    ++record.calls;
    record.x_scale_materialized = std::get<X_SCALE>(tuple).tensor_.materialized;
    record.weight_type = std::get<WEIGHT>(tuple).dtype;
    record.weight_shape = std::get<WEIGHT>(tuple).tensor_list_[0].shape;
    record.smooth = std::get<SMOOTH>(tuple).identity;
    record.groups = std::get<GROUPS>(tuple).identity;
    record.dequant_mode = std::get<DEQUANT>(tuple);
    record.dequant_dtype = std::get<DEQUANT_TYPE>(tuple);
    record.quant_mode = std::get<QUANT>(tuple);
    record.clamp = std::get<CLAMP>(tuple);
    record.alpha = std::get<ALPHA>(tuple);
    record.bias = std::get<BIAS>(tuple);
}
inline void require(bool value)
{
    if (!value)
        throw std::runtime_error("contract assertion");
}
#define TORCH_CHECK(c, ...) \
    do { \
        if (!(c)) \
            throw std::runtime_error("bridge check"); \
    } while (0)
#define ACLNN_CMD(api, ...) record_call(__VA_ARGS__)
