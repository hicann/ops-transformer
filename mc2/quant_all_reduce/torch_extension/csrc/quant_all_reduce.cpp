/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file quant_reduce_scatter.cpp
 * \brief
 */
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

static const int DIM_TWO = 2;
static const int DIM_THREE = 3;
static const int DIM_FOUR = 4;
static const int NUM_64 = 64;
static const int NUM_128 = 128;
static const int H_LOWER_LIMIT = 1024;
static const int H_UPPER_LIMIT = 8192;

// worldSize
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8};
// x valid dtype
const std::set<int64_t> SUPPORT_X_DTYPE_LIST{aclDataType::ACL_INT8,        aclDataType::ACL_HIFLOAT8,
                                             aclDataType::ACL_FLOAT8_E5M2, aclDataType::ACL_FLOAT8_E4M3FN,
                                             aclDataType::ACL_FLOAT4_E1M2, aclDataType::ACL_FLOAT4_E2M1};
// scales valid dtype
const std::set<int64_t> SUPPORT_SCALES_DTYPE_LIST{aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT8_E8M0};

inline TensorWrapper make_wrapper(const at::Tensor &tensor, aclDataType tensorAcltype)
{
    return {tensor, tensorAcltype};
}

at::Tensor NpuQuantAllReduce(const at::Tensor &context, const at::Tensor &x, const at::Tensor &scales,
                             int64_t hcclBufferSize, int64_t worldSize, c10::optional<std::string> reduceOp,
                             c10::optional<int64_t> outputDtype, c10::optional<int64_t> xDtype,
                             c10::optional<int64_t> scalesDtype)
{
    // 校验空tensor
    TORCH_CHECK(x.defined(), "The input tensor x can not be None.");
    TORCH_CHECK(scales.defined(), "The input tensor scales can not be None.");
    // 校验x的shape, 2维(bs, h)或3维(b, s, h)
    TORCH_CHECK(x.dim() == DIM_TWO || x.dim() == DIM_THREE,
                "The input x tensor shape is required to be 2 or 3 dim, but the actual input shape is ", x.dim());
    // x不能为空tensor
    if (x.dim() == DIM_TWO) {
        TORCH_CHECK(x.size(0) != 0 && x.size(1) != 0, "The input 2 dim tensor x can not be empty tensor");
    } else if (x.dim() == DIM_THREE) {
        TORCH_CHECK(x.size(0) != 0 && x.size(1) != 0 && x.size(DIM_TWO) != 0,
                    "The input 3 dim tensor x can not be empty tensor");
    }

    // 校验x的dtype
    if (xDtype.has_value()) {
        TORCH_CHECK(SUPPORT_X_DTYPE_LIST.find(GetAclDataType(xDtype.value())) != SUPPORT_X_DTYPE_LIST.end(),
                    "The optional parameter x_dtype only supports int8/hifloat8/float8_e4m3fn/float8_e5m2, but now is ",
                    xDtype.value());
    }

    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(worldSize) != SUPPORT_WORLD_SIZE_LIST.end(),
                "The world_size should be in ", c10::Join(", ", SUPPORT_WORLD_SIZE_LIST), ", but the actual value is ",
                worldSize);

    // x.shape是(bs, h)或者(b, s, h)，所以第0维可能是bs，也可能是b
    int64_t axisBs = x.size(0);
    if (x.dim() == DIM_THREE) {
        axisBs = axisBs * x.size(1);
    }
    TORCH_CHECK(axisBs % worldSize == 0, "The x BS-axis should be divisible by world_size");

    // (bs, h)或者(b, s, h), h范围在[1024, 8192]内，且h满足128对齐
    uint32_t axisH = (x.dim() == DIM_THREE ? 2 : 1);
    TORCH_CHECK(x.size(axisH) >= H_LOWER_LIMIT && x.size(axisH) <= H_UPPER_LIMIT && x.size(axisH) % NUM_128 == 0,
                "The x H-axis should be in [1024, 8192] and divisible by 128");

    // 校验scales的shape
    TORCH_CHECK(scales.dim() == DIM_TWO || scales.dim() == DIM_THREE || scales.dim() == DIM_FOUR,
                "The input scales tensor shape is required to be equal to x in TG QuantMode, "
                "or be equal to x plus 1 in MX QuantMode, but the actual input scales shape is ",
                scales.dim());
    // 校验scales是否为空tensor
    if (scales.dim() == DIM_TWO) {
        TORCH_CHECK(scales.size(0) != 0 && scales.size(1) != 0,
                    "The input 2 dim tensor scales can not be empty tensor");
    } else if (scales.dim() == DIM_THREE) {
        TORCH_CHECK(scales.size(0) != 0 && scales.size(1) != 0 && scales.size(DIM_TWO) != 0,
                    "The input 3 dim tensor scales can not be empty tensor");
    } else if (scales.dim() == DIM_FOUR) {
        TORCH_CHECK(
            scales.size(0) != 0 && scales.size(1) != 0 && scales.size(DIM_TWO) != 0 && scales.size(DIM_THREE) != 0,
            "The input 4 dim tensor scales can not be empty tensor");
    }

    // 校验scales的dtype
    if (scalesDtype.has_value()) {
        TORCH_CHECK(
            SUPPORT_SCALES_DTYPE_LIST.find(GetAclDataType(scalesDtype.value())) != SUPPORT_SCALES_DTYPE_LIST.end(),
            "The optional parameter scales_dtype only supports float/float_e8m0, but now is ", scalesDtype.value());
    }

    // pta主要是为了推导output的shape和dtype，如果这里的output_dtype没有传入，则默认是bf16
    at::ScalarType outputDefaultDtype = at::kBFloat16;
    if (outputDtype.has_value()) {
        // 这里应该校验output_dtype，但是目前没有bfloat16的类型定义。怕影响正常功能，因此这里不校验了
        aclDataType outputAclDtype = GetAclDataType(outputDtype.value());
        if (outputAclDtype == ACL_FLOAT16) {
            outputDefaultDtype = at::kHalf;
        } else if (outputAclDtype == ACL_BF16) {
            outputDefaultDtype = at::kBFloat16;
        } else if (outputAclDtype == ACL_FLOAT) {
            outputDefaultDtype = at::kFloat;
        } else {
            TORCH_CHECK(false, "unsupported output dtype: ", static_cast<int32_t>(outputAclDtype));
        }
    }
    auto outputSize = x.sizes();
    // 输出的outputTensor需要自己推导，outputTensor按照实际的shape和dtype去创建
    at::Tensor outputTensor = at::empty(outputSize, x.options().dtype(outputDefaultDtype));

    // attr
    std::string reduceOpValueStr = std::string(reduceOp.value_or("sum"));
    char *reduceOpPtr = const_cast<char *>(reduceOpValueStr.c_str());

    // 自定义dtype，使用wrapper封装，相当于打一个标签，标记真实的属性，让aclnn接口识别到传入tensor具体的dtype
    aclDataType xAclDtype = xDtype.has_value() ? GetAclDataType(xDtype.value()) : ConvertToAclDataType(x.scalar_type());
    aclDataType scalesAclDtype =
        scalesDtype.has_value() ? GetAclDataType(scalesDtype.value()) : ConvertToAclDataType(scales.scalar_type());
    TensorWrapper xWrapper = make_wrapper(x, xAclDtype);
    TensorWrapper scalesWrapper = make_wrapper(scales, scalesAclDtype);

    // 前面的wrapper打包传进去之后，这里直接调用aclnn接口
    ACLNN_CMD(aclnnQuantAllReduce, context, xWrapper, scalesWrapper, hcclBufferSize, worldSize, reduceOpPtr,
              outputTensor);
    return outputTensor;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_quant_all_reduce", &NpuQuantAllReduce, "npu_quant_all_reduce");
}

} // namespace op_api
