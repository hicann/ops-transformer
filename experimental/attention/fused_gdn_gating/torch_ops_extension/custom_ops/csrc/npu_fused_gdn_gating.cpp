/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t A_BATCH_DIM = 0;
const int64_t A_HEADS_DIM = 1;

// 工具函数，推导输出 g / beta_output 的 shape 与 dtype
//   - g:           FLOAT [1, batch, num_heads]
//   - beta_output: 与 b 同 dtype [1, batch, num_heads]
std::tuple<at::Tensor, at::Tensor> construct_fused_gdn_gating_output_tensors(const at::Tensor &a, const at::Tensor &b)
{
    for (auto i = 0; i < a.sizes().size(); i++) {
        TORCH_CHECK(a.size(i) > 0, "All values within a's shape should be greater than 0, but shape[", i, "] is ",
                    a.size(i));
    }

    int64_t batch = a.size(A_BATCH_DIM);
    int64_t num_heads = a.size(A_HEADS_DIM);

    at::SmallVector<int64_t, SIZE> output_size = {1, batch, num_heads};

    at::Tensor g = at::empty(output_size, a.options().dtype(at::kFloat));
    at::Tensor beta_output = at::empty(output_size, b.options());
    return std::tuple<at::Tensor, at::Tensor>(g, beta_output);
}

// step2, 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor> npu_fused_gdn_gating_npu(const at::Tensor &A_log, const at::Tensor &a,
                                                            const at::Tensor &b, const at::Tensor &dt_bias, double beta,
                                                            double threshold)
{
    TORCH_CHECK(A_log.dim() == DIM_ONE, "A_log should be 1-D [num_heads], got ", A_log.dim(), "D.");
    TORCH_CHECK(dt_bias.dim() == DIM_ONE, "dt_bias should be 1-D [num_heads], got ", dt_bias.dim(), "D.");
    TORCH_CHECK(a.dim() == DIM_TWO, "a should be 2-D [batch, num_heads], got ", a.dim(), "D.");
    TORCH_CHECK(b.dim() == DIM_TWO, "b should be 2-D [batch, num_heads], got ", b.dim(), "D.");
    TORCH_CHECK(b.size(A_BATCH_DIM) == a.size(A_BATCH_DIM) && b.size(A_HEADS_DIM) == a.size(A_HEADS_DIM),
                "a and b must have the same shape, got a=", a.sizes(), " b=", b.sizes());
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype, got a=", a.scalar_type(),
                " b=", b.scalar_type());
    TORCH_CHECK(A_log.scalar_type() == dt_bias.scalar_type(),
                "A_log and dt_bias must have the same dtype, got A_log=", A_log.scalar_type(),
                " dt_bias=", dt_bias.scalar_type());
    TORCH_CHECK(a.size(A_HEADS_DIM) == A_log.size(0),
                "a second dim (num_heads) must equal A_log first dim, got a.size(1)=", a.size(A_HEADS_DIM),
                " A_log.size(0)=", A_log.size(0));

    // construct the output tensors
    std::tuple<at::Tensor, at::Tensor> outputs = construct_fused_gdn_gating_output_tensors(a, b);
    at::Tensor g = std::get<0>(outputs);
    at::Tensor beta_output = std::get<1>(outputs);

    // torch schema的float参数在C++侧为double，aclnn接口形参为float，
    // 必须显式转换：ARM64下double/float共用v寄存器，不转换时callee读到
    // double低位32位（对0.5/1.0/20.0等值恰为0x00000000），属性全部变0.0
    float beta_val = static_cast<float>(beta);
    float threshold_val = static_cast<float>(threshold);

    // EXEC_NPU_CMD_V1 实参顺序 = 算子 IR 声明顺序（输入 -> 属性 -> 输出）
    EXEC_NPU_CMD_V1(aclnnFusedGdnGating, A_log, a, b, dt_bias, beta_val, threshold_val, g, beta_output);

    return std::tuple<at::Tensor, at::Tensor>(g, beta_output);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_gdn_gating_meta(const at::Tensor &A_log, const at::Tensor &a,
                                                             const at::Tensor &b, const at::Tensor &dt_bias,
                                                             double beta, double threshold)
{
    return construct_fused_gdn_gating_output_tensors(a, b);
}
} // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_fused_gdn_gating", &custom::npu_fused_gdn_gating_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_fused_gdn_gating", &custom::npu_fused_gdn_gating_meta);
}
