/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_QUANT_TORCH_ADPT_H
#define GROUPED_MATMUL_QUANT_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor grouped_matmul_quant(const at::Tensor& x, const at::Tensor& quantized_weight, const at::Tensor& weight_scale,
                                const at::Tensor& weight_offset, const c10::optional<at::Tensor>& group_list,
                                int64_t scale_group_size)
{
    constexpr int64_t FRACTAL_FLOAT16 = 16;
    int64_t m = x.size(0);
    int64_t fracN = quantized_weight.size(2);
    int64_t n = fracN * FRACTAL_FLOAT16;
    at::Tensor output = at::empty({m, n}, x.options());
    ACLNN_CMD(aclnnGroupedMatmulQuant, x, quantized_weight, weight_scale, weight_offset, group_list, scale_group_size,
              output);
    return output;
}

} // namespace vllm_ascend
#endif
