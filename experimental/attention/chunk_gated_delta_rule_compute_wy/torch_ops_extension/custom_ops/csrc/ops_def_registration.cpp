/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册chunk_gated_delta_rule_compute_wy算子。
// schema 入参顺序：必选张量 + 必选标量在前。chunk_size 保持位置参数（而非 '*' 之后的
// 带默认值属性），这样调用形式与 vllm-ascend 的
// torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, 64) 完全一致，
// 便于直接对拍。
TORCH_LIBRARY(custom, m)
{
    m.def("npu_chunk_gated_delta_rule_compute_wy(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
          "int chunk_size) -> (Tensor q_kernel, Tensor k_kernel, Tensor w_kernel, Tensor u_kernel, "
          "Tensor g_kernel)");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
