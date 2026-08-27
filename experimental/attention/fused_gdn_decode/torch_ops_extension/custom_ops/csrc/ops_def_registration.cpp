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

// 在custom命名空间里注册fused_gdn_decode算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
//   - schema 入参顺序：必选张量 + 必选标量在前，'*' 之后为可选张量与带默认值的属性
//   - state_ref为原地更新Tensor，需在schema中标记为Tensor(a!)
TORCH_LIBRARY(custom, m)
{
    m.def("npu_fused_gdn_decode(Tensor mixed_qkv, Tensor a, Tensor b, Tensor a_log, Tensor dt_bias, "
          "Tensor(a!) state_ref, Tensor ssm_state_indices, float scale, *, "
          "float softplus_threshold=20.0) -> Tensor");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
