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

// 在custom命名空间里注册fused_gdn_gating算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
//   - schema 入参顺序：必选张量在前，带默认值的属性在后
//   - 张量 dtype / 属性默认值对齐 op_host/fused_gdn_gating_def.cpp
TORCH_LIBRARY(custom, m)
{
    m.def("npu_fused_gdn_gating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, "
          "float beta=1.0, float threshold=20.0) -> (Tensor g, Tensor beta_output)");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
