/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_ops_def.cpp
 * \brief
 */

#define Py_LIMITED_API_VERSION 0x03080000
#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"

#include <vector>

extern "C" {
PyObject *PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace npu_ops_transformer_ext {

TORCH_LIBRARY(npu_ops_transformer_ext, m)
{
    // set def like the rule below: m.def("dummy(Tensor x) -> Tensor");
    m.def("rope_matrix(Tensor x, Tensor y, Tensor sin, Tensor cos) -> Tensor");
    m.def("mambav2_causal_conv1d(Tensor x, Tensor w, Tensor bias) -> Tensor");
    m.def("mambav2_rmsnormgated(Tensor x, Tensor z, Tensor w, int G, float E) -> Tensor");
    m.def("mambav2_chunk_cumsum(Tensor at, Tensor dt, Tensor dt_bias, Tensor dt_mask) -> (Tensor, Tensor, Tensor)");
    m.def("mambav2_chunk_state(Tensor dt_out, Tensor dacs, Tensor bt, Tensor xt) -> Tensor");
    m.def("mambav2_chunk_state_passing(Tensor dacs, Tensor init_states, Tensor states, Tensor ct) -> (Tensor, Tensor)");
    m.def("mambav2_chunk_scan(Tensor ct, Tensor bt, Tensor xt, Tensor dt, Tensor states, Tensor dacs, Tensor dtout) -> "
          "Tensor");
    m.def("grouped_matmul_quant(Tensor x, Tensor quantized_weight, Tensor weight_scale, Tensor weight_offset, "
          "Tensor? group_list, int scale_group_size) -> Tensor");
}

} // namespace npu_ops_transformer_ext
