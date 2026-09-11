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

TORCH_LIBRARY(custom, m)
{
    m.def("npu_minimax_sparse_attention_split_kv("
          "Tensor query, Tensor key, Tensor value, Tensor? block_table, "
          "Tensor k2q_row_ptr, Tensor k2q_q_indices, Tensor k2q_slot_indices, "
          "Tensor actual_seq_lengths, Tensor actual_seq_lengths_kv, "
          "int num_key_value_heads, float scale_value, int block_size, int top_k, "
          "int inner_precise=4, bool softmax_lse_flag=False, str input_layout=\"TND\""
          ") -> (Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
