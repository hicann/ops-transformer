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
    m.def("npu_turbo_quant_sparse_flash_attention("
          "Tensor query, Tensor key, Tensor value, Tensor sparse_indices, *, "
          "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
          "Tensor block_table, Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
          "float scale_value=1.0, int key_quant_mode=3, int value_quant_mode=3, "
          "int sparse_block_size=1, str layout_query='TND', str layout_kv='PA_BSND', "
          "int sparse_mode=3, int pre_tokens=9223372036854775807, "
          "int next_tokens=9223372036854775807, int attention_mode=2, "
          "int quant_scale_repo_mode=1, int tile_size=128, int rope_head_dim=64, "
          "bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
