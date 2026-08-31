#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

import os
import numpy as np
import time
import math
import random
from utils import get_torch_dtype


def _load_inputs_to_npu(input_dict):
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value

    return {key: to_npu(value) for key, value in input_dict.items()}


def _get_kv_torch_dtype(kv_dtype_str):
    if kv_dtype_str == "hifloat8":
        return torch_npu.hifloat8
    return None


def call_npu_eager(torch_tensor_dict, params):
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        torch_tensor_dict["query_cache"],
        torch_tensor_dict["key_cache"],
        torch_tensor_dict["value_cache"],
        sparse_indices=torch_tensor_dict["sparse_indices"],
        scale_value=params.get("scalevalue", 1),
        sparse_block_size=params.get("sparse_blocksize", 0),
        key_quant_mode=params.get("key_quant_mode", 2),
        value_quant_mode=params.get("value_quant_mode", 2),
        key_dequant_scale=None,
        value_dequant_scale=None,
        block_table=torch_tensor_dict["block_table"]
        if params["layout_kv"] == "PA_BSND"
        else None,
        actual_seq_lengths_query=torch.tensor(
            params["actualseqlengths"], dtype=torch.int32
        ).to("npu"),
        actual_seq_lengths_kv=torch.tensor(
            params["actualseqlengthskv"], dtype=torch.int32
        ).to("npu"),
        layout_query=params.get("layout_query", "BSND"),
        layout_kv=params.get("layout_kv", "PA_BSND"),
        sparse_mode=params.get("sparsemode", 0),
        attention_mode=params.get("attention_mode", 2),
        quant_scale_repo_mode=params.get("quant_scale_repo_mode", 1),
        tile_size=params.get("tile_size", 128),
        rope_head_dim=params.get("rope_head_dim", 64),
        pre_tokens=params.get("pre_tokens", (1 << 63) - 1),
        next_tokens=params.get("next_tokens", (1 << 63) - 1),
        key_dtype=_get_kv_torch_dtype(params["dtype_input"]["key"]),
        value_dtype=_get_kv_torch_dtype(params["dtype_input"]["value"]),
    )


def call_npu(input_tensor_dict, params):
    torch_npu.npu.set_device(0)
    torch_tensor_dict = _load_inputs_to_npu(input_tensor_dict)
    npu_result = call_npu_eager(torch_tensor_dict, params)
    # 路径3
    # npu_result = call_npu_graph(torch_tensor_dict, params)
    torch.npu.synchronize()
    return npu_result


class Network(torch.nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        key_dtype,
        value_dtype,
        pre_tokens,
        next_tokens,
    ):
        return torch_npu.npu_kv_quant_sparse_flash_attention(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=scale_value,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
            key_dequant_scale=key_dequant_scale,
            value_dequant_scale=value_dequant_scale,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_block_size=sparse_block_size,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            attention_mode=attention_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            rope_head_dim=rope_head_dim,
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )


def call_npu_graph(torch_tensor_dict, params):
    torch._dynamo.reset()
    npu_model = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_model = torch.compile(
        npu_model, fullgraph=True, backend=npu_backend, dynamic=False
    )

    print("npu_kv_quant_sparse_flash_attention (graph mode) ...")
    return npu_model(
        torch_tensor_dict["query_cache"],
        torch_tensor_dict["key_cache"],
        torch_tensor_dict["value_cache"],
        sparse_indices=torch_tensor_dict["sparse_indices"],
        scale_value=params.get("scalevalue", 1),
        sparse_block_size=params.get("sparse_blocksize", 0),
        key_quant_mode=params.get("key_quant_mode", 2),
        value_quant_mode=params.get("value_quant_mode", 2),
        key_dequant_scale=None,
        value_dequant_scale=None,
        block_table=torch_tensor_dict["block_table"]
        if params["layout_kv"] == "PA_BSND"
        else None,
        actual_seq_lengths_query=torch.tensor(
            params["actualseqlengths"], dtype=torch.int32
        ).to("npu"),
        actual_seq_lengths_kv=torch.tensor(
            params["actualseqlengthskv"], dtype=torch.int32
        ).to("npu"),
        layout_query=params.get("layout_query", "BSND"),
        layout_kv=params.get("layout_kv", "PA_BSND"),
        sparse_mode=params.get("sparsemode", 0),
        attention_mode=params.get("attention_mode", 2),
        quant_scale_repo_mode=params.get("quant_scale_repo_mode", 1),
        tile_size=params.get("tile_size", 128),
        rope_head_dim=params.get("rope_head_dim", 64),
        pre_tokens=params.get("pre_tokens", (1 << 63) - 1),
        next_tokens=params.get("next_tokens", (1 << 63) - 1),
        key_dtype=_get_kv_torch_dtype(params["dtype_input"]["key"]),
        value_dtype=_get_kv_torch_dtype(params["dtype_input"]["value"]),
    )
