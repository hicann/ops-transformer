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
import numpy as np


def _load_inputs_to_npu(input_dict):
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value

    return {key: to_npu(value) for key, value in input_dict.items()}


def call_npu_eager(torch_tensor_dict, params):
    kwargs = {
        "query": torch_tensor_dict.get("query"),
        "key": torch_tensor_dict.get("key_cache"),
        "value": torch_tensor_dict.get("value_cache"),
        "sparse_indices": torch_tensor_dict.get("sparse_indices"),
        "scale_value": params.get("scalevalue"),
        "block_table": torch_tensor_dict.get("block_table")
        if params.get("layout_kv") == "PA_BSND"
        else None,
        "actual_seq_lengths_query": torch.tensor(
            params["actual_seq_q"], dtype=torch.int32
        ).to("npu"),
        "actual_seq_lengths_kv": torch.tensor(
            params["actual_seq_kv"], dtype=torch.int32
        ).to("npu"),
        "query_rope": torch_tensor_dict.get("query_rope"),
        "key_rope": torch_tensor_dict.get("key_rope_cache"),
        "sparse_block_size": params.get("sparse_blocksize", 1),
        "layout_query": params.get("layout_query"),
        "layout_kv": params.get("layout_kv"),
        "sparse_mode": params.get("sparsemode", 3),
        "pre_tokens": params.get("pre_tokens", (1 << 63) - 1),
        "next_tokens": params.get("next_tokens", (1 << 63) - 1),
        "attention_mode": params.get("attention_mode", 2),
        "return_softmax_lse": params.get("return_softmax_lse", False),
        "sinks": torch_tensor_dict.get("sinks"),
    }
    return torch_npu.npu_sparse_flash_attention(**kwargs)


def call_npu(input_tensor_dict, params, device_id=0):
    torch_npu.npu.set_device(device_id)
    tensor_keys = [
        "query",
        "key_cache",
        "value_cache",
        "sparse_indices",
        "block_table",
        "query_rope",
        "key_rope_cache",
        "sinks",
    ]
    filtered_input_dict = {
        k: input_tensor_dict.get(k)
        for k in tensor_keys
        if input_tensor_dict.get(k) is not None
    }
    torch_tensor_dict = _load_inputs_to_npu(filtered_input_dict)
    npu_result = call_npu_eager(torch_tensor_dict, params)
    # 路径3:
    # npu_result = call_npu_graph(torch_tensor_dict, params)
    torch.npu.synchronize()
    return npu_result


class Network(torch.nn.Module):
    # 图模式下的最小封装，便于 torch.compile 捕获整个调用。
    def forward(
        self,
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        return_softmax_lse,
        sinks,
    ):
        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "scale_value": scale_value,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "query_rope": query_rope,
            "key_rope": key_rope,
            "sparse_block_size": sparse_block_size,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparse_mode": sparse_mode,
            "pre_tokens": pre_tokens,
            "next_tokens": next_tokens,
            "attention_mode": attention_mode,
            "return_softmax_lse": return_softmax_lse,
            "sinks": sinks,
        }
        return torch_npu.npu_sparse_flash_attention(**kwargs)


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

    print("npu_sparse_flash_attention (graph mode) ...")
    return npu_model(
        query=torch_tensor_dict.get("query"),
        key=torch_tensor_dict.get("key_cache"),
        value=torch_tensor_dict.get("value_cache"),
        sparse_indices=torch_tensor_dict.get("sparse_indices"),
        scale_value=params.get("scalevalue"),
        block_table=torch_tensor_dict.get("block_table")
        if params.get("layout_kv") == "PA_BSND"
        else None,
        actual_seq_lengths_query=torch.tensor(
            params["actual_seq_q"], dtype=torch.int32
        ).to("npu"),
        actual_seq_lengths_kv=torch.tensor(
            params["actual_seq_kv"], dtype=torch.int32
        ).to("npu"),
        query_rope=torch_tensor_dict.get("query_rope"),
        key_rope=torch_tensor_dict.get("key_rope_cache"),
        sparse_block_size=params.get("sparse_blocksize", 1),
        layout_query=params.get("layout_query"),
        layout_kv=params.get("layout_kv"),
        sparse_mode=params.get("sparsemode", 3),
        pre_tokens=params.get("pre_tokens", (1 << 63) - 1),
        next_tokens=params.get("next_tokens", (1 << 63) - 1),
        attention_mode=params.get("attention_mode", 2),
        return_softmax_lse=params.get("return_softmax_lse", False),
        sinks=torch_tensor_dict.get("sinks"),
    )
