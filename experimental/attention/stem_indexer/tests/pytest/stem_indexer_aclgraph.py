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

import os
import sys

import torch
import torch_npu

import stem_indexer_golden  # noqa: E402
import custom_ops  # noqa: E402, F401

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    _HAS_TORCHAIR = True
except ImportError:
    _HAS_TORCHAIR = False


class StemIndexerNetwork(torch.nn.Module):
    """torch.compile 图模式包装器，将 stem_indexer 算子调用封装为 nn.Module forward。"""

    def __init__(self):
        super(StemIndexerNetwork, self).__init__()

    def forward(
        self,
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        metadata,
        causal,
        stem_block_size,
        stem_stride,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
        topk_score_precision,
    ):
        sparse_indices, sparse_seq_len = torch.ops.custom.npu_stem_indexer(
            qflat,
            kflat,
            vbias,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens=num_prompt_tokens,
            metadata=metadata,
            causal=causal,
            stem_block_size=stem_block_size,
            stem_stride=stem_stride,
            alpha=alpha,
            initial_blocks=initial_blocks,
            window_size=window_size,
            k_block_num_rate_medium=k_block_num_rate_medium,
            k_block_num_bias_medium=k_block_num_bias_medium,
            k_block_num_rate_large=k_block_num_rate_large,
            k_block_num_bias_large=k_block_num_bias_large,
            topk_score_precision=topk_score_precision,
        )
        return sparse_indices, sparse_seq_len


def build_metadata(case, npu_inputs):
    metadata_attrs = stem_indexer_golden.get_metadata_attrs(case)
    return torch.ops.custom.npu_stem_indexer_metadata(
        npu_inputs["q_seq_lens"],
        npu_inputs["kv_seq_lens"],
        case["q_heads"],
        case["kv_heads"],
        **metadata_attrs,
    )


def _build_graph_network():
    """构建 torch.compile 编译后的 StemIndexerNetwork（每次调用重建，保证用例隔离）。"""
    if not _HAS_TORCHAIR:
        raise ImportError("graph 模式需要 torchair，请先安装")

    torch._dynamo.reset()
    net = StemIndexerNetwork().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    compiled_net = torch.compile(
        net, fullgraph=True, backend=npu_backend, dynamic=False
    )
    return compiled_net


def call_stem_indexer_graph(case, npu_inputs, device_id=0):
    """graph 模式：通过 torch.compile + torchair 编译后的 Network 调用 stem_indexer。"""
    torch_npu.npu.set_device(device_id)
    compiled_net = _build_graph_network()

    attrs = stem_indexer_golden.get_call_attrs(case)
    metadata = build_metadata(case, npu_inputs)

    print("StemIndexer (Graph模式)...")
    npu_result = compiled_net(
        npu_inputs["qflat"],
        npu_inputs["kflat"],
        npu_inputs["vbias"],
        npu_inputs["q_seq_lens"],
        npu_inputs["kv_seq_lens"],
        npu_inputs["num_prompt_tokens"],
        metadata,
        attrs["causal"],
        attrs["stem_block_size"],
        attrs["stem_stride"],
        attrs["alpha"],
        attrs["initial_blocks"],
        attrs["window_size"],
        attrs["k_block_num_rate_medium"],
        attrs["k_block_num_bias_medium"],
        attrs["k_block_num_rate_large"],
        attrs["k_block_num_bias_large"],
        attrs["topk_score_precision"],
    )
    print("StemIndexer (Graph模式) end")
    return npu_result