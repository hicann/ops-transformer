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
import torchair
from torchair.configs.compiler_config import CompilerConfig

# Register sparse_flash_mla and sparse_flash_mla_metadata via PTA
TORCH_EXT_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../torch_extension"
    )
)
if TORCH_EXT_PATH not in sys.path:
    sys.path.insert(0, TORCH_EXT_PATH)
from cann_ops_transformer.ops import sparse_flash_mla as _smla_registration  # noqa: F401
from cann_ops_transformer.ops import sparse_flash_mla_metadata as _metadata_registration  # noqa: F401


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        topk_value_mode,
        return_softmax_lse,
    ):
        npu_result, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=ori_topk_length,
            cmp_topk_length=cmp_topk_length,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
            **({"ori_win_right": ori_win_right} if ori_win_right is not None else {}),
            layout_q=layout_q,
            layout_kv=layout_kv,
            topk_value_mode=topk_value_mode,
            return_softmax_lse=return_softmax_lse,
        )
        return npu_result, softmax_lse


def call_npu(input_data):
    params = input_data["params"]
    metadata_input = input_data["metadata_input"]
    tensor_input = input_data["input"]
    print("用例参数: ", params)

    # metadata解析
    K1 = metadata_input["K1"]
    K = metadata_input["K"]
    cmp_ratio = metadata_input["cmp_ratio"]
    N1 = metadata_input["N1"]
    N2 = metadata_input["N2"]
    D = metadata_input["D"]
    B = metadata_input["B"]

    # tensor解析
    q = tensor_input["q"].npu()
    ori_kv = (
        tensor_input["ori_kv"].npu() if tensor_input["ori_kv"] is not None else None
    )
    cmp_kv = (
        tensor_input["cmp_kv"].npu() if tensor_input["cmp_kv"] is not None else None
    )
    ori_block_table = (
        tensor_input["ori_block_table"].npu()
        if tensor_input["ori_block_table"] is not None
        else None
    )
    cu_seqlens_q = (
        tensor_input["cu_seqlens_q"].npu()
        if "cu_seqlens_q" in tensor_input and tensor_input["cu_seqlens_q"] is not None
        else None
    )
    cu_seqlens_ori_kv = (
        tensor_input["cu_seqlens_ori_kv"].npu()
        if tensor_input["cu_seqlens_ori_kv"] is not None
        else None
    )
    cu_seqlens_cmp_kv = (
        tensor_input["cu_seqlens_cmp_kv"].npu()
        if tensor_input["cu_seqlens_cmp_kv"] is not None
        else None
    )
    used_seqused_q_flag = False
    if "seqused_q" in tensor_input and tensor_input["seqused_q"] is not None:
        seqused_q = tensor_input["seqused_q"].npu()
        used_seqused_q_flag = True
    else:
        seqused_q = None

    seqused_ori_kv = (
        tensor_input["seqused_ori_kv"].npu()
        if tensor_input["seqused_ori_kv"] is not None
        else None
    )
    seqused_cmp_kv = (
        tensor_input["seqused_cmp_kv"].npu()
        if tensor_input["seqused_cmp_kv"] is not None
        else None
    )
    cmp_residual_kv = (
        tensor_input["cmp_residual_kv"].npu()
        if tensor_input["cmp_residual_kv"] is not None
        else None
    )
    sinks = tensor_input["sinks"].npu()
    softmax_scale = tensor_input["softmax_scale"]
    ori_mask_mode = tensor_input["ori_mask_mode"]
    cmp_mask_mode = tensor_input["cmp_mask_mode"]
    ori_win_left = tensor_input["ori_win_left"]
    ori_win_right = tensor_input["ori_win_right"]
    layout_q = (
        tensor_input["layout_q"]
        if type(tensor_input["layout_q"]) == type("TND")
        else tensor_input["layout_q"][0]
    )
    layout_kv = tensor_input["layout_kv"]
    max_seqlen_q = metadata_input["max_seqlen_q"]
    max_seqlen_ori_kv = metadata_input["max_seqlen_ori_kv"]
    max_seqlen_cmp_kv = metadata_input["max_seqlen_cmp_kv"]
    ori_sparse_indices = tensor_input["ori_sparse_indices"]
    cmp_sparse_indices = tensor_input["cmp_sparse_indices"]
    cmp_block_table = tensor_input["cmp_block_table"]
    ori_topk_length = metadata_input.get("ori_topk_length")
    cmp_topk_length = metadata_input.get("cmp_topk_length")
    return_softmax_lse = params.get("return_softmax_lse")

    # 将需要上NPU的tensor搬到NPU
    if ori_sparse_indices is not None:
        ori_sparse_indices = ori_sparse_indices.npu()
    if cmp_sparse_indices is not None:
        cmp_sparse_indices = cmp_sparse_indices.npu()
    if cmp_block_table is not None:
        cmp_block_table = cmp_block_table.npu()
    if ori_topk_length is not None:
        ori_topk_length = ori_topk_length.npu()
    if cmp_topk_length is not None:
        cmp_topk_length = cmp_topk_length.npu()

    # 生成 metadata
    print("sparse_flash_mla_metadata...")
    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        ori_topk=K1 if ori_sparse_indices is not None else 0,
        cmp_topk=K if cmp_sparse_indices is not None else 0,
        cmp_ratio=cmp_ratio if cmp_ratio is not None else 1,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode if cmp_mask_mode is not None else 3,
        **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
        **({"ori_win_right": ori_win_right} if ori_win_right is not None else {}),
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=ori_kv != None,
        has_cmp_kv=cmp_kv != None,
    )

    # 统一调用 sparse_flash_mla，所有参数可选带默认值
    print("sparse_flash_mla...")
    npu_result, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q if layout_q == "TND" else None,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q if used_seqused_q_flag else None,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio if cmp_ratio is not None else 1,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode if cmp_mask_mode is not None else 0,
        **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
        **({"ori_win_right": ori_win_right} if ori_win_right is not None else {}),
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=1,
        return_softmax_lse=return_softmax_lse
        if return_softmax_lse is not None
        else False,
    )
    print("sparse_flash_mla end")

    torch.npu.synchronize()
    return npu_result, softmax_lse


def call_npu_graph(input_data, device_id=0):
    params = input_data["params"]
    metadata_input = input_data["metadata_input"]
    tensor_input = input_data["input"]
    print("用例参数(Graph模式): ", params)
    torch_npu.npu.set_device(device_id)

    torch._dynamo.reset()
    npu_mode = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(
        npu_mode, fullgraph=True, backend=npu_backend, dynamic=False
    )

    # metadata解析
    K1 = metadata_input["K1"]
    K = metadata_input["K"]
    cmp_ratio = metadata_input["cmp_ratio"]
    N1 = metadata_input["N1"]
    N2 = metadata_input["N2"]
    D = metadata_input["D"]
    B = metadata_input["B"]

    # tensor解析
    q = tensor_input["q"].npu()
    ori_kv = (
        tensor_input["ori_kv"].npu() if tensor_input["ori_kv"] is not None else None
    )
    cmp_kv = (
        tensor_input["cmp_kv"].npu() if tensor_input["cmp_kv"] is not None else None
    )
    ori_block_table = (
        tensor_input["ori_block_table"].npu()
        if tensor_input["ori_block_table"] is not None
        else None
    )
    cu_seqlens_q = (
        tensor_input["cu_seqlens_q"].npu()
        if "cu_seqlens_q" in tensor_input and tensor_input["cu_seqlens_q"] is not None
        else None
    )
    cu_seqlens_ori_kv = (
        tensor_input["cu_seqlens_ori_kv"].npu()
        if tensor_input["cu_seqlens_ori_kv"] is not None
        else None
    )
    cu_seqlens_cmp_kv = (
        tensor_input["cu_seqlens_cmp_kv"].npu()
        if tensor_input["cu_seqlens_cmp_kv"] is not None
        else None
    )
    used_seqused_q_flag = False
    if "seqused_q" in tensor_input and tensor_input["seqused_q"] is not None:
        seqused_q = tensor_input["seqused_q"].npu()
        used_seqused_q_flag = True
    else:
        seqused_q = None
    seqused_ori_kv = (
        tensor_input["seqused_ori_kv"].npu()
        if tensor_input["seqused_ori_kv"] is not None
        else None
    )
    seqused_cmp_kv = (
        tensor_input["seqused_cmp_kv"].npu()
        if tensor_input["seqused_cmp_kv"] is not None
        else None
    )
    cmp_residual_kv = (
        tensor_input["cmp_residual_kv"].npu()
        if tensor_input["cmp_residual_kv"] is not None
        else None
    )
    sinks = tensor_input["sinks"].npu()
    softmax_scale = tensor_input["softmax_scale"]
    ori_mask_mode = tensor_input["ori_mask_mode"]
    cmp_mask_mode = tensor_input["cmp_mask_mode"]
    ori_win_left = tensor_input["ori_win_left"]
    ori_win_right = tensor_input["ori_win_right"]
    layout_q = (
        tensor_input["layout_q"]
        if type(tensor_input["layout_q"]) == type("TND")
        else tensor_input["layout_q"][0]
    )
    layout_kv = tensor_input["layout_kv"]
    max_seqlen_q = metadata_input["max_seqlen_q"]
    max_seqlen_ori_kv = metadata_input["max_seqlen_ori_kv"]
    max_seqlen_cmp_kv = metadata_input["max_seqlen_cmp_kv"]
    ori_sparse_indices = tensor_input["ori_sparse_indices"]
    cmp_sparse_indices = tensor_input["cmp_sparse_indices"]
    cmp_block_table = tensor_input["cmp_block_table"]
    ori_topk_length = metadata_input.get("ori_topk_length")
    cmp_topk_length = metadata_input.get("cmp_topk_length")
    return_softmax_lse = params.get("return_softmax_lse")

    # 将需要上NPU的tensor搬到NPU
    if ori_sparse_indices is not None:
        ori_sparse_indices = ori_sparse_indices.npu()
    if cmp_sparse_indices is not None:
        cmp_sparse_indices = cmp_sparse_indices.npu()
    if cmp_block_table is not None:
        cmp_block_table = cmp_block_table.npu()
    if ori_topk_length is not None:
        ori_topk_length = ori_topk_length.npu()
    if cmp_topk_length is not None:
        cmp_topk_length = cmp_topk_length.npu()

    # 生成 metadata
    print("sparse_flash_mla_metadata...")
    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        ori_topk=K1 if ori_sparse_indices is not None else 0,
        cmp_topk=K if cmp_sparse_indices is not None else 0,
        cmp_ratio=cmp_ratio if cmp_ratio is not None else 1,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode if cmp_mask_mode is not None else 3,
        **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
        **({"ori_win_right": ori_win_right} if ori_win_right is not None else {}),
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=ori_kv != None,
        has_cmp_kv=cmp_kv != None,
    )

    torch.npu.synchronize()
    metadata.npu()

    # 通过编译后的Network调用 sparse_flash_mla
    print("sparse_flash_mla (Graph模式)...")
    npu_result, softmax_lse = npu_mode(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q if layout_q == "TND" else None,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q if used_seqused_q_flag else None,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio if cmp_ratio is not None else 1,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode if cmp_mask_mode is not None else 3,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=1,
        return_softmax_lse=return_softmax_lse
        if return_softmax_lse is not None
        else False,
    )
    print("sparse_flash_mla (Graph模式) end")

    torch.npu.synchronize()
    return npu_result, softmax_lse
