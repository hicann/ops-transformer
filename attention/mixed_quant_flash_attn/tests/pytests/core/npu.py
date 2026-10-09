#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
import cann_ops_transformer

# from cann_ops_transformer.ops import mixed_quant_flash_attn
from .cpu import get_axis_from_tensor
from .meta_cpu.mixed_quant_flash_attn_metadata_op import MixedQuantFlashAttnMetadataOp

# 'esl' 'npu'
RUN_METHOD = "esl"


def _to_npu_quant_dtype(
    tensor: torch.Tensor, quant_compute_mode: int, is_scale: bool
) -> torch.Tensor:
    """Convert generated placeholder dtype to the NPU-expected dtype for the
    mixed_quant_flash_attn operator.

    Background: torch_npu in this environment exposes the special quant dtypes
    (float4_e1m2fn_x2, hifloat4_scale, ...) as plain integer enum ids (e.g.
    torch_npu.hifloat4_scale == 299) instead of real ``torch.dtype`` objects.
    Therefore we cannot bit-cast via ``tensor.view(some_dtype)``. The NPU op
    instead relies on ``quant_compute_mode`` plus the host storage layout:

      * quant_compute_mode=1 (mxfp4): kv stored as torch.uint8 (fp4_e2m1 packed),
        descale stored as torch.float8_e8m0fnu (real torch.dtype).
      * quant_compute_mode=2 (hifp4): kv stored as torch.uint8 (fp4_e1m2 packed),
        descale stored as torch.float32 (hifloat4_scale is 4-byte, same as
        float32). The op interprets the float32 bits as hifloat4_scale based
        on quant_compute_mode.

    This helper is currently a no-op but kept as a single point of change in
    case a future torch_npu registers real dtypes for these quant types.
    """
    if tensor is None:
        return None
    # Intentionally no conversion: pass through the host storage dtype.
    return tensor


class MixedFlashAttnGraphNetwork(torch.nn.Module):
    def __init__(self):
        super(MixedFlashAttnGraphNetwork, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        k_descale,
        v_descale,
        block_table,
        cu_seqlens_q,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        quant_compute_mode,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_attn_out,
        return_softmax_lse,
    ):
        attn_out, softmax_lse = torch.ops.cann_ops_transformer.mixed_quant_flash_attn(
            q,
            k,
            v,
            k_descale,
            v_descale,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            quant_compute_mode=quant_compute_mode,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_attn_out=layout_attn_out,
            return_softmax_lse=return_softmax_lse,
        )
        return attn_out, softmax_lse


def run_npu(input_data, device_id=0, mode="eager"):
    q = input_data["q"]
    k = input_data["k"]
    v = input_data["v"]
    k_descale = input_data["k_descale"]
    v_descale = input_data["v_descale"]
    block_table = input_data["block_table"]
    cu_seqlens_q = input_data["cu_seqlens_q"]
    seqused_q = input_data["seqused_q"]
    seqused_kv = input_data["seqused_kv"]
    sinks = input_data["sinks"]
    attn_mask = input_data["attn_mask"]
    quant_compute_mode = input_data["quant_compute_mode"]
    softmax_scale = input_data["softmax_scale"]
    mask_mode = input_data["mask_mode"]
    win_left = input_data["win_left"]
    win_right = input_data["win_right"]
    max_seqlen_q = input_data["max_seqlen_q"]
    max_seqlen_kv = input_data["max_seqlen_kv"]
    layout_q = input_data["layout_q"]
    layout_kv = input_data["layout_kv"]
    layout_attn_out = input_data["layout_attn_out"]
    return_softmax_lse = input_data["return_softmax_lse"]
    batch = input_data["batch_size"]
    n1 = input_data["num_heads_q"]
    n2 = input_data["num_heads_kv"]
    d = input_data["head_dim"]

    # quant_compute_mode = 1
    if RUN_METHOD == "esl":
        meta_op = MixedQuantFlashAttnMetadataOp()
        metadata = meta_op.npu_mixed_quant_flash_attn_metadata(
            num_heads_q=n1,
            num_heads_kv=n2,
            head_dim=d,
            quant_compute_mode=quant_compute_mode,
            batch_size=batch if layout_q != "TND" else None,
            cu_seqlens_q=cu_seqlens_q.cpu().tolist()
            if cu_seqlens_q is not None
            else None,
            seqused_q=seqused_q.cpu().tolist()
            if isinstance(seqused_q, torch.Tensor)
            else seqused_q,
            seqused_kv=seqused_kv.cpu().tolist()
            if isinstance(seqused_kv, torch.Tensor)
            else seqused_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_attn_out,
        )
        metadata = torch.tensor(metadata, dtype=torch.int32).to("npu:%s" % device_id)
    elif RUN_METHOD == "npu":
        metadata = torch.ops.cann_ops_transformer.mixed_quant_flash_attn_metadata(
            num_heads_q=n1,
            num_heads_kv=n2,
            head_dim=d,
            quant_compute_mode=quant_compute_mode,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=batch if layout_q != "TND" else None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_attn_out=layout_attn_out,
        )

    q = q.to("npu:%s" % device_id)
    k = k.to("npu:%s" % device_id)
    v = v.to("npu:%s" % device_id)
    k_descale = k_descale.to("npu:%s" % device_id)
    v_descale = v_descale.to("npu:%s" % device_id)
    if block_table is not None:
        block_table = block_table.to("npu:%s" % device_id)

    if cu_seqlens_q is not None:
        cu_seqlens_q = cu_seqlens_q.to("npu:%s" % device_id)

    if seqused_q is not None:
        if isinstance(seqused_q, list):
            seqused_q = torch.tensor(seqused_q, dtype=torch.int32)
        seqused_q = seqused_q.to("npu:%s" % device_id)

    if seqused_kv is not None:
        if isinstance(seqused_kv, list):
            seqused_kv = torch.tensor(seqused_kv, dtype=torch.int32)
        seqused_kv = seqused_kv.to("npu:%s" % device_id)

    if sinks is not None:
        sinks = sinks.to("npu:%s" % device_id)

    if attn_mask is not None:
        print(f"attn_mask: device {attn_mask.device}")
        attn_mask = attn_mask.to("npu:%s" % device_id)
        print(f"attn_mask: device {attn_mask.device}")
    if mode == "eager":
        attn_out, softmax_lse = torch.ops.cann_ops_transformer.mixed_quant_flash_attn(
            q=q,
            k=k,
            v=v,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            quant_compute_mode=quant_compute_mode,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_attn_out=layout_attn_out,
            return_softmax_lse=return_softmax_lse,
        )
    elif mode == "aclgraph":
        torch._dynamo.reset()
        net = MixedFlashAttnGraphNetwork().npu()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        config.experimental_config.topology_sorting_strategy = "StableRDFS"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        graph_net = torch.compile(
            net, fullgraph=False, backend=npu_backend, dynamic=False
        )
        attn_out, softmax_lse = graph_net(
            q=q,
            k=k,
            v=v,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            quant_compute_mode=quant_compute_mode,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_attn_out=layout_attn_out,
            return_softmax_lse=return_softmax_lse,
        )
    else:
        raise ValueError("mode must be 'eager' or 'aclgraph'")
    torch.npu.synchronize()
    return {
        "attn_out": attn_out.cpu(),
        "softmax_lse": softmax_lse.cpu() if softmax_lse is not None else None,
    }
