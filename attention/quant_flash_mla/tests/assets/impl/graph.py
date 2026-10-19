#!/usr/bin/python3
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
"""ACLGraph target module for quant_flash_mla.

Reuse the tensors prepared by TTK/customize_inputs. Metadata is materialized
once by npu_preprocess before capture, and forward contains only the main op.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class QuantMlaAclGraph(torch.nn.Module):
    """ACLGraph module that reuses the exact TTK-prepared operator inputs.

    npu_preprocess creates metadata before capture. The module keeps the
    prepared tensors as graph constants and forward invokes only quant_flash_mla.
    """

    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        quant_mode: int,
        *,
        cu_seqlens_q=None,
        seqused_q=None,
        attn_mask=None,
        metadata=None,
        softmax_scale=1.0,
        mask_mode=0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_v=512,
        layout_q="TND",
        layout_kv="PA_BNBD",
        layout_out="TND",
        return_softmax_lse=False,
        **kwargs,
    ):
        super().__init__()
        import torch_npu

        torch_npu.npu.set_device(int(kwargs.get("device_id", 0)))

        # TTK has already applied customize_inputs and prepared device arguments.
        # Reuse those exact tensors so eager, ACLGraph and golden all operate on
        # the same data. Regenerating Q/K/cache here loses CSV-only generation
        # attributes (for example q_data_range/k_data_range and seeds).
        def _required_npu(tensor, name):
            if tensor is None or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"QuantFlashMla graph requires tensor input {name}")
            return tensor.npu()

        def _optional_npu(tensor):
            if (
                tensor is None
                or not isinstance(tensor, torch.Tensor)
                or tensor.numel() == 0
            ):
                return None
            return tensor.npu()

        self.q = _required_npu(q, "q")
        self.k_cache = _required_npu(k_cache, "k_cache")
        self.q_descale = _required_npu(q_descale, "q_descale")
        self.k_descale = _required_npu(k_descale, "k_descale")
        self.block_table = _required_npu(block_table, "block_table")
        self.cache_seqlens = _required_npu(cache_seqlens, "cache_seqlens")
        self.cu_seqlens_q = _optional_npu(cu_seqlens_q) if layout_q == "TND" else None
        self.seqused_q = _optional_npu(seqused_q)
        self.attn_mask = _optional_npu(attn_mask) if mask_mode == 3 else None

        torch.npu.synchronize()

        # ---- 5. metadata: 只消费 npu_preprocess 的结果 (对齐 QFA graph.py 风格) ----
        # 动态槽位 (CSV tensor_view_shapes 含 -1): ttk 在 invoke/apply npu_preprocess
        # 后把 {"metadata": generated} 回填到主算子参数, 再按 split_params 传进本模块
        # __init__; 静态槽位: hook 已原地回填占位 tensor。metadata 生成位于图捕获与
        # 主算子计时之外, 此处只消费, 不重复生成。
        if (
            metadata is None
            or not isinstance(metadata, torch.Tensor)
            or metadata.numel() == 0
        ):
            raise ValueError(
                "QuantFlashMla graph requires metadata prepared by npu_preprocess"
            )
        # metadata 可能建在错误的 device 上, 对齐 q.device
        if metadata.device != self.q.device:
            metadata = metadata.to(self.q.device)
        self.metadata = metadata

        # ---- 6. 存 forward 需要的属性 ----
        self.quant_mode = int(quant_mode)
        self.softmax_scale = softmax_scale
        self.mask_mode = int(mask_mode)
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_kv = int(max_seqlen_kv)
        self.head_dim_v = int(head_dim_v)
        self.return_softmax_lse = bool(return_softmax_lse)

        logger.info(
            "[GRAPH] __init__ done: q=%s, k_cache=%s, layout_q=%s, layout_kv=%s, "
            "metadata=%s",
            self.q.shape,
            self.k_cache.shape,
            self.layout_q,
            self.layout_kv,
            self.metadata.shape,
        )

    def forward(self):
        """只调 quant_flash_mla 主算子, 可被 npugraph_ex 捕获。

        所有入参来自 self. 属性 (__init__ 预处理结果), forward 无参数。
        """
        attn_out, lse_out = torch.ops.cann_ops_transformer.quant_flash_mla(
            self.q,
            self.k_cache,
            self.q_descale,
            self.k_descale,
            self.block_table,
            self.cache_seqlens,
            self.quant_mode,
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_q=self.seqused_q,
            attn_mask=self.attn_mask,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.mask_mode,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            head_dim_v=self.head_dim_v,
            layout_q=self.layout_q,
            layout_kv=self.layout_kv,
            layout_out=self.layout_out,
            return_softmax_lse=self.return_softmax_lse,
        )

        if not self.return_softmax_lse:
            lse_out = None
        return attn_out, lse_out
