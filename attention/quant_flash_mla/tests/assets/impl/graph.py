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
        softmax_scale=None,
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
        # TTK owns H2D, metadata materialization and device selection. Keep the
        # exact arguments: dropping forbidden cu/mask or normalizing empty
        # tensors here would make ACLGraph test a different call from EAGER.
        self.q = q
        self.k_cache = k_cache
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.block_table = block_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_q = seqused_q
        self.attn_mask = attn_mask
        self.metadata = metadata
        self.quant_mode = quant_mode
        self.softmax_scale = softmax_scale
        self.mask_mode = mask_mode
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.head_dim_v = head_dim_v
        self.return_softmax_lse = return_softmax_lse

        logger.info(
            "[GRAPH] __init__ done: q=%s, k_cache=%s, layout_q=%s, layout_kv=%s, "
            "metadata=%s",
            getattr(self.q, "shape", None),
            getattr(self.k_cache, "shape", None),
            self.layout_q,
            self.layout_kv,
            getattr(self.metadata, "shape", None),
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

        # The operator already returns the disabled-LSE placeholder. Do not
        # synthesize or replace an output in the graph wrapper.
        return attn_out, lse_out
