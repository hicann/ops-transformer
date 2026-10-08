# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
import os
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

QLI_METADATA_SIZE = 1024
QLI_METADATA_OP_NAME = "quant_lightning_indexer_metadata"


class QuantSparseLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantSparseLightningIndexerOpBuilder, self).__init__(
            "quant_sparse_lightning_indexer", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_sparse_lightning_indexer.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            # P13 拆分: consumer 专属入口, candidate_topk_index 必传 (本算子模式固定为 consumer)
            "quant_sparse_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, "
            "Tensor candidate_topk_index, Tensor? candidate_block_length=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
            "Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, "
            'int max_seqlen_q=-1, str layout_q="BSND", str layout_k="BSND", int mask_mode=0, '
            "int cmp_ratio=1, "
            "int candidate_block_size=8) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        # 注: 此处原有一份从第一级算子拷贝的 impl(..., self.name, "Meta")，
        # 其签名是主算子接口（含 return_value、无 candidate_topk_index），且注册名 self.name
        # (2026-09-29 起本算子 torch 入口改名为 quant_sparse_lightning_indexer, 与 self.name 一致)。
        # 本算子的 Meta 覆盖统一由下面的 register_fake 提供，签名与 schema 一致。
        @torch.library.register_fake(
            "cann_ops_transformer::quant_sparse_lightning_indexer"
        )
        def quant_sparse_lightning_indexer_meta(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            topk,
            quant_mode,
            *,
            candidate_topk_index,
            candidate_block_length=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
            candidate_block_size=8,
        ):
            key_head_num = key.shape[1] if layout_k == "TND" else key.shape[2]
            if layout_q == "BSND":
                sparse_indices_out = torch.empty(
                    [query.shape[0], query.shape[1], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                sparse_indices_out = torch.empty(
                    [query.shape[0], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            sparse_values_out = torch.empty([0], dtype=torch.bfloat16, device="meta")
            return (sparse_indices_out, sparse_values_out)


# Instantiate the builder (schema def 幂等: 直接判 torch.ops, 避免 ops/__init__
# 自动发现链回滚重试导致的重复 define)
quant_sparse_lightning_indexer_op_builder = QuantSparseLightningIndexerOpBuilder()
try:
    quant_sparse_lightning_indexer_op_builder._ensure_initialized()
except RuntimeError:
    # define 报"已注册": schema 已在 torch Library 中 (此前导入尝试回滚后残留)。
    # _ensure_initialized 在 register_schema 处中断, 其余初始化 (路径属性) 需补齐,
    # 否则 load() 访问 _package_path 抛 AttributeError。
    _b = quant_sparse_lightning_indexer_op_builder
    _b._initialized = True
    import torch_npu as _torch_npu

    _b._torch_npu_path = os.path.dirname(os.path.abspath(_torch_npu.__file__))
    _b._package_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def quant_sparse_lightning_indexer(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_topk_index=None,
    candidate_block_length=None,  # 预留接口: 仅接受空 tensor (N4/O13, 本期不消费)
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    candidate_block_size=8,
):
    """consumer 专属入口: candidate_topk_index 必传, 在候选块内选 topk
    (候选块个数由 candidate_topk_index 末维决定, 不再是入参)"""
    op_module = quant_sparse_lightning_indexer_op_builder.load()
    return op_module.quant_sparse_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        candidate_topk_index,
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        candidate_block_size,
    )


try:
    impl(get_as_library(), "quant_sparse_lightning_indexer", "PrivateUse1")(
        quant_sparse_lightning_indexer
    )
except RuntimeError:
    pass  # 重复 import (循环导入回滚重试) 时 schema 已注册
torch.compiler.allow_in_graph(quant_sparse_lightning_indexer)
