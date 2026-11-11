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
# 第二级 consumer 算子自带的 metadata 入口：输出长度固定 QLI_METADATA_SIZE（与 tiling 的
# METADATA_LIMIT 对齐），不依赖第一级 metadata 算子的设备相关长度。
QSLI_METADATA_OP_NAME = "quant_sparse_lightning_indexer_metadata"


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
            # 第二级算子自带的 metadata 入口（分核信息）
            "quant_sparse_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, "
            "int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None,"
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int? batch_size=None, int? max_seqlen_q=None,"
            "int? max_seqlen_k=None, str? layout_q=None, str? layout_k=None, int? mask_mode=None, "
            "int? cmp_ratio=None, Tensor? candidate_block_length=None, int candidate_block_size=8) -> Tensor",
            # consumer 专属入口: candidate_block_indices 必传 (本算子模式固定为 consumer)
            "quant_sparse_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, "
            "Tensor candidate_block_indices, Tensor? candidate_block_length=None, "
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
        # 其签名是主算子接口（含 return_value、无 candidate_block_indices），且注册名 self.name
        # (2026-09-29 起本算子 torch 入口改名为 quant_sparse_lightning_indexer, 与 self.name 一致)。
        # 本算子的 Meta 覆盖统一由下面的 register_fake 提供，签名与 schema 一致。
        @torch.library.register_fake("cann_ops_transformer::" + QSLI_METADATA_OP_NAME)
        def quant_sparse_lightning_indexer_metadata_meta(
            num_heads_q: int,
            num_heads_k: int,
            head_dim: int,
            topk: int,
            quant_mode: int,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_k: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_k: Optional[torch.Tensor] = None,
            cmp_residual_k: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_k: Optional[int] = None,
            layout_q: Optional[str] = None,
            layout_k: Optional[str] = None,
            mask_mode: Optional[int] = None,
            cmp_ratio: Optional[int] = None,
            candidate_block_length: Optional[torch.Tensor] = None,
            candidate_block_size: int = 8,
        ):
            return torch.empty((QLI_METADATA_SIZE,), dtype=torch.int32, device="meta")

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
            candidate_block_indices,
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
    candidate_block_indices=None,
    candidate_block_length=None,  # 预留接口: 仅接受空 tensor
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
    """consumer 专属入口: candidate_block_indices 必传, 在候选块内选 topk
    (候选块个数由 candidate_block_indices 末维决定, 不再是入参)"""
    op_module = quant_sparse_lightning_indexer_op_builder.load()
    return op_module.quant_sparse_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        candidate_block_indices,
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


def quant_sparse_lightning_indexer_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
    candidate_block_length=None,
    candidate_block_size=8,
):
    """第二级 consumer 算子的分核信息（metadata）生成入口，输出固定 1024 个 int32。

    与第一级 metadata 的差异：长度不随设备核数变化，直接对齐本算子 tiling 的
    METADATA_LIMIT（1024），因此可以被 quant_sparse_lightning_indexer 直接消费，
    不需要调用方再做补齐。
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    op_module = quant_sparse_lightning_indexer_op_builder.load()
    return op_module.quant_sparse_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        candidate_block_length,
        candidate_block_size,
    )


@torch.library.register_kernel("cann_ops_transformer::" + QSLI_METADATA_OP_NAME, None)
def quant_sparse_lightning_indexer_metadata_fallback(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_k: Optional[str] = None,
    mask_mode: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
    candidate_block_length: Optional[torch.Tensor] = None,
    candidate_block_size: int = 8,
):
    return quant_sparse_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        candidate_block_length=candidate_block_length,
        candidate_block_size=candidate_block_size,
    )


try:
    impl(get_as_library(), QSLI_METADATA_OP_NAME, "PrivateUse1")(
        quant_sparse_lightning_indexer_metadata
    )
except RuntimeError:
    pass  # 重复 import (循环导入回滚重试) 时 schema 已注册
torch.compiler.allow_in_graph(quant_sparse_lightning_indexer_metadata)


try:
    impl(get_as_library(), "quant_sparse_lightning_indexer", "PrivateUse1")(
        quant_sparse_lightning_indexer
    )
except RuntimeError:
    pass  # 重复 import (循环导入回滚重试) 时 schema 已注册
torch.compiler.allow_in_graph(quant_sparse_lightning_indexer)
