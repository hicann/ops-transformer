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


def _is_ascend_950(device):
    device = torch.device(device)
    return device.type == "npu" and torch.npu.get_device_name(device.index).startswith(
        "Ascend950"
    )


def _check_tensor_devices(device, **tensors):
    for name, tensor in tensors.items():
        if tensor is not None and tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _get_metadata_device(*tensors):
    device = next((tensor.device for tensor in tensors if tensor is not None), None)
    if device is None:
        device = torch.device("npu", torch.npu.current_device())
    if any(tensor.device != device for tensor in tensors if tensor is not None):
        raise ValueError("Metadata input tensors must be on the same device")
    return device


def _as_mxfp4_storage(tensor):
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype != getattr(torch, "float4_e2m1fn_x2", None):
        raise TypeError("MXFP4 data must use packed uint8 or float4_e2m1fn_x2")
    return tensor.view(torch.uint8)


def _as_e8m0_storage(tensor, allow_bytes=False):
    if tensor is None:
        raise TypeError("MXFP4 scales must be provided")
    if allow_bytes and tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype != getattr(torch, "float8_e8m0fnu", None):
        raise TypeError("MXFP4 scales must use float8_e8m0fnu")
    return tensor.view(torch.uint8)


def _validate_950_mode(quant_mode):
    if quant_mode != 5:
        raise ValueError(
            "When using quant_sparse_lightning_indexer on Ascend 950, "
            f"quant_mode must be 5, but got {quant_mode}."
        )


def _validate_950_layout(layout_q, layout_k, return_value=0):
    if (layout_q, layout_k) not in (
        ("TND", "TND"),
        ("TND", "PA_BBND"),
        ("BSND", "BSND"),
        ("BSND", "PA_BBND"),
    ):
        raise ValueError("MXFP4 requires matching TND/BSND layouts or a PA_BBND key")
    if return_value not in (0, 1):
        raise ValueError("return_value must be 0 or 1")


def _quant_sparse_lightning_indexer_950(
    q,
    k,
    w,
    descale_q,
    descale_k,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    candidate_block_size,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=0,
):
    """Run QSLI with the public candidate and quantization contract."""
    if candidate_block_indices is None or candidate_block_length is None:
        raise ValueError(
            "Ascend 950 requires candidate_block_indices and candidate_block_length"
        )
    _validate_950_layout(layout_q, layout_k, return_value)
    _validate_950_mode(quant_mode)
    if candidate_block_size != 8:
        raise ValueError("candidate_block_size must be 8")
    _check_tensor_devices(
        q.device,
        k=k,
        w=w,
        descale_q=descale_q,
        descale_k=descale_k,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
    )
    if layout_k == "PA_BBND":
        if k.dtype != torch.uint8:
            raise TypeError("PA_BBND key must use packed uint8 storage")
        if (
            descale_k is None
            or descale_k.dtype != torch.uint8
            or descale_k.numel() != 0
        ):
            raise ValueError("PA_BBND descale_k must be an empty uint8 tensor")
        key_scale = None
    else:
        key_scale = _as_e8m0_storage(descale_k, allow_bytes=True)
    with torch.npu.device(q.device):
        from .op_dsl.quant_sparse_lightning_indexer import (
            quant_sparse_lightning_indexer as compute_950,
        )

        return compute_950(
            _as_mxfp4_storage(q),
            _as_mxfp4_storage(k),
            w,
            _as_e8m0_storage(descale_q, allow_bytes=True),
            candidate_block_indices,
            candidate_block_length,
            descale_k=key_scale,
            topk=topk,
            candidate_block_size=candidate_block_size,
            quant_mode=5,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            block_table=block_table,
            output_idx_offset=output_idx_offset,
            metadata=metadata,
            max_seqlen_q=max_seqlen_q,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
        )


def _quant_sparse_lightning_indexer_metadata_950(
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
    candidate_block_length,
    candidate_block_size,
):
    if candidate_block_length is None:
        raise ValueError("Ascend 950 requires candidate_block_length")
    _validate_950_mode(quant_mode)
    if candidate_block_size != 8:
        raise ValueError("candidate_block_size must be 8")
    device = _get_metadata_device(
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
    )
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    _validate_950_layout(layout_q, layout_k)
    with torch.npu.device(device):
        from .op_dsl.quant_sparse_lightning_indexer_metadata import (
            quant_sparse_lightning_indexer_metadata as metadata_950,
        )

        return metadata_950(
            candidate_block_length,
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            head_dim=head_dim,
            topk=topk,
            quant_mode=5,
            candidate_block_size=candidate_block_size,
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
        )


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
            "int candidate_block_size=8, int return_value=0) -> (Tensor, Tensor)",
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
            return torch.empty(
                (QLI_METADATA_SIZE,),
                dtype=torch.int32,
                device=_get_metadata_device(
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqused_q,
                    seqused_k,
                    cmp_residual_k,
                    candidate_block_length,
                ),
            )

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
            return_value=0,
        ):
            key_head_num = (
                1
                if layout_k == "PA_BBND" and key.dim() == 3
                else (key.shape[1] if layout_k == "TND" else key.shape[2])
            )
            if layout_q == "BSND":
                sparse_indices_out = torch.empty(
                    [query.shape[0], query.shape[1], key_head_num, topk],
                    dtype=torch.int32,
                    device=query.device,
                )
            else:
                sparse_indices_out = torch.empty(
                    [query.shape[0], key_head_num, topk],
                    dtype=torch.int32,
                    device=query.device,
                )
            sparse_values_out = torch.empty(
                sparse_indices_out.shape if return_value else (0,),
                dtype=torch.bfloat16,
                device=query.device,
            )
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


def _quant_sparse_lightning_indexer_npu(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_block_indices=None,
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
    return_value=0,
):
    """consumer 专属入口: candidate_block_indices 必传, 在候选块内选 topk
    (候选块个数由 candidate_block_indices 末维决定, 不再是入参)"""
    if query.device.type == "cpu":
        raise ValueError("Query must be an NPU tensor")
    if _is_ascend_950(query.device):
        return _quant_sparse_lightning_indexer_950(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            candidate_block_indices,
            candidate_block_length,
            topk,
            quant_mode,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            block_table=block_table,
            output_idx_offset=output_idx_offset,
            metadata=metadata,
            candidate_block_size=candidate_block_size,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            layout_k=layout_k,
            return_value=return_value,
        )
    if return_value != 0:
        raise ValueError("return_value=1 requires Ascend 950")
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


def _quant_sparse_lightning_indexer_metadata_impl(
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
    device = None
    for tensor in (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        candidate_block_length,
    ):
        if tensor is not None:
            device = tensor.device
            break
    if device is None:
        device = torch.device("npu", torch.npu.current_device())
    if device.type == "cpu":
        raise ValueError("Metadata sequence inputs must be on NPU")
    if _is_ascend_950(device):
        return _quant_sparse_lightning_indexer_metadata_950(
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
    return_value=0,
):
    """consumer 专属入口: candidate_block_indices 必传, 在候选块内选 topk
    (候选块个数由 candidate_block_indices 末维决定, 不再是入参)"""
    return torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        candidate_block_size=candidate_block_size,
        return_value=return_value,
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
    return torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer_metadata(
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
    return _quant_sparse_lightning_indexer_metadata_impl(
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
        _quant_sparse_lightning_indexer_metadata_impl
    )
except RuntimeError:
    pass  # 重复 import (循环导入回滚重试) 时 schema 已注册
torch.compiler.allow_in_graph(quant_sparse_lightning_indexer_metadata)


try:
    impl(get_as_library(), "quant_sparse_lightning_indexer", "PrivateUse1")(
        _quant_sparse_lightning_indexer_npu
    )
except RuntimeError:
    pass  # 重复 import (循环导入回滚重试) 时 schema 已注册
torch.compiler.allow_in_graph(quant_sparse_lightning_indexer)
