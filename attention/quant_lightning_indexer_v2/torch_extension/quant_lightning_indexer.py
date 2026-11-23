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
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


QLI_METADATA_EXTENSION_SIZE = 256
# V3 metadata（candidate 与 DCP 共用）固定输出 1024 个 int32
QLI_V3_METADATA_SIZE = 1024
KEY_HEAD_DIM_TND = 1
KEY_HEAD_DIM_BSND = 2


def _calculate_metadata_size(device=None):
    properties = torch.npu.get_device_properties(device)
    return (
        8 * (1 + properties.cube_core_num + properties.vector_core_num)
        + QLI_METADATA_EXTENSION_SIZE
    )


QLI_METADATA_OP_NAME = "quant_lightning_indexer_metadata"
QLI_CANDIDATE_OP_NAME = "quant_lightning_indexer.candidate"
QLI_METADATA_CANDIDATE_OP_NAME = "quant_lightning_indexer_metadata.candidate"
QLI_DCP_OP_NAME = "quant_lightning_indexer.dcp"
QLI_METADATA_DCP_OP_NAME = "quant_lightning_indexer_metadata.dcp"


def _validate_candidate_parameters(candidate_topk_blocks, candidate_block_size):
    disabled = candidate_topk_blocks == -1 and candidate_block_size == -1
    enabled = candidate_topk_blocks == 2048 and candidate_block_size == 8
    if not (disabled or enabled):
        raise ValueError(
            "(candidate_topk_blocks, candidate_block_size) must be "
            "(-1, -1) or (2048, 8)"
        )
    return enabled


def _is_dsl_candidate_device(device):
    device = torch.device(device)
    if device.type != "npu":
        raise ValueError("Lightning indexer candidate inputs must be on an NPU")
    name = torch.npu.get_device_name(device.index)
    return "Ascend950" in name


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


def _validate_dsl_mode(quant_mode):
    if quant_mode != 5:
        raise ValueError(
            "When the two-level TopK candidate source mode is enabled on Ascend 950, "
            f"quant_mode must be 5, but got {quant_mode}."
        )


def _validate_dsl_layout(layout_q, layout_k, return_value=0):
    if (layout_q, layout_k) not in (
        ("TND", "TND"),
        ("TND", "PA_BBND"),
        ("BSND", "BSND"),
        ("BSND", "PA_BBND"),
    ):
        raise ValueError("MXFP4 requires matching TND/BSND layouts or a PA_BBND key")
    if return_value not in (0, 1):
        raise ValueError("return_value must be 0 or 1")


class QuantLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerOpBuilder, self).__init__(
            "quant_lightning_indexer", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_lightning_indexer.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, "
            "int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None,"
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int? batch_size=None, int? max_seqlen_q=None,"
            "int? max_seqlen_k=None, str? layout_q=None, str? layout_k=None, int? mask_mode=None, "
            "int? cmp_ratio=None) -> Tensor",
            "quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
            "Tensor key_dequant_scale, int topk, int quant_mode, *, Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? "
            "cmp_residual_k = None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, "
            'int max_seqlen_q=-1, str layout_q="BSND", str layout_k="BSND", int mask_mode=0, '
            "int cmp_ratio=1, int return_value=0) -> (Tensor, Tensor)",
            "quant_lightning_indexer.candidate(Tensor query, Tensor key, Tensor weights, Tensor "
            "query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, Tensor? "
            "cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
            "Tensor? cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? "
            "metadata=None, int max_seqlen_q=-1, str layout_q='BSND', str layout_k='BSND', int mask_mode=0, "
            "int cmp_ratio=1, int return_value=0, int candidate_topk_blocks, int candidate_block_size) -> "
            "(Tensor, Tensor, Tensor, Tensor)",
            "quant_lightning_indexer_metadata.candidate(int num_heads_q, int num_heads_k, int head_dim, int "
            "topk, int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? "
            "seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int? batch_size=None, int? "
            "max_seqlen_q=None, int? max_seqlen_k=None, str? layout_q=None, str? layout_k=None, int? "
            "mask_mode=None, int? cmp_ratio=None, int candidate_topk_blocks, int candidate_block_size) -> "
            "Tensor",
            # DCP（Decode Context Parallel）：两输出，cp_world>1 启用；与 candidate 互斥
            "quant_lightning_indexer.dcp(Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? block_table=None, "
            "Tensor? output_idx_offset=None, Tensor? metadata=None, Tensor? causal_seqlens_kv_global=None, "
            "int max_seqlen_q=-1, str layout_q='TND', "
            "str layout_k='PA_BBND', int mask_mode=3, int cmp_ratio=1, "
            "int return_value=1, int? cp_world=None, int? cp_rank=None, "
            "int? cp_kv_cache_interleave_size=None) -> (Tensor, Tensor)",
            "quant_lightning_indexer_metadata.dcp(int num_heads_q, int num_heads_k, int head_dim, int topk, "
            "int quant_mode, *, Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
            "Tensor? cmp_residual_k=None, Tensor? causal_seqlens_kv_global=None, "
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_k=None, "
            "str? layout_q='TND', str? layout_k='PA_BBND', int? mask_mode=3, int? cmp_ratio=1, int? cp_world=None, "
            "int? cp_rank=None, int? cp_kv_cache_interleave_size=None) -> Tensor",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + QLI_METADATA_OP_NAME)
        def quant_lightning_indexer_metadata_meta(
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
        ):
            return torch.empty(
                (_calculate_metadata_size()), dtype=torch.int32, device="npu"
            )

        @impl(get_as_library(), self.name, "Meta")
        def quant_lightning_indexer_meta(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
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
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
            return_value=0,
        ):
            key_head_num = (
                key.shape[KEY_HEAD_DIM_TND]
                if layout_k == "TND"
                else key.shape[KEY_HEAD_DIM_BSND]
            )

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
            if return_value:
                if layout_q == "BSND":
                    sparse_values_out = torch.empty(
                        [query.shape[0], query.shape[1], key_head_num, topk],
                        dtype=torch.bfloat16,
                        device="meta",
                    )
                else:
                    sparse_values_out = torch.empty(
                        [query.shape[0], key_head_num, topk],
                        dtype=torch.bfloat16,
                        device="meta",
                    )
            else:
                sparse_values_out = torch.empty(
                    [0], dtype=torch.bfloat16, device="meta"
                )
            return (sparse_indices_out, sparse_values_out)


# Instantiate the builder
quant_lightning_indexer_op_builder = QuantLightningIndexerOpBuilder()
quant_lightning_indexer_op_builder._ensure_initialized()


@torch.library.register_fake("cann_ops_transformer::" + QLI_METADATA_DCP_OP_NAME)
def quant_lightning_indexer_metadata_dcp_meta(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    causal_seqlens_kv_global=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
    cmp_ratio=1,
    cp_world: Optional[int] = None,
    cp_rank: Optional[int] = None,
    cp_kv_cache_interleave_size: Optional[int] = None,
):
    # 上游 V3Metadata 按 1024 元素校验输出，DCP 复用 V3 metadata 接口
    return torch.empty((QLI_V3_METADATA_SIZE), dtype=torch.int32, device="npu")


@torch.library.register_fake("cann_ops_transformer::" + QLI_DCP_OP_NAME)
def quant_lightning_indexer_dcp_meta(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
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
    causal_seqlens_kv_global=None,
    max_seqlen_q=-1,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
    cmp_ratio=1,
    return_value=1,
    cp_world: Optional[int] = None,
    cp_rank: Optional[int] = None,
    cp_kv_cache_interleave_size: Optional[int] = None,
):
    # DCP 仅支持 TND + PA_BBND，score 输出 BF16（与 arch35 对齐）
    key_head_num = (
        key.shape[KEY_HEAD_DIM_TND]
        if layout_k == "TND"
        else key.shape[KEY_HEAD_DIM_BSND]
    )
    out_shape = (query.shape[0], key_head_num, topk)
    indices = query.new_empty(out_shape, dtype=torch.int32)
    values = (
        query.new_empty(out_shape, dtype=torch.bfloat16)
        if return_value
        else query.new_empty((0,), dtype=torch.bfloat16)
    )
    return indices, values


@impl(get_as_library(), QLI_METADATA_OP_NAME, "PrivateUse1")
def _quant_lightning_indexer_metadata_npu(
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
):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    op_module = quant_lightning_indexer_op_builder.load()
    tensors = (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k)
    device = next((t.device for t in tensors if t is not None), None)
    return op_module.quant_lightning_indexer_metadata(
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
        _calculate_metadata_size(device),
    )


@torch.library.register_kernel("cann_ops_transformer::" + QLI_METADATA_OP_NAME, None)
def quant_lightning_indexer_metadata_fallback(
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
):
    # 处理所有 tensor 都为 None 的情况
    # 调用 NPU 实现
    return _quant_lightning_indexer_metadata_npu(
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
    )


@impl(get_as_library(), quant_lightning_indexer_op_builder.name, "PrivateUse1")
def _quant_lightning_indexer_npu(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
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
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = quant_lightning_indexer_op_builder.load()
    return op_module.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
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
        return_value,
    )


@torch.library.register_fake("cann_ops_transformer::" + QLI_CANDIDATE_OP_NAME)
def _quant_lightning_indexer_candidate_meta(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
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
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_topk_blocks,
    candidate_block_size,
):
    enabled = _validate_candidate_parameters(
        candidate_topk_blocks, candidate_block_size
    )
    if layout_q not in ("BSND", "TND"):
        raise ValueError("layout_q must be BSND or TND")
    key_head_num = (
        key.shape[KEY_HEAD_DIM_TND]
        if layout_k == "TND"
        else key.shape[KEY_HEAD_DIM_BSND]
    )
    rows = (
        (query.shape[0], query.shape[1], key_head_num)
        if layout_q == "BSND"
        else (query.shape[0], key_head_num)
    )
    indices = query.new_empty((*rows, topk), dtype=torch.int32)
    values = query.new_empty(
        (*rows, topk) if return_value else (0,), dtype=torch.bfloat16
    )
    candidates = query.new_empty(
        (*rows, candidate_topk_blocks) if enabled else (0,), dtype=torch.int32
    )
    lengths = query.new_empty(rows if enabled else (0,), dtype=torch.int32)
    return indices, values, candidates, lengths


@impl(get_as_library(), QLI_CANDIDATE_OP_NAME, "PrivateUse1")
def _quant_lightning_indexer_candidate_npu(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
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
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_topk_blocks,
    candidate_block_size,
):
    enabled = _validate_candidate_parameters(
        candidate_topk_blocks, candidate_block_size
    )
    if not enabled:
        indices, values = _quant_lightning_indexer_npu(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
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
            max_seqlen_q=max_seqlen_q,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
        )
        return (
            indices,
            values,
            query.new_empty((0,), dtype=torch.int32),
            query.new_empty((0,), dtype=torch.int32),
        )
    if not _is_dsl_candidate_device(query.device):
        with torch.npu.device(query.device):
            op_module = quant_lightning_indexer_op_builder.load()
            return op_module.quant_lightning_indexer_candidate(
                query,
                key,
                weights,
                query_dequant_scale,
                key_dequant_scale,
                topk,
                quant_mode,
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
                return_value,
                candidate_topk_blocks,
                candidate_block_size,
            )
    _validate_dsl_mode(quant_mode)
    _validate_dsl_layout(layout_q, layout_k, return_value)
    _check_tensor_devices(
        query.device,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
    )
    with torch.npu.device(query.device):
        from .op_dsl.quant_lightning_indexer_v2 import (
            quant_lightning_indexer as dsl_compute,
        )

        return dsl_compute(
            _as_mxfp4_storage(query),
            _as_mxfp4_storage(key),
            weights,
            _as_e8m0_storage(query_dequant_scale),
            _as_e8m0_storage(key_dequant_scale),
            topk=topk,
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
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )


@torch.library.register_fake("cann_ops_transformer::" + QLI_METADATA_CANDIDATE_OP_NAME)
def _quant_lightning_indexer_metadata_candidate_meta(
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
    candidate_topk_blocks,
    candidate_block_size,
):
    enabled = _validate_candidate_parameters(
        candidate_topk_blocks, candidate_block_size
    )
    device = _get_metadata_device(
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k
    )
    size = QLI_V3_METADATA_SIZE if enabled else _calculate_metadata_size(device)
    return torch.empty((size,), dtype=torch.int32, device=device)


@torch.library.register_kernel(
    "cann_ops_transformer::" + QLI_METADATA_CANDIDATE_OP_NAME, None
)
@impl(get_as_library(), QLI_METADATA_CANDIDATE_OP_NAME, "PrivateUse1")
def _quant_lightning_indexer_metadata_candidate_npu(
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
    candidate_topk_blocks,
    candidate_block_size,
):
    enabled = _validate_candidate_parameters(
        candidate_topk_blocks, candidate_block_size
    )
    if not enabled:
        return _quant_lightning_indexer_metadata_npu(
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
        )
    device = _get_metadata_device(
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k
    )
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    if not _is_dsl_candidate_device(device):
        with torch.npu.device(device):
            op_module = quant_lightning_indexer_op_builder.load()
            return op_module.quant_lightning_indexer_metadata_candidate(
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
                0 if batch_size is None else batch_size,
                max_seqlen_q,
                max_seqlen_k,
                layout_q,
                layout_k,
                mask_mode,
                cmp_ratio,
                candidate_topk_blocks,
                candidate_block_size,
            )
    _validate_dsl_mode(quant_mode)
    _validate_dsl_layout(layout_q, layout_k)
    with torch.npu.device(device):
        from .op_dsl.quant_lightning_indexer_metadata import (
            quant_lightning_indexer_metadata as dsl_metadata,
        )

        return dsl_metadata(
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            head_dim=head_dim,
            topk=topk,
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
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )


@impl(get_as_library(), QLI_METADATA_DCP_OP_NAME, "PrivateUse1")
def _quant_lightning_indexer_metadata_dcp_npu(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    quant_mode: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_k: Optional[str] = None,
    mask_mode: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
    cp_world: Optional[int] = None,
    cp_rank: Optional[int] = None,
    cp_kv_cache_interleave_size: Optional[int] = None,
):
    if causal_seqlens_kv_global is None:
        raise ValueError("DCP requires causal_seqlens_kv_global")
    if cp_world is None or cp_world <= 1:
        raise ValueError(f"DCP requires cp_world > 1, got {cp_world}")
    if cp_rank is None:
        raise ValueError("DCP requires cp_rank")
    if cp_kv_cache_interleave_size is None:
        raise ValueError("DCP requires cp_kv_cache_interleave_size")
    if cp_rank < 0 or cp_rank >= cp_world:
        raise ValueError(
            f"DCP requires 0 <= cp_rank < cp_world, got rank={cp_rank}, world={cp_world}"
        )
    if cp_kv_cache_interleave_size < 1:
        raise ValueError("DCP requires cp_kv_cache_interleave_size >= 1")
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "TND" if layout_q is None else layout_q
    layout_k = "PA_BBND" if layout_k is None else layout_k
    mask_mode = 3 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    op_module = quant_lightning_indexer_op_builder.load()
    return op_module.quant_lightning_indexer_metadata_dcp(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        causal_seqlens_kv_global,
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
        QLI_V3_METADATA_SIZE,
        cp_world,
        cp_rank,
        cp_kv_cache_interleave_size,
    )


def _resolve_dcp_common_attrs(layout_q, layout_k, mask_mode, cmp_ratio):
    """DCP 分支的参数默认值与显式校验。

    cp_world>1 时，未显式传入的参数使用 DCP 语义默认值：
    layout_q=TND、layout_k=PA_BBND、mask_mode=3、cmp_ratio=1；
    显式传入非法值直接报错，不静默改写。
    """
    layout_q = "TND" if layout_q is None else layout_q
    layout_k = "PA_BBND" if layout_k is None else layout_k
    mask_mode = 3 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    if layout_q != "TND":
        raise ValueError(f"DCP requires layout_q=TND, got {layout_q!r}")
    if layout_k != "PA_BBND":
        raise ValueError(f"DCP requires layout_k=PA_BBND, got {layout_k!r}")
    if mask_mode != 3:
        raise ValueError(f"DCP requires mask_mode=3, got {mask_mode}")
    if cmp_ratio != 1:
        raise ValueError(f"DCP requires cmp_ratio=1, got {cmp_ratio}")
    return layout_q, layout_k, mask_mode, cmp_ratio


def _resolve_dcp_return_value(return_value):
    """DCP 的 return_value 默认 1；显式传 0 等非 1 值直接报错。"""
    value = 1 if return_value is None else return_value
    if value != 1:
        raise ValueError(f"DCP requires return_value=1, got {value}")
    return value


def _reject_dcp_params_without_cp_world(
    causal_seqlens_kv_global, cp_rank, cp_kv_cache_interleave_size
):
    """cp_world 未启用时，DCP 专属参数只能保持默认值。"""
    message = (
        "DCP parameters require cp_world > 1; when cp_world is default, "
        "causal_seqlens_kv_global must be None, cp_rank must be 0 and "
        "cp_kv_cache_interleave_size must be 1"
    )
    if causal_seqlens_kv_global is not None:
        raise ValueError(message)
    if cp_rank is not None and cp_rank != 0:
        raise ValueError(message)
    if cp_kv_cache_interleave_size is not None and cp_kv_cache_interleave_size != 1:
        raise ValueError(message)


def _require_dcp_rank_and_interleave(cp_rank, cp_kv_cache_interleave_size):
    """DCP 启用时，rank 和 KV cache 交错粒度必须显式传入。"""
    if cp_rank is None:
        raise ValueError("DCP requires cp_rank")
    if cp_kv_cache_interleave_size is None:
        raise ValueError("DCP requires cp_kv_cache_interleave_size")


@impl(get_as_library(), QLI_DCP_OP_NAME, "PrivateUse1")
def _quant_lightning_indexer_dcp_npu(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
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
    causal_seqlens_kv_global=None,
    max_seqlen_q=-1,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
    cmp_ratio=1,
    return_value=1,
    cp_world: Optional[int] = None,
    cp_rank: Optional[int] = None,
    cp_kv_cache_interleave_size: Optional[int] = None,
):
    if causal_seqlens_kv_global is None:
        raise ValueError("DCP requires causal_seqlens_kv_global")
    if cp_world is None or cp_world <= 1:
        raise ValueError(f"DCP requires cp_world > 1, got {cp_world}")
    if cp_rank is None:
        raise ValueError("DCP requires cp_rank")
    if cp_kv_cache_interleave_size is None:
        raise ValueError("DCP requires cp_kv_cache_interleave_size")
    if cp_rank < 0 or cp_rank >= cp_world:
        raise ValueError(
            f"DCP requires 0 <= cp_rank < cp_world, got rank={cp_rank}, world={cp_world}"
        )
    if cp_kv_cache_interleave_size < 1:
        raise ValueError("DCP requires cp_kv_cache_interleave_size >= 1")
    if output_idx_offset is not None:
        raise ValueError("DCP requires output_idx_offset to be None")
    op_module = quant_lightning_indexer_op_builder.load()
    return op_module.quant_lightning_indexer_dcp(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        causal_seqlens_kv_global,
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
        return_value,
        cp_world,
        cp_rank,
        cp_kv_cache_interleave_size,
    )


_CANDIDATE_UNSET = object()


def quant_lightning_indexer(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
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
    # 保持既有默认值；DCP 场景需显式传 layout_q="TND"、layout_k="PA_BBND"、
    # mask_mode=3、return_value=1，避免按 V2 默认值静默路由。
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_topk_blocks=_CANDIDATE_UNSET,
    candidate_block_size=_CANDIDATE_UNSET,
    causal_seqlens_kv_global=None,
    cp_world: Optional[int] = None,
    cp_rank: Optional[int] = None,
    cp_kv_cache_interleave_size: Optional[int] = None,
):
    """按 cp_world / candidate 参数路由到 DCP / V2 / V3 接口。"""
    op = torch.ops.cann_ops_transformer.quant_lightning_indexer
    if (cp_world is not None) and (cp_world < 1):
        raise ValueError(f"cp_world must be >= 1, got {cp_world}")
    if (cp_world is not None) and (cp_world > 1):
        if (
            candidate_topk_blocks is not _CANDIDATE_UNSET
            or candidate_block_size is not _CANDIDATE_UNSET
        ):
            raise ValueError("DCP is mutually exclusive with candidate")
        if causal_seqlens_kv_global is None:
            raise ValueError("DCP requires causal_seqlens_kv_global")
        if output_idx_offset is not None:
            raise ValueError("DCP requires output_idx_offset to be None")
        if int(quant_mode) != 2:
            raise ValueError(f"DCP requires quant_mode=2 on arch22, got {quant_mode}")
        layout_q, layout_k, mask_mode, cmp_ratio = _resolve_dcp_common_attrs(
            layout_q, layout_k, mask_mode, cmp_ratio
        )
        return_value = _resolve_dcp_return_value(return_value)
        max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
        _require_dcp_rank_and_interleave(cp_rank, cp_kv_cache_interleave_size)
        return op.dcp(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
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
            causal_seqlens_kv_global=causal_seqlens_kv_global,
            max_seqlen_q=max_seqlen_q,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
            cp_world=cp_world,
            cp_rank=cp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
    # 非 DCP 场景：DCP 参数不是“禁止出现”，而是“只能是默认值”
    _reject_dcp_params_without_cp_world(
        causal_seqlens_kv_global, cp_rank, cp_kv_cache_interleave_size
    )
    # 非 DCP 路由恢复 V2 语义默认值
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    return_value = 0 if return_value is None else return_value
    options = dict(
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
    if (
        candidate_topk_blocks is _CANDIDATE_UNSET
        and candidate_block_size is _CANDIDATE_UNSET
    ):
        return op.default(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            topk,
            quant_mode,
            **options,
        )
    if (
        candidate_topk_blocks is _CANDIDATE_UNSET
        or candidate_block_size is _CANDIDATE_UNSET
    ):
        raise ValueError(
            "candidate_topk_blocks and candidate_block_size must be provided together"
        )
    _validate_candidate_parameters(candidate_topk_blocks, candidate_block_size)
    return op.candidate(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        topk,
        quant_mode,
        **options,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


def quant_lightning_indexer_metadata(
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
    *,
    candidate_topk_blocks=_CANDIDATE_UNSET,
    candidate_block_size=_CANDIDATE_UNSET,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    cp_world: Optional[int] = None,
    cp_rank: Optional[int] = None,
    cp_kv_cache_interleave_size: Optional[int] = None,
):
    """按 cp_world / candidate 参数路由到 DCP / V2 / V3 metadata 接口。"""
    op = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata
    if (cp_world is not None) and (cp_world < 1):
        raise ValueError(f"cp_world must be >= 1, got {cp_world}")
    if (cp_world is not None) and (cp_world > 1):
        if (
            candidate_topk_blocks is not _CANDIDATE_UNSET
            or candidate_block_size is not _CANDIDATE_UNSET
        ):
            raise ValueError("DCP is mutually exclusive with candidate")
        if causal_seqlens_kv_global is None:
            raise ValueError("DCP requires causal_seqlens_kv_global")
        if int(quant_mode) != 2:
            raise ValueError(f"DCP requires quant_mode=2 on arch22, got {quant_mode}")
        layout_q, layout_k, mask_mode, cmp_ratio = _resolve_dcp_common_attrs(
            layout_q, layout_k, mask_mode, cmp_ratio
        )
        _require_dcp_rank_and_interleave(cp_rank, cp_kv_cache_interleave_size)
        return op.dcp(
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
            causal_seqlens_kv_global=causal_seqlens_kv_global,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            cp_world=cp_world,
            cp_rank=cp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
    _reject_dcp_params_without_cp_world(
        causal_seqlens_kv_global, cp_rank, cp_kv_cache_interleave_size
    )
    options = dict(
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
    if (
        candidate_topk_blocks is _CANDIDATE_UNSET
        and candidate_block_size is _CANDIDATE_UNSET
    ):
        return op.default(
            num_heads_q, num_heads_k, head_dim, topk, quant_mode, **options
        )
    if (
        candidate_topk_blocks is _CANDIDATE_UNSET
        or candidate_block_size is _CANDIDATE_UNSET
    ):
        raise ValueError(
            "candidate_topk_blocks and candidate_block_size must be provided together"
        )
    _validate_candidate_parameters(candidate_topk_blocks, candidate_block_size)
    return op.candidate(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        quant_mode,
        **options,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


torch.compiler.allow_in_graph(quant_lightning_indexer_metadata)
