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
import torch_npu
from enum import IntEnum
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

FA_METADATA_OP_NAME = "mixed_quant_flash_attn_metadata"


class MaskMode(IntEnum):
    NO_MASK = 0
    CAUSAL = 3
    SLIDING_WINDOW = 4


class QuantComputeMode(IntEnum):
    A16C4_KV_MXFP4_SOFTMAX_FP32 = 1
    A16C4_KV_HIF4_SOFTMAX_FP32 = 2


_npu_core_cache = None


def get_npu_core_count():
    global _npu_core_cache
    if _npu_core_cache is not None:
        return _npu_core_cache
    try:
        device_id = torch.npu.current_device()
        limits = torch.npu.get_device_limit(device_id)
        _npu_core_cache = (limits["cube_core_num"], limits["vector_core_num"])
    except Exception:
        props = torch.npu.get_device_properties()
        _npu_core_cache = (props.cube_core_num, props.vector_core_num)
    return _npu_core_cache


def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    if batch_size is not None:
        return batch_size
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        return cu_seqlens_q.size(0) - 1
    elif seqused_q is not None:
        return seqused_q.size(0)
    return 0


def _calculate_metadata_size(batch_size, num_heads_kv):
    """计算 metadata tensor 的对齐后大小"""
    aic_num, aiv_num = get_npu_core_count()
    metadata_size = ((aic_num + aiv_num) * batch_size * num_heads_kv + 1) * 16
    return ((metadata_size + 4095) // 4096) * 4096


class MixedQuantFlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(MixedQuantFlashAttenOpBuilder, self).__init__(
            "mixed_quant_flash_attn", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/mixed_quant_flash_attn.cpp"]

    def schema(self) -> list[str]:
        """PyTorch operator signature."""
        return [
            "mixed_quant_flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, int quant_compute_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, Tensor? seqused_kv=None,"
            "int? batch_size=None, int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            "int? mask_mode=0, int? win_left=-1, int? win_right=-1, "
            'str? layout_q="BSND", str? layout_kv="PA_BBND", str? layout_attn_out="BSND") -> Tensor',
            "mixed_quant_flash_attn(Tensor q, Tensor k, Tensor v, Tensor k_descale, Tensor v_descale,"
            "Tensor? block_table=None, Tensor? cu_seqlens_q=None,"
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, Tensor? attn_mask=None,"
            "Tensor? metadata=None,"
            "int quant_compute_mode=1, float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1,"
            "int max_seqlen_q=-1, int max_seqlen_kv=-1,"
            'str layout_q="BSND", str layout_kv="PA_BBND", str layout_attn_out="BSND",'
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + FA_METADATA_OP_NAME)
        def mixed_quant_flash_attn_metadata_meta(
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            quant_compute_mode: QuantComputeMode,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            mask_mode: Optional[MaskMode] = MaskMode.NO_MASK,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "PA_BBND",
            layout_attn_out: Optional[str] = "BSND",
        ) -> torch.Tensor:
            batch_size = (
                _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
                if batch_size is None
                else batch_size
            )
            metadata_size = _calculate_metadata_size(batch_size, num_heads_kv)
            return torch.empty((metadata_size,), dtype=torch.int32, device="npu")

        @impl(get_as_library(), self.name, "Meta")
        def mixed_quant_flash_attn_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            k_descale: torch.Tensor,
            v_descale: torch.Tensor,
            block_table: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            quant_compute_mode: Optional[
                QuantComputeMode
            ] = QuantComputeMode.A16C4_KV_MXFP4_SOFTMAX_FP32,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[MaskMode] = MaskMode.NO_MASK,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "PA_BBND",
            layout_attn_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                d_size = v.size(2)
                softmax_out_size = (n_size, t_size)
            elif layout_q == "BSND":
                b_size = q.size(0)
                s_size = q.size(1)
                n_size = q.size(2)
                d_size = v.size(3)
                softmax_out_size = (b_size, n_size, s_size)
            else:
                b_size = q.size(0)
                n_size = q.size(1)
                s_size = q.size(2)
                d_size = v.size(3)
                softmax_out_size = (b_size, n_size, s_size)

            if layout_attn_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: f"When the layout of output is TND, the layout of query must be TND, but got {layout_q}",
                )
                attention_out_size = (t_size, n_size, d_size)
            elif layout_attn_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}",
                )
                attention_out_size = (b_size, n_size, s_size, d_size)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: f"When the layout of output is BSND, the layout of query must be BNSD or BSND, but got {layout_q}",
                )
                attention_out_size = (b_size, s_size, n_size, d_size)

            return (
                torch.empty(attention_out_size, dtype=q.dtype, device="meta"),
                torch.empty(softmax_out_size, dtype=torch.float, device="meta"),
            )


# Instantiate the builder
mixed_quant_flash_attn_op_builder = MixedQuantFlashAttenOpBuilder()
mixed_quant_flash_attn_op_builder._ensure_initialized()


@impl(get_as_library(), FA_METADATA_OP_NAME, "PrivateUse1")
def mixed_quant_flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_compute_mode: QuantComputeMode,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    mask_mode: Optional[MaskMode] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_attn_out: Optional[str] = "BSND",
) -> torch.Tensor:
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    op_module = mixed_quant_flash_attn_op_builder.load()
    batch_size = (
        _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
        if batch_size is None
        else batch_size
    )
    quant_compute_mode = (
        quant_compute_mode.value
        if isinstance(quant_compute_mode, IntEnum)
        else quant_compute_mode
    )
    mask_mode = mask_mode.value if isinstance(mask_mode, IntEnum) else mask_mode

    metadata_size = _calculate_metadata_size(batch_size, num_heads_kv)
    output = torch.empty((metadata_size,), dtype=torch.int32, device="npu")

    return op_module.mixed_quant_flash_attn_metadata(
        cu_seqlens_q,
        seqused_q,
        seqused_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_compute_mode,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        mask_mode,
        win_left,
        win_right,
        layout_q,
        layout_kv,
        layout_attn_out,
        output,
    )


_mixed_quant_flash_attn_metadata = mixed_quant_flash_attn_metadata


@torch.library.register_kernel("cann_ops_transformer::" + FA_METADATA_OP_NAME, None)
def mixed_quant_flash_attn_metadata_fallback(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_compute_mode: QuantComputeMode,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    mask_mode: Optional[MaskMode] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_attn_out: Optional[str] = "BSND",
) -> torch.Tensor:
    return _mixed_quant_flash_attn_metadata(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_compute_mode=quant_compute_mode,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_attn_out=layout_attn_out,
    )


@impl(get_as_library(), mixed_quant_flash_attn_op_builder.name, "PrivateUse1")
def mixed_quant_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    quant_compute_mode: Optional[
        QuantComputeMode
    ] = QuantComputeMode.A16C4_KV_MXFP4_SOFTMAX_FP32,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[MaskMode] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_attn_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = mixed_quant_flash_attn_op_builder.load()
    quant_compute_mode = (
        quant_compute_mode.value
        if isinstance(quant_compute_mode, IntEnum)
        else quant_compute_mode
    )
    mask_mode = mask_mode.value if isinstance(mask_mode, IntEnum) else mask_mode

    return op_module.mixed_quant_flash_attn(
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
    )


mixed_quant_flash_attn = torch.ops.cann_ops_transformer.mixed_quant_flash_attn
mixed_quant_flash_attn.MaskMode = MaskMode
mixed_quant_flash_attn.QuantComputeMode = QuantComputeMode

mixed_quant_flash_attn_metadata = (
    torch.ops.cann_ops_transformer.mixed_quant_flash_attn_metadata
)
mixed_quant_flash_attn_metadata.MaskMode = MaskMode
mixed_quant_flash_attn_metadata.QuantComputeMode = QuantComputeMode
