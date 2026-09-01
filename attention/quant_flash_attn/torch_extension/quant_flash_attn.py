# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
from typing import Optional, Union

import torch
import torch_npu
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl

QFA_METADATA_OP_NAME = "quant_flash_attn_metadata"


class QuantMode(IntEnum):
    """quant_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 1
    A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP32 = 2
    A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16 = 3
    A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP16 = 4
    A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16 = 5
    A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 6
    A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 0
    A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16 = 8
    A4C4_QKV_HIF4_P_HIF4_LEVEL2_SOFTMAX_FP16 = 9
    A4C4_QKV_HIF4_P_HIF4_LEVEL3_SOFTMAX_FP16 = 10


class MaskMode(IntEnum):
    """mask_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    NO_MASK = 0
    CAUSAL = 3
    SLIDING_WINDOW = 4


def _resolve_quant_mode(quant_mode: Union[str, int, "QuantMode"]) -> int:
    """对外 str/int/IntEnum quant_mode 统一为传给算子侧的 int；校验取值合法性。"""
    if isinstance(quant_mode, str):
        try:
            return int(QuantMode[quant_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in QuantMode)
            raise ValueError(
                f"quant_mode should be one of [{valid}], but got {quant_mode!r}"
            ) from exc
    return int(QuantMode(quant_mode))


def _resolve_mask_mode(mask_mode: Union[str, int, "MaskMode", None]) -> int:
    """对外 str/int/IntEnum/None mask_mode 统一为传给算子侧的 int；None 视为默认 0。"""
    if mask_mode is None:
        return int(MaskMode.NO_MASK)
    if isinstance(mask_mode, str):
        try:
            return int(MaskMode[mask_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in MaskMode)
            raise ValueError(
                f"mask_mode should be one of [{valid}], but got {mask_mode!r}"
            ) from exc
    return int(MaskMode(mask_mode))


def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    if seqused_q is not None:
        return seqused_q.size(0)
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        return cu_seqlens_q.size(0) - 1
    elif batch_size is not None:
        return batch_size
    return 0


def _calculate_max_schedule_size():
    return 4096


# 各 layout 期望的 Q/K/V tensor 维度数 (N2TGD 是 descale 专用 layout, 不在此表)
_LAYOUT_EXPECTED_NDIM = {
    "TND": 3,
    "NTD": 3,
    "BSND": 4,
    "BNSD": 4,
    "PA_NZ": 5,
    "PA_BNBD": 4,
    "PA_BBND": 4,
}


class QuantFlashAttnOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashAttnOpBuilder, self).__init__(
            "quant_flash_attn", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_flash_attn.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "quant_flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "int? batch_size=None, int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            "int? head_dim_v=None, int? mask_mode=0, int? win_left=-1, int? win_right=-1, "
            'str? layout_q="BSND", str? layout_q_descale="BSND", '
            'str? layout_kv="BSND", str? layout_out="BSND", '
            "bool is_grad_enabled=False) -> Tensor",
            "quant_flash_attn(Tensor q, Tensor k, Tensor v, "
            "Tensor q_descale, Tensor k_descale, Tensor v_descale, int quant_mode, "
            "Tensor? block_table=None, Tensor? p_scale=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, "
            "float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="BSND", str layout_q_descale="BSND", str layout_kv="BSND", str layout_out="BSND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + QFA_METADATA_OP_NAME)
        def quant_flash_attn_metadata_meta(
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            quant_mode: Union[QuantMode, int],
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            head_dim_v: Optional[int] = None,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_q_descale: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
            is_grad_enabled: Optional[bool] = False,
        ):
            max_schedule_size = _calculate_max_schedule_size()
            return torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")

        @impl(get_as_library(), self.name, "Meta")
        def quant_flash_attn_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            v_descale: torch.Tensor,
            quant_mode: Union[QuantMode, int],
            block_table: Optional[torch.Tensor] = None,
            p_scale: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_q_descale: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
        ):
            # 取 shape 前校验 q/v 非空及维度, 避免 None 或维度不足导致 AttributeError/IndexError
            torch._check(q is not None, lambda: "q must not be None")
            torch._check(v is not None, lambda: "v must not be None")
            q_expected = _LAYOUT_EXPECTED_NDIM.get(layout_q)
            torch._check(
                q_expected is not None,
                lambda: f"Unsupported layout_q: {layout_q!r}, expected one of TND/NTD/BSND/BNSD",
            )
            torch._check(
                q.dim() == q_expected,
                lambda: f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims",
            )
            kv_expected = _LAYOUT_EXPECTED_NDIM.get(layout_kv)
            torch._check(
                kv_expected is not None,
                lambda: f"Unsupported layout_kv: {layout_kv!r}, expected one of TND/BSND/BNSD/PA_NZ/PA_BNBD/PA_BBND",
            )
            torch._check(
                v.dim() == kv_expected,
                lambda: f"v with layout {layout_kv} expects {kv_expected} dims, but got {v.dim()} dims",
            )
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                softmax_out_size = (n_size, t_size)
            elif layout_q == "BSND":
                b_size = q.size(0)
                s_size = q.size(1)
                n_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)
            else:
                b_size = q.size(0)
                n_size = q.size(1)
                s_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)

            if layout_kv == "PA_NZ":
                d_size = v.size(2) * v.size(4)
            elif layout_kv == "TND":
                d_size = v.size(2)
            else:
                d_size = v.size(3)

            if layout_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: f"When the layout of output is TND, the layout of query must be TND, but got {layout_q}",
                )
                attention_out_size = (t_size, n_size, d_size)
            elif layout_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}",
                )
                attention_out_size = (b_size, n_size, s_size, d_size)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: f"When the layout of output is BSND, the layout of query "
                    f"must be BNSD or BSND, but got {layout_q}",
                )
                attention_out_size = (b_size, s_size, n_size, d_size)

            return (
                torch.empty(attention_out_size, dtype=torch.bfloat16, device="meta"),
                torch.empty(softmax_out_size, dtype=torch.float32, device="meta"),
            )


# Instantiate the builder
quant_flash_attn_op_builder = QuantFlashAttnOpBuilder()
quant_flash_attn_op_builder._ensure_initialized()


@impl(get_as_library(), QFA_METADATA_OP_NAME, "PrivateUse1")
def quant_flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_v: Optional[int] = None,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_q_descale: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    is_grad_enabled: Optional[bool] = False,
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    torch._check(
        quant_mode in (1, 5, 0),
        lambda: f"The quant_mode of quant_flash_attn_metadata only supports 1 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32), "
        f"5 (A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16) "
        f"or 0 (A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32), but got {quant_mode}",
    )
    if layout_q == "TND":
        torch._check(
            batch_size == None,
            lambda: f"When the layout of query is TND, the attribute batch_size of quant_flash_attn_metadata must be None, but got {batch_size}.",
        )
    torch._check(
        is_grad_enabled is not True or quant_mode == 0,
        lambda: f"When is_grad_enabled is True, quant_mode must be 0, but got quant_mode={quant_mode}",
    )
    batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_q_descale = "BSND" if layout_q_descale is None else layout_q_descale
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out
    head_dim_v = head_dim if head_dim_v is None else head_dim_v

    max_schedule_size = _calculate_max_schedule_size()
    output = torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")

    op_module = quant_flash_attn_op_builder.load()
    return op_module.quant_flash_attn_metadata(
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        head_dim_v,
        quant_mode,
        mask_mode,
        win_left,
        win_right,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        is_grad_enabled,
        output,
    )


@torch.library.register_kernel("cann_ops_transformer::" + QFA_METADATA_OP_NAME, None)
def quant_flash_attn_metadata_fallback(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_v: Optional[int] = None,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_q_descale: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    is_grad_enabled: Optional[bool] = False,
):
    # 处理所有 tensor 都为 None 的情况
    return quant_flash_attn_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_v,
        mask_mode,
        win_left,
        win_right,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        is_grad_enabled,
    )


@impl(get_as_library(), quant_flash_attn_op_builder.name, "PrivateUse1")
def quant_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: Union[QuantMode, int],
    block_table: Optional[torch.Tensor] = None,
    p_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_q_descale: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    torch._check(
        quant_mode in (1, 5, 0),
        lambda: f"The quant_mode of quant_flash_attn only supports 1 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32), "
        f"5 (A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16) "
        f"or 0 (A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32), but got {quant_mode}",
    )
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)

    # 取 shape 前校验 q/v 非空及维度, 避免 None 或维度不足导致 AttributeError/IndexError
    torch._check(q is not None, lambda: "q must not be None")
    torch._check(v is not None, lambda: "v must not be None")
    q_expected = _LAYOUT_EXPECTED_NDIM.get(layout_q)
    torch._check(
        q_expected is not None,
        lambda: f"Unsupported layout_q: {layout_q!r}, expected one of TND/NTD/BSND/BNSD",
    )
    torch._check(
        q.dim() == q_expected,
        lambda: f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims",
    )
    kv_expected = _LAYOUT_EXPECTED_NDIM.get(layout_kv)
    torch._check(
        kv_expected is not None,
        lambda: f"Unsupported layout_kv: {layout_kv!r}, expected one of TND/BSND/BNSD/PA_NZ/PA_BNBD/PA_BBND",
    )
    torch._check(
        v.dim() == kv_expected,
        lambda: f"v with layout {layout_kv} expects {kv_expected} dims, but got {v.dim()} dims",
    )

    if quant_mode == int(QuantMode.A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16):
        if q.dtype != torch.uint8:
            raise ValueError(
                f"In MxFP4 mode (quant_mode=5), q must be uint8, but got {q.dtype}"
            )
        if k.dtype != torch.uint8:
            raise ValueError(
                f"In MxFP4 mode (quant_mode=5), k must be uint8, but got {k.dtype}"
            )
        if v.dtype != torch.uint8:
            raise ValueError(
                f"In MxFP4 mode (quant_mode=5), v must be uint8, but got {v.dtype}"
            )

    op_module = quant_flash_attn_op_builder.load()
    return op_module.quant_flash_attn(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        quant_mode,
        block_table,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        return_softmax_lse,
    )


quant_flash_attn.QuantMode = QuantMode
quant_flash_attn.MaskMode = MaskMode
quant_flash_attn_metadata.QuantMode = QuantMode
quant_flash_attn_metadata.MaskMode = MaskMode
