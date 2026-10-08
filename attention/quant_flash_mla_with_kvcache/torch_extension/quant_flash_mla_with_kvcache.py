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

QMLA_METADATA_OP_NAME = "quant_flash_mla_with_kvcache_metadata"

# MLA固定维度: head_dim_qk = nope(512) + rope(64) = 576, head_dim_v = 512
MLA_HEAD_DIM_QK = 576
MLA_HEAD_DIM_V = 512


class QuantMode(IntEnum):
    """quant_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    MLA_FP8_E4M3_FULLQUANT = 1
    MLA_HIF8_FULLQUANT = 2


class MaskMode(IntEnum):
    """mask_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    NO_MASK = 0
    CAUSAL = 3


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


def _calculate_max_schedule_size():
    return 4096


# 各 layout 期望的 tensor 维度数
_LAYOUT_Q_EXPECTED_NDIM = {
    "TND": 3,
    "BSND": 4,
    "BNSD": 4,
}
_LAYOUT_KV_EXPECTED_NDIM = {
    "PA_NZ": 5,
    "PA_BNBD": 4,
    "PA_BBND": 4,
}


class QuantFlashMlaWithKvcacheOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashMlaWithKvcacheOpBuilder, self).__init__(
            "quant_flash_mla_with_kvcache", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_flash_mla_with_kvcache.cpp"]

    def schema(self) -> list:
        """PyTorch operator signature."""
        return [
            "quant_flash_mla_with_kvcache_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, "
            "int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            "int? head_dim_qk=576, int? head_dim_v=512, int? mask_mode=0, "
            'str? layout_q="BSND") -> Tensor',
            "quant_flash_mla_with_kvcache(Tensor q, Tensor k_cache, Tensor q_descale, Tensor k_descale, "
            "Tensor block_table, Tensor cache_seqlens, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "Tensor? attn_mask=None, Tensor? metadata=None, "
            "int head_dim_v=512, float softmax_scale=1.0, int mask_mode=0, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="BSND", str layout_kv="PA_BNBD", str layout_out="BSND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + QMLA_METADATA_OP_NAME)
        def quant_flash_mla_with_kvcache_metadata_meta(
            cache_seqlens: torch.Tensor,
            num_heads_q: int,
            num_heads_kv: int,
            quant_mode: Union[QuantMode, int],
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            head_dim_qk: Optional[int] = MLA_HEAD_DIM_QK,
            head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            layout_q: Optional[str] = "BSND",
        ):
            max_schedule_size = _calculate_max_schedule_size()
            return torch.empty((2, max_schedule_size), dtype=torch.int32, device="meta")

        @torch.library.register_fake(
            "cann_ops_transformer::quant_flash_mla_with_kvcache"
        )
        def quant_flash_mla_with_kvcache_meta(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            block_table: torch.Tensor,
            cache_seqlens: torch.Tensor,
            quant_mode: Union[QuantMode, int],
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "PA_BNBD",
            layout_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
        ):
            if q is None:
                raise ValueError("q must not be None")
            q_expected = _LAYOUT_Q_EXPECTED_NDIM.get(layout_q)
            if q_expected is None:
                raise ValueError(
                    f"Unsupported layout_q: {layout_q!r}, expected one of TND/BSND/BNSD"
                )
            if q.dim() != q_expected:
                raise ValueError(
                    f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims"
                )
            kv_expected = _LAYOUT_KV_EXPECTED_NDIM.get(layout_kv)
            if kv_expected is None:
                raise ValueError(
                    f"Unsupported layout_kv: {layout_kv!r}, expected one of PA_BBND/PA_BNBD/PA_NZ"
                )
            if k_cache.dim() != kv_expected:
                raise ValueError(
                    f"k_cache with layout {layout_kv} expects {kv_expected} dims, but got {k_cache.dim()} dims"
                )
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                b_size = 1
                s_size = 0
                softmax_out_size = (n_size, t_size)
            else:
                if layout_q == "BSND":
                    b_size = q.size(0)
                    s_size = q.size(1)
                    n_size = q.size(2)
                else:
                    b_size = q.size(0)
                    n_size = q.size(1)
                    s_size = q.size(2)
                t_size = 0
                softmax_out_size = (b_size, n_size, s_size)

            if layout_out not in ("BSND", "BNSD", "TND", "NTD"):
                raise ValueError(
                    f"Unsupported layout_out: {layout_out!r}, expected one of BSND/BNSD/TND/NTD"
                )
            if layout_out == "BSND":
                attention_out_size = (b_size, s_size, n_size, head_dim_v)
            elif layout_out == "BNSD":
                attention_out_size = (b_size, n_size, s_size, head_dim_v)
            elif layout_out == "TND":
                attention_out_size = (t_size, n_size, head_dim_v)
            else:
                attention_out_size = (n_size, t_size, head_dim_v)

            return (
                torch.empty(attention_out_size, dtype=torch.bfloat16, device=q.device),
                torch.empty(softmax_out_size, dtype=torch.float, device=q.device),
            )


# Instantiate the builder
quant_flash_mla_with_kvcache_op_builder = QuantFlashMlaWithKvcacheOpBuilder()
quant_flash_mla_with_kvcache_op_builder._ensure_initialized()
# 预加载 C++ 扩展，避免 aclgraph/torch.compile 时在 compile 区域内首次调用 load() 触发 graph break
quant_flash_mla_with_kvcache_op_builder.load(verbose=False)


@impl(get_as_library(), QMLA_METADATA_OP_NAME, "PrivateUse1")
def quant_flash_mla_with_kvcache_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_qk: Optional[int] = MLA_HEAD_DIM_QK,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    layout_q: Optional[str] = "BSND",
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    if quant_mode not in (1, 2):
        raise ValueError(
            f"The quant_mode of quant_flash_mla_with_kvcache_metadata only supports "
            f"1 (MLA_FP8_E4M3_FULLQUANT) or 2 (MLA_HIF8_FULLQUANT), but got {quant_mode}"
        )
    if num_heads_kv != 1:
        raise ValueError(
            f"The num_heads_kv of quant_flash_mla_with_kvcache only supports 1 (MLA KV shared), "
            f"but got {num_heads_kv}"
        )
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    layout_q = "BSND" if layout_q is None else layout_q
    head_dim_qk = MLA_HEAD_DIM_QK if head_dim_qk is None else head_dim_qk
    head_dim_v = MLA_HEAD_DIM_V if head_dim_v is None else head_dim_v

    max_schedule_size = _calculate_max_schedule_size()
    output = torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")

    op_module = quant_flash_mla_with_kvcache_op_builder.load()
    return op_module.quant_flash_mla_with_kvcache_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
        output,
    )


@torch.library.register_kernel("cann_ops_transformer::" + QMLA_METADATA_OP_NAME, None)
def quant_flash_mla_with_kvcache_metadata_fallback(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_qk: Optional[int] = MLA_HEAD_DIM_QK,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    layout_q: Optional[str] = "BSND",
):
    # 处理所有 tensor 都为 None 的情况
    return quant_flash_mla_with_kvcache_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
    )


@impl(get_as_library(), quant_flash_mla_with_kvcache_op_builder.name, "PrivateUse1")
def quant_flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    quant_mode: Union[QuantMode, int],
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "PA_BNBD",
    layout_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    if quant_mode not in (1, 2):
        raise ValueError(
            f"The quant_mode of quant_flash_mla_with_kvcache only supports "
            f"1 (MLA_FP8_E4M3_FULLQUANT) or 2 (MLA_HIF8_FULLQUANT), but got {quant_mode}"
        )
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    if head_dim_v is None:
        head_dim_v = MLA_HEAD_DIM_V

    # 取 shape 前校验 q/k_cache 非空及维度
    if q is None:
        raise ValueError("q must not be None")
    if k_cache is None:
        raise ValueError("k_cache must not be None")
    q_expected = _LAYOUT_Q_EXPECTED_NDIM.get(layout_q)
    if q_expected is None:
        raise ValueError(
            f"Unsupported layout_q: {layout_q!r}, expected one of TND/BSND/BNSD"
        )
    if q.dim() != q_expected:
        raise ValueError(
            f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims"
        )
    kv_expected = _LAYOUT_KV_EXPECTED_NDIM.get(layout_kv)
    if kv_expected is None:
        raise ValueError(
            f"Unsupported layout_kv: {layout_kv!r}, expected one of PA_BBND/PA_BNBD/PA_NZ"
        )
    if k_cache.dim() != kv_expected:
        raise ValueError(
            f"k_cache with layout {layout_kv} expects {kv_expected} dims, but got {k_cache.dim()} dims"
        )
    if metadata is None:
        raise ValueError(
            "metadata should be provided by quant_flash_mla_with_kvcache_metadata"
        )
    if quant_mode == int(QuantMode.MLA_FP8_E4M3_FULLQUANT):
        if q.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"In FP8_E4M3 mode (quant_mode=1), q must be float8_e4m3fn, but got {q.dtype}"
            )
        if k_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"In FP8_E4M3 mode (quant_mode=1), k_cache must be float8_e4m3fn, but got {k_cache.dtype}"
            )

    op_module = quant_flash_mla_with_kvcache_op_builder.load()
    return op_module.quant_flash_mla_with_kvcache(
        q,
        k_cache,
        q_descale,
        k_descale,
        block_table,
        cache_seqlens,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        attn_mask,
        metadata,
        head_dim_v,
        softmax_scale,
        mask_mode,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        return_softmax_lse,
    )


quant_flash_mla_with_kvcache.QuantMode = QuantMode
quant_flash_mla_with_kvcache.MaskMode = MaskMode
quant_flash_mla_with_kvcache_metadata.QuantMode = QuantMode
quant_flash_mla_with_kvcache_metadata.MaskMode = MaskMode

quant_flash_mla_with_kvcache = (
    torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache
)
quant_flash_mla_with_kvcache_metadata = (
    torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache_metadata
)
