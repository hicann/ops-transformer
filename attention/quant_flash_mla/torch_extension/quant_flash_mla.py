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

QFMLA_METADATA_OP_NAME = "quant_flash_mla_metadata"
METADATA_STRIDE = 16

# MLA固定维度: head_dim_qk = nope(512) + rope(64) = 576, head_dim_v = 512
MLA_HEAD_DIM_QK = 576
MLA_HEAD_DIM_V = 512
# arch22 int8全量化: rope分离, q/k_cache为nope维度(512), rope单独传入(64)
MLA_INT8_HEAD_DIM_QK = 512
MLA_ROPE_DIM = 64


class QuantMode(IntEnum):
    """quant_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR = 0
    A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR = 1
    MLA_INT8_FULLQUANT = (
        2  # arch22: Q/KV INT8（Q per-token-head、KV per-tensor），rope分离
    )


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
    return int(quant_mode)


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
    return int(mask_mode)


def _normalize_optional_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """空 (0,) 张量归一成 None。

    ttk 等 CSV 驱动框架对未填写的 optional 输入会以空 (0,) slot 张量传入;
    host 端 CheckSeqLens 会把 size(0)=0 判为非法 shape ("cuSeqLens shape must
    be (batchSize+1,), but got 0")。归一成 None 才能正确表达"未提供"语义。
    """
    if t is not None and t.numel() == 0:
        return None
    return t


def _resolve_softmax_scale(q, softmax_scale):
    """Use the documented default only when softmax_scale is omitted or None."""
    if softmax_scale is None:
        torch._check(q.dim() > 0, lambda: "q must have at least one dimension")
        head_dim_qk = q.size(-1)
        torch._check(head_dim_qk > 0, lambda: "head_dim_qk must be greater than 0")
        return head_dim_qk**-0.5
    return softmax_scale


def _get_core_nums():
    """Return the core counts used by the metadata AICPU scheduler."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return 36, 72
    props = npu.get_device_properties()
    return props.cube_core_num, props.vector_core_num


def _calculate_max_schedule_size(batch_size, num_heads_kv):
    """Allocate one row for the worst-case MLA metadata section count."""
    align_size = 4096
    batch_size = batch_size if batch_size and batch_size > 0 else 1
    num_heads_kv = num_heads_kv if num_heads_kv and num_heads_kv > 0 else 1
    aic_num, aiv_num = _get_core_nums()
    schedule_size = METADATA_STRIDE + (
        (aic_num + aiv_num) * METADATA_STRIDE * batch_size * num_heads_kv
    )
    return ((schedule_size + align_size - 1) // align_size) * align_size


# 各 layout 期望的 tensor 维度数
_LAYOUT_Q_EXPECTED_NDIM = {
    "TND": 3,
    "BSND": 4,
    "BNSD": 4,
    "BSH": 3,  # arch22 int8全量化
}
_LAYOUT_KV_EXPECTED_NDIM = {
    "PA_NZ": 5,
    "PA_BNBD": 4,
    "PA_BBND": 4,
}

_SUPPORTED_NUM_HEADS_Q = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
# arch22 int8全量化: 对齐IFA现状 G∈{1,2,4,8,16,32,64,128}
_SUPPORTED_NUM_HEADS_Q_INT8 = (1, 2, 4, 8, 16, 32, 64, 128)


def _check_mla_layout_pair(layout_q, layout_out) -> None:
    # Reject unsupported pairs before allocating an empty output (for example,
    # TND -> BSND leaves the C++ output sequence dimension at zero).
    torch._check(
        (layout_q == "TND" and layout_out in ("TND", "NTD"))
        or (layout_q == "BSND" and layout_out == "BSND")
        or (layout_q == "BNSD" and layout_out == "BNSD")
        or (layout_q == "BSH" and layout_out == "BSH"),
        lambda: (
            "layout_q and layout_out must be one of BSND->BSND, BNSD->BNSD, "
            f"TND->TND, TND->NTD (TND_NTD), BSH->BSH (int8), but got {layout_q}->{layout_out}"
        ),
    )


def _check_required_mla_parameters(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    cu_seqlens_q,
    attn_mask,
    metadata,
    mask_mode,
    layout_q,
    layout_kv,
    layout_out,
) -> None:
    # Keep Eager and FakeTensor presence checks consistent with the Torch API
    # contract. Check existence only; tensor values remain the caller's concern.
    for name, value in (
        ("q", q),
        ("k_cache", k_cache),
        ("q_descale", q_descale),
        ("k_descale", k_descale),
        ("block_table", block_table),
        ("cache_seqlens", cache_seqlens),
        ("quant_mode", quant_mode),
        ("layout_q", layout_q),
        ("layout_kv", layout_kv),
        ("layout_out", layout_out),
        ("metadata", metadata),
    ):
        torch._check(value is not None, lambda name=name: f"{name} must be provided")
    if layout_q == "TND":
        torch._check(
            cu_seqlens_q is not None,
            lambda: "cu_seqlens_q must be provided when layout_q is TND",
        )
    if mask_mode == 3:
        torch._check(
            attn_mask is not None,
            lambda: "attn_mask must be provided when mask_mode is CAUSAL (3)",
        )


def _check_int32_1d(name: str, tensor: torch.Tensor) -> None:
    torch._check(tensor is not None, lambda: f"{name} must be provided")
    torch._check(
        tensor.dtype == torch.int32,
        lambda: f"{name} must be int32, but got {tensor.dtype}",
    )
    torch._check(
        tensor.dim() == 1, lambda: f"{name} must be 1D, but got {tensor.dim()}D"
    )
    torch._check(tensor.numel() > 0, lambda: f"{name} must be non-empty")


def _check_metadata_parameters(
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
) -> None:
    torch._check(
        quant_mode
        in (
            int(QuantMode.A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR),
            int(QuantMode.A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR),
            int(QuantMode.MLA_INT8_FULLQUANT),
        ),
        lambda: f"quant_mode only supports 0 (A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR), "
        f"1 (A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR) or 2 (MLA_INT8_FULLQUANT, arch22), but got {quant_mode}",
    )
    is_int8 = quant_mode == int(QuantMode.MLA_INT8_FULLQUANT)
    torch._check(
        layout_q in ("TND", "BSND", "BNSD", "BSH")
        if is_int8
        else layout_q in ("TND", "BSND", "BNSD"),
        lambda: f"layout_q only supports 'TND'/'BSH' (int8) or 'TND'/'BSND'/'BNSD', but got {layout_q!r}",
    )
    expect_num_heads = (
        _SUPPORTED_NUM_HEADS_Q_INT8 if is_int8 else _SUPPORTED_NUM_HEADS_Q
    )
    torch._check(
        num_heads_q in expect_num_heads,
        lambda: f"num_heads_q must be one of {expect_num_heads}, but got {num_heads_q}",
    )
    torch._check(
        num_heads_kv == 1,
        lambda: f"num_heads_kv only supports 1, but got {num_heads_kv}",
    )
    expect_head_dim_qk = MLA_INT8_HEAD_DIM_QK if is_int8 else MLA_HEAD_DIM_QK
    torch._check(
        head_dim_qk == expect_head_dim_qk,
        lambda: f"head_dim_qk only supports {expect_head_dim_qk}"
        + (" (nope, rope split)" if is_int8 else "")
        + f", but got {head_dim_qk}",
    )
    torch._check(
        head_dim_v == MLA_HEAD_DIM_V,
        lambda: f"head_dim_v only supports 512, but got {head_dim_v}",
    )
    torch._check(
        mask_mode in (0, 3),
        lambda: f"mask_mode only supports 0 or 3, but got {mask_mode}",
    )
    torch._check(
        max_seqlen_kv == -1 or max_seqlen_kv >= 0,
        lambda: f"max_seqlen_kv must be -1 or greater than or equal to 0, but got {max_seqlen_kv}",
    )
    # max_seqlen_q 上限: fp8/hif8 为 16, arch22 int8 为 32 (与主算子/metadata aclnn 校验一致)
    max_seq_q_limit = 32 if is_int8 else 16
    # 下界: arch22 int8全量化要求 >= 1; 其余场景沿用上游允许 0
    lower_seq_q = 1 if is_int8 else 0
    torch._check(
        max_seqlen_q == -1
        or (max_seqlen_q >= lower_seq_q and max_seqlen_q <= max_seq_q_limit),
        lambda: f"max_seqlen_q must be -1 or in range [{lower_seq_q}, {max_seq_q_limit}], but got {max_seqlen_q}",
    )
    # cache_seqlens dtype: int32; arch22 int8全量化放行 int64 (对齐FIA调用现状)
    torch._check(
        cache_seqlens.dtype == torch.int32
        or (is_int8 and cache_seqlens.dtype == torch.int64),
        lambda: f"cache_seqlens must be int32 (int64 in int8 mode), but got {cache_seqlens.dtype}",
    )
    batch_size = cache_seqlens.numel()
    if layout_q == "TND":
        _check_int32_1d("cu_seqlens_q", cu_seqlens_q)
        torch._check(
            cu_seqlens_q.numel() == batch_size + 1,
            lambda: f"cu_seqlens_q shape must be (B+1,), but got {tuple(cu_seqlens_q.shape)} for B={batch_size}",
        )
        # TND 由 cu_seqlens_q 提供精确 Q 长度, 不强制 seqused_q/max_seqlen_q
    else:
        torch._check(
            cu_seqlens_q is None,
            lambda: "cu_seqlens_q is only supported when layout_q is TND",
        )
        # 非 TND (BSND/BNSD): seqused_q 与 max_seqlen_q 至少传入其一
        torch._check(
            seqused_q is not None or max_seqlen_q >= 0,
            lambda: "When layout_q is not TND, at least one of seqused_q or max_seqlen_q must be provided",
        )
    if seqused_q is not None:
        _check_int32_1d("seqused_q", seqused_q)
        torch._check(
            seqused_q.numel() == batch_size,
            lambda: f"seqused_q shape must be (B,), but got {tuple(seqused_q.shape)} for B={batch_size}",
        )


class QuantFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashMlaOpBuilder, self).__init__(
            "quant_flash_mla", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_flash_mla.cpp"]

    def schema(self) -> list:
        """PyTorch operator signature."""
        return [
            "quant_flash_mla_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, "
            "int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            "int? head_dim_qk=576, int? head_dim_v=512, int? mask_mode=0, "
            'str? layout_q="TND") -> Tensor',
            "quant_flash_mla(Tensor q, Tensor k_cache, Tensor q_descale, Tensor k_descale, "
            "Tensor block_table, Tensor cache_seqlens, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "Tensor? attn_mask=None, Tensor? metadata=None, "
            "Tensor? q_rope=None, Tensor? k_rope=None, "
            "int head_dim_v=512, float? softmax_scale=None, int mask_mode=0, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="TND", str layout_kv="PA_BNBD", str layout_out="TND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + QFMLA_METADATA_OP_NAME)
        def quant_flash_mla_metadata_meta(
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
            layout_q: Optional[str] = "TND",
        ):
            batch_size = cache_seqlens.size(0) if cache_seqlens.dim() > 0 else 0
            max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv)
            return torch.empty((2, max_schedule_size), dtype=torch.int32, device="meta")

        @torch.library.register_fake("cann_ops_transformer::quant_flash_mla")
        def quant_flash_mla_meta(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            block_table: torch.Tensor = None,
            cache_seqlens: torch.Tensor = None,
            quant_mode: Union[QuantMode, int] = 1,
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            q_rope: Optional[torch.Tensor] = None,
            k_rope: Optional[torch.Tensor] = None,
            head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
            softmax_scale: Optional[float] = None,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "TND",
            layout_kv: Optional[str] = "PA_BNBD",
            layout_out: Optional[str] = "TND",
            return_softmax_lse: Optional[bool] = False,
        ):
            _check_required_mla_parameters(
                q,
                k_cache,
                q_descale,
                k_descale,
                block_table,
                cache_seqlens,
                quant_mode,
                cu_seqlens_q,
                attn_mask,
                metadata,
                mask_mode,
                layout_q,
                layout_kv,
                layout_out,
            )
            torch._check(
                q.dim() >= 2,
                lambda: f"q must have at least 2 dimensions, but got {q.dim()}D",
            )
            softmax_scale = _resolve_softmax_scale(q, softmax_scale)
            # 形状推导与 csrc/quant_flash_mla.cpp 运行时保持同一套 layout_q/layout_out 分支
            layout_q_str = str(layout_q).upper() if layout_q else "TND"
            layout_out_str = str(layout_out).upper() if layout_out else "TND"
            _check_mla_layout_pair(layout_q_str, layout_out_str)
            if layout_q_str == "TND":
                t_size, n_size = q.size(0), q.size(1)
                if cu_seqlens_q is not None and cu_seqlens_q.dim() > 0:
                    b_size = cu_seqlens_q.size(0) - 1
                else:
                    b_size = cache_seqlens.size(0) if cache_seqlens.dim() > 0 else 1
                s_size = (
                    max_seqlen_q if max_seqlen_q is not None and max_seqlen_q > 0 else 0
                )
            elif layout_q_str == "BSND":
                b_size, s_size, n_size = q.size(0), q.size(1), q.size(2)
                t_size = 0
            elif layout_q_str == "BSH":
                # arch22 int8全量化: 3D (B, S, N*512)
                b_size, s_size = q.size(0), q.size(1)
                n_size = q.size(2) // MLA_INT8_HEAD_DIM_QK
                t_size = 0
            else:  # BNSD
                b_size, n_size, s_size = q.size(0), q.size(1), q.size(2)
                t_size = 0
            # lse: TND 输入为 (N, T), BSND/BNSD 输入为 (B, N, S)
            if return_softmax_lse:
                softmax_out_size = (
                    (n_size, t_size)
                    if layout_q_str == "TND"
                    else (b_size, n_size, s_size)
                )
            else:
                softmax_out_size = (0,)
            # attn_out: BSND/BNSD 出 4D, TND/NTD/BSH 出 3D
            if layout_out_str == "BSND":
                attention_out_size = (b_size, s_size, n_size, head_dim_v)
            elif layout_out_str == "BNSD":
                attention_out_size = (b_size, n_size, s_size, head_dim_v)
            elif layout_out_str == "BSH":
                # arch22 int8全量化: 输出跟随q layout, 3D (B, S, N*512)
                attention_out_size = (b_size, s_size, n_size * head_dim_v)
            elif layout_out_str == "TND":
                attention_out_size = (t_size, n_size, head_dim_v)
            else:  # NTD
                attention_out_size = (n_size, t_size, head_dim_v)

            return (
                torch.empty(attention_out_size, dtype=torch.bfloat16, device=q.device),
                torch.empty(softmax_out_size, dtype=torch.float, device=q.device),
            )


# Instantiate the builder
quant_flash_mla_op_builder = QuantFlashMlaOpBuilder()
quant_flash_mla_op_builder._ensure_initialized()
# 预加载 C++ 扩展，避免 aclgraph/torch.compile 时在 compile 区域内首次调用 load() 触发 graph break
quant_flash_mla_op_builder.load(verbose=False)


@impl(get_as_library(), QFMLA_METADATA_OP_NAME, "PrivateUse1")
def quant_flash_mla_metadata(
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
    layout_q: Optional[str] = "TND",
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    if quant_mode not in (0, 1, 2):
        raise ValueError(
            f"The quant_mode of quant_flash_mla_metadata only supports "
            f"0 (A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR), 1 (A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR) or "
            f"2 (MLA_INT8_FULLQUANT, arch22), but got {quant_mode}"
        )
    if num_heads_kv != 1:
        raise ValueError(
            f"The num_heads_kv of quant_flash_mla only supports 1 (MLA KV shared), "
            f"but got {num_heads_kv}"
        )
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    is_int8 = quant_mode == int(QuantMode.MLA_INT8_FULLQUANT)
    expect_head_dim_qk = MLA_INT8_HEAD_DIM_QK if is_int8 else MLA_HEAD_DIM_QK
    head_dim_qk = expect_head_dim_qk if head_dim_qk is None else head_dim_qk
    head_dim_v = MLA_HEAD_DIM_V if head_dim_v is None else head_dim_v
    cu_seqlens_q = _normalize_optional_tensor(cu_seqlens_q)
    seqused_q = _normalize_optional_tensor(seqused_q)

    _check_metadata_parameters(
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

    max_schedule_size = _calculate_max_schedule_size(
        cache_seqlens.size(0), num_heads_kv
    )
    output = torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")

    op_module = quant_flash_mla_op_builder.load()
    return op_module.quant_flash_mla_metadata(
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


@torch.library.register_kernel("cann_ops_transformer::" + QFMLA_METADATA_OP_NAME, None)
def quant_flash_mla_metadata_fallback(
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
    layout_q: Optional[str] = "TND",
):
    # 处理所有 tensor 都为 None 的情况
    return quant_flash_mla_metadata(
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


@impl(get_as_library(), quant_flash_mla_op_builder.name, "PrivateUse1")
def quant_flash_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    block_table: torch.Tensor = None,
    cache_seqlens: torch.Tensor = None,
    quant_mode: Union[QuantMode, int] = 1,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    softmax_scale: Optional[float] = None,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "TND",
    layout_kv: Optional[str] = "PA_BNBD",
    layout_out: Optional[str] = "TND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    _check_required_mla_parameters(
        q,
        k_cache,
        q_descale,
        k_descale,
        block_table,
        cache_seqlens,
        quant_mode,
        cu_seqlens_q,
        attn_mask,
        metadata,
        mask_mode,
        layout_q,
        layout_kv,
        layout_out,
    )
    if quant_mode not in (0, 1, 2):
        raise ValueError(
            f"The quant_mode of quant_flash_mla only supports "
            f"0 (A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR), 1 (A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR) or "
            f"2 (MLA_INT8_FULLQUANT, arch22), but got {quant_mode}"
        )
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    is_int8 = quant_mode == int(QuantMode.MLA_INT8_FULLQUANT)
    if head_dim_v is None:
        head_dim_v = MLA_HEAD_DIM_V
    cu_seqlens_q = _normalize_optional_tensor(cu_seqlens_q)
    seqused_q = _normalize_optional_tensor(seqused_q)
    q_rope = _normalize_optional_tensor(q_rope)
    k_rope = _normalize_optional_tensor(k_rope)

    if is_int8:
        # arch22 int8全量化: rope分离必传（A2必须传入）; KV descale统一由k_descale承载
        torch._check(
            q_rope is not None and k_rope is not None,
            lambda: "q_rope and k_rope must be provided when quant_mode is 2 (MLA_INT8_FULLQUANT)",
        )
        for name, tensor in (("q_rope", q_rope), ("k_rope", k_rope)):
            torch._check(
                tensor.dtype == torch.bfloat16,
                lambda name=name, tensor=tensor: (
                    f"{name} must be bfloat16 in int8 mode, but got {tensor.dtype}"
                ),
            )
        torch._check(
            return_softmax_lse is False,
            lambda: "return_softmax_lse is not supported in int8 mode",
        )
        # arch22 kernel 侧 cu_seqlens_q 按 uint64(IFA actualSeqLengthsQ)读取:
        # int32 输入上转 int64, 避免 int32 缓冲被按 8 字节误读
        if cu_seqlens_q is not None and cu_seqlens_q.dtype != torch.int64:
            cu_seqlens_q = cu_seqlens_q.to(torch.int64)
    else:
        # fp8/hif8: rope 不支持传入（A5无需传入保持默认）
        torch._check(
            q_rope is None and k_rope is None,
            lambda: "q_rope/k_rope is only supported when quant_mode is 2 (MLA_INT8_FULLQUANT)",
        )

    # Prevent empty outputs from being short-circuited by the framework before
    # host tiling validation. QFA applies the same guard to output-related dims.
    # Rank errors remain the responsibility of the C++ checker.
    if q.dim() >= 2:
        torch._check(
            all(size > 0 for size in (q.size(0), q.size(1), head_dim_v)),
            lambda: (
                "The output-related dims T/N of q and head_dim_v must be greater than 0, "
                f"but got q.shape={tuple(q.shape)}, head_dim_v={head_dim_v}"
            ),
        )
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
    if quant_mode == int(
        QuantMode.A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR
    ):
        if q.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"In FP8_E4M3 mode (quant_mode=1), q must be float8_e4m3fn, but got {q.dtype}"
            )
        if k_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"In FP8_E4M3 mode (quant_mode=1), k_cache must be float8_e4m3fn, but got {k_cache.dtype}"
            )
    elif quant_mode == int(QuantMode.A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR):
        # HIF8: torch侧无原生hifloat8 dtype, 以uint8承载hif8编码, csrc侧映射为ACL_HIFLOAT8
        if q.dtype != torch.uint8:
            raise ValueError(
                f"In HIF8 mode (quant_mode=0), q must be uint8 (hif8 encoded), but got {q.dtype}"
            )
        if k_cache.dtype != torch.uint8:
            raise ValueError(
                f"In HIF8 mode (quant_mode=0), k_cache must be uint8 (hif8 encoded), but got {k_cache.dtype}"
            )
    elif quant_mode == int(QuantMode.MLA_INT8_FULLQUANT):
        # arch22 int8全量化: q/k_cache为int8, rope分离bf16, descale fp32, 输出bf16
        if q.dtype != torch.int8:
            raise ValueError(
                f"In INT8 mode (quant_mode=2), q must be int8, but got {q.dtype}"
            )
        if k_cache.dtype != torch.int8:
            raise ValueError(
                f"In INT8 mode (quant_mode=2), k_cache must be int8, but got {k_cache.dtype}"
            )
        if q_descale.dtype != torch.float or k_descale.dtype != torch.float:
            raise ValueError(
                f"In INT8 mode (quant_mode=2), q_descale/k_descale must be float32, "
                f"but got {q_descale.dtype}/{k_descale.dtype}"
            )

    # MLA 不支持 k_cache 的 stride 中含 0（仅 k_cache 支持 PA 非连续）
    for name, tensor in (("k_cache", k_cache),):
        if tensor is not None and 0 in tensor.stride():
            raise ValueError(
                f"quant_flash_mla does not support {name} with stride containing 0, "
                f"got {name}.stride() = {tensor.stride()}"
            )

    _check_mla_layout_pair(layout_q, layout_out)
    softmax_scale = _resolve_softmax_scale(q, softmax_scale)

    op_module = quant_flash_mla_op_builder.load()
    return op_module.quant_flash_mla(
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
        q_rope,
        k_rope,
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


quant_flash_mla.QuantMode = QuantMode
quant_flash_mla.MaskMode = MaskMode
quant_flash_mla_metadata.QuantMode = QuantMode
quant_flash_mla_metadata.MaskMode = MaskMode

quant_flash_mla = torch.ops.cann_ops_transformer.quant_flash_mla
quant_flash_mla_metadata = torch.ops.cann_ops_transformer.quant_flash_mla_metadata
