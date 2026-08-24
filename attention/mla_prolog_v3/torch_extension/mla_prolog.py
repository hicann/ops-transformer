# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


OP_NAME = "mla_prolog"
DEFAULT_CACHE_MODE = "PA_BSND"
DIM_2 = 2
DIM_3 = 3
MODE_1 = 1
MODE_2 = 2
MODE_3 = 3
MODE_4 = 4
MODE_5 = 5
FP8_E4M3_BLOCK_SIZE = 32

# 290 -> torch_npu.hifloat8（ACL_HIFLOAT8=34，hifloat8 以 uint8 存储时，须通过 *_dtype 显式指定）
HIFLOAT8_DTYPE_ENUM = 290

# shape 格式字段约束
SUPPORTED_HE = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192)
SUPPORTED_HCQ = (1536, 2048)
SUPPORTED_D = (128, 192)
HCKV = 512
DR = 64
NKV = 1
BLOCK_SIZE_MIN = 16
BLOCK_SIZE_MAX = 1024
DTILE_NON_PTPG = 512
DTILE_PTPG = 656
# N（Head-Num 多头数）支持 [1, 128] 之间的任意整型值
HEAD_NUM_MIN = 1
HEAD_NUM_MAX = 128
PA_CACHE_MODES = ("PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ")
SUPPORTED_CACHE_MODES = PA_CACHE_MODES + ("BSND", "TND")


def _has_defined(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and tensor.numel() > 0


def _resolve_do_rope(
    rope_sin: Optional[torch.Tensor], rope_cos: Optional[torch.Tensor]
) -> bool:
    has_sin = _has_defined(rope_sin)
    has_cos = _has_defined(rope_cos)
    if has_sin != has_cos:
        raise ValueError(
            "rope_sin and rope_cos must both be provided or both be empty/None"
        )
    return has_sin


def _resolve_rope_dim(
    token_x: torch.Tensor,
    rope_sin: Optional[torch.Tensor],
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
) -> int:
    if not _has_defined(rope_sin):
        # do_rope=false：Dr 由 weightDkvKr 的 (Hckv+Dr) 与 weightUk 的 Hckv 推导，
        if weight_dkv_kr.dim() == 4:
            hckv_plus_dr = weight_dkv_kr.size(0) * weight_dkv_kr.size(3)
        else:
            hckv_plus_dr = weight_dkv_kr.size(1)
        return hckv_plus_dr - weight_uk.size(2)
    if token_x.dim() == DIM_3:
        if rope_sin.dim() != DIM_3:
            raise ValueError("when token_x dim num is 3, rope_sin dim num should be 3")
        return rope_sin.size(2)
    if rope_sin.dim() != DIM_2:
        raise ValueError("when token_x dim num is 2, rope_sin dim num should be 2")
    return rope_sin.size(1)


def _resolve_weight_dims(tensor: torch.Tensor) -> Tuple[int, int]:
    """返回权重张量的逻辑行列数：2 维（ND）直接取 (s0, s1)；4 维（FRACTAL_NZ）取 (s0*16, s1*16)。"""
    if tensor.dim() == 4:
        return tensor.size(0) * 16, tensor.size(1) * 16
    return tensor.size(0), tensor.size(1)


def _require_fractal_nz(tensor: torch.Tensor, name: str) -> None:
    """算子对 weight_dq/weight_uq_qr/weight_dkv_kr 严格要求 FRACTAL_NZ 格式，此处提前校验。"""
    if tensor.device.type != "npu":
        return
    fmt = torch_npu.get_npu_format(tensor)
    # 兼容返回 int（ACL 格式枚举值）或 Format 枚举的不同 torch_npu 版本
    fmt_id = fmt.value if hasattr(fmt, "value") else int(fmt)
    if fmt_id != 29:  # 29 = ACL_FORMAT_FRACTAL_NZ
        raise ValueError(
            f"{name} must be in FRACTAL_NZ format "
            f"(use torch_npu.npu_format_cast(t, 29)), but got format {fmt}"
        )


def _validate_shape_constraints(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    rope_sin: Optional[torch.Tensor],
    rope_cos: Optional[torch.Tensor],
    cache_index: Optional[torch.Tensor],
    cache_mode: str,
    kv_cache_quant_mode: int,
) -> None:
    """校验接口的 shape 格式字段约束。"""
    if token_x.dim() not in (DIM_2, DIM_3):
        raise ValueError(
            f"token_x dim num should be 2 or 3, but the actual value is {token_x.dim()}"
        )
    he = token_x.size(-1)
    if type(he) is int and he not in SUPPORTED_HE:
        raise ValueError(
            f"head size He (token_x last dim) must be one of {SUPPORTED_HE}, but got {he}"
        )
    # B 约束：3 维时 B = size(0) 需在 [0, 65536] 内
    if (
        token_x.dim() == DIM_3
        and type(token_x.size(0)) is int
        and not (0 <= token_x.size(0) <= 65536)
    ):
        raise ValueError(
            f"batch B (token_x.size(0)) must be in [0, 65536], but got {token_x.size(0)}"
        )

    if weight_uk.dim() != DIM_3:
        raise ValueError(
            f"weight_uk dim num should be 3, but the actual value is {weight_uk.dim()}"
        )
    head_num, qk_dim, kv_lora_rank = (
        weight_uk.size(0),
        weight_uk.size(1),
        weight_uk.size(2),
    )
    if not (HEAD_NUM_MIN <= head_num <= HEAD_NUM_MAX):
        raise ValueError(
            f"head num N (weight_uk.size(0)) must be an integer in [{HEAD_NUM_MIN}, {HEAD_NUM_MAX}], "
            f"but got {head_num}"
        )
    if type(qk_dim) is int and qk_dim not in SUPPORTED_D:
        raise ValueError(
            f"qk dim D (weight_uk.size(1)) must be one of {SUPPORTED_D}, but got {qk_dim}"
        )
    if type(kv_lora_rank) is int and kv_lora_rank != HCKV:
        raise ValueError(
            f"kv lora rank Hckv (weight_uk.size(2)) must be {HCKV}, but got {kv_lora_rank}"
        )

    dq_rows, hcq = _resolve_weight_dims(weight_dq)
    _require_fractal_nz(weight_dq, "weight_dq")
    if type(dq_rows) is int and dq_rows != he:
        raise ValueError(f"weight_dq.size(0) ({dq_rows}) must equal He ({he})")
    if type(hcq) is int and hcq not in SUPPORTED_HCQ:
        raise ValueError(
            f"q lora rank Hcq (weight_dq.size(1)) must be one of {SUPPORTED_HCQ}, but got {hcq}"
        )

    rope_dim = _resolve_rope_dim(token_x, rope_sin, weight_uk, weight_dkv_kr)
    if type(rope_dim) is int and rope_dim != DR:
        raise ValueError(f"qk rope dim Dr must be {DR}, but got {rope_dim}")
    if _has_defined(rope_cos) and rope_cos.size(-1) != rope_dim:
        raise ValueError("rope_sin and rope_cos last dim must be equal")

    uq_rows, uq_cols = _resolve_weight_dims(weight_uq_qr)
    _require_fractal_nz(weight_uq_qr, "weight_uq_qr")
    if type(uq_rows) is int and uq_rows != hcq:
        raise ValueError(f"weight_uq_qr.size(0) ({uq_rows}) must equal Hcq ({hcq})")
    expected_uq_cols = head_num * (qk_dim + rope_dim)
    if (
        type(uq_cols) is int
        and type(expected_uq_cols) is int
        and uq_cols != expected_uq_cols
    ):
        raise ValueError(
            f"weight_uq_qr.size(1) ({uq_cols}) must equal N*(D+Dr) = {expected_uq_cols}"
        )

    dkv_rows, dkv_cols = _resolve_weight_dims(weight_dkv_kr)
    _require_fractal_nz(weight_dkv_kr, "weight_dkv_kr")
    if type(dkv_rows) is int and dkv_rows != he:
        raise ValueError(f"weight_dkv_kr.size(0) ({dkv_rows}) must equal He ({he})")
    expected_dkv_cols = kv_lora_rank + rope_dim
    if (
        type(dkv_cols) is int
        and type(expected_dkv_cols) is int
        and dkv_cols != expected_dkv_cols
    ):
        raise ValueError(
            f"weight_dkv_kr.size(1) ({dkv_cols}) must equal Hckv+Dr = {expected_dkv_cols}"
        )

    if rmsnorm_gamma_cq.dim() != 1 or (
        type(rmsnorm_gamma_cq.size(0)) is int and rmsnorm_gamma_cq.size(0) != hcq
    ):
        raise ValueError(
            f"rmsnorm_gamma_cq must be 1D with shape [{hcq}], but got {tuple(rmsnorm_gamma_cq.shape)}"
        )
    if rmsnorm_gamma_ckv.dim() != 1 or (
        type(rmsnorm_gamma_ckv.size(0)) is int
        and rmsnorm_gamma_ckv.size(0) != kv_lora_rank
    ):
        raise ValueError(
            f"rmsnorm_gamma_ckv must be 1D with shape [{kv_lora_rank}], "
            f"but got {tuple(rmsnorm_gamma_ckv.shape)}"
        )

    _validate_cache_shapes(
        kv_cache, kr_cache, cache_mode, kv_cache_quant_mode, rope_dim, cache_index
    )


def _validate_cache_shapes(
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    cache_mode: str,
    kv_cache_quant_mode: int,
    rope_dim: int,
    cache_index: Optional[torch.Tensor],
) -> None:
    if cache_mode not in SUPPORTED_CACHE_MODES:
        raise ValueError(
            f"cache_mode must be one of {SUPPORTED_CACHE_MODES}, but got {cache_mode}"
        )

    if cache_mode == "TND":
        if kv_cache.dim() != DIM_3 or kr_cache.dim() != DIM_3:
            raise ValueError("when cache_mode is TND, kv_cache/kr_cache must be 3D")
        if type(kv_cache.size(-2)) is int and kv_cache.size(-2) != NKV:
            raise ValueError(
                f"kv head num Nkv (kv_cache dim -2) must be {NKV}, but got {kv_cache.size(-2)}"
            )
        if type(kr_cache.size(-2)) is int and kr_cache.size(-2) != NKV:
            raise ValueError(
                f"kv head num Nkv (kr_cache dim -2) must be {NKV}, but got {kr_cache.size(-2)}"
            )
        if type(kr_cache.size(-1)) is int and kr_cache.size(-1) != rope_dim:
            raise ValueError(
                f"kr_cache last dim must equal Dr={rope_dim}, but got {kr_cache.size(-1)}"
            )
        _check_dtile(kv_cache, kv_cache_quant_mode)
        return

    if kv_cache.dim() != 4 or kr_cache.dim() != 4:
        raise ValueError(
            f"when cache_mode is {cache_mode}, kv_cache/kr_cache must be 4D"
        )
    if type(kv_cache.size(-2)) is int and kv_cache.size(-2) != NKV:
        raise ValueError(
            f"kv head num Nkv (kv_cache dim -2) must be {NKV}, but got {kv_cache.size(-2)}"
        )
    if type(kr_cache.size(-2)) is int and kr_cache.size(-2) != NKV:
        raise ValueError(
            f"kv head num Nkv (kr_cache dim -2) must be {NKV}, but got {kr_cache.size(-2)}"
        )
    if type(kr_cache.size(-1)) is int and kr_cache.size(-1) != rope_dim:
        raise ValueError(
            f"kr_cache last dim must equal Dr={rope_dim}, but got {kr_cache.size(-1)}"
        )
    _check_dtile(kv_cache, kv_cache_quant_mode)

    if cache_mode in PA_CACHE_MODES:
        if not _has_defined(cache_index):
            raise ValueError(
                f"when cache_mode is {cache_mode}, cache_index must be provided "
                f"(PagedAttention cache modes require a non-empty cache_index)"
            )
        block_size = kv_cache.size(1)
        if type(block_size) is int and not (
            BLOCK_SIZE_MIN <= block_size <= BLOCK_SIZE_MAX and block_size % 16 == 0
        ):
            raise ValueError(
                f"BlockSize (kv_cache dim 1) must be in [{BLOCK_SIZE_MIN}, {BLOCK_SIZE_MAX}] "
                f"and a multiple of 16, but got {block_size}"
            )
        if (
            type(kr_cache.size(1)) is int
            and type(block_size) is int
            and kr_cache.size(1) != block_size
        ):
            raise ValueError(
                f"kr_cache dim 1 must equal BlockSize ({block_size}), but got {kr_cache.size(1)}"
            )


def _check_dtile(kv_cache: torch.Tensor, kv_cache_quant_mode: int) -> None:
    dtile = kv_cache.size(-1)
    expected_dtile = DTILE_PTPG if kv_cache_quant_mode == MODE_3 else DTILE_NON_PTPG
    if type(dtile) is int and dtile != expected_dtile:
        raise ValueError(
            f"kv_cache last dim (Dtile) must be {expected_dtile} "
            f"(kv_cache_quant_mode={kv_cache_quant_mode}), but got {dtile}"
        )


def _is_full_quant_kv(weight_quant_mode: int, kv_cache_quant_mode: int) -> bool:
    return (
        weight_quant_mode in (MODE_2, MODE_3, MODE_4, MODE_5)
        and kv_cache_quant_mode == MODE_1
    )


def _is_hifloat8(weight_quant_mode: int, token_x: torch.Tensor) -> bool:
    return weight_quant_mode == MODE_5 and token_x.dtype == torch.uint8


def _query_shape(token_x: torch.Tensor, weight_uk: torch.Tensor) -> List[int]:
    if token_x.dim() == DIM_3:
        return [token_x.size(0), token_x.size(1), weight_uk.size(0), weight_uk.size(2)]
    return [token_x.size(0), weight_uk.size(0), weight_uk.size(2)]


def _meta_outputs(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    rope_sin: Optional[torch.Tensor],
    rope_cos: Optional[torch.Tensor],
    kr_cache: torch.Tensor,
    query_norm_flag: bool,
    weight_quant_mode: int,
    kv_cache_quant_mode: int,
    dequant_scale_x: Optional[torch.Tensor],
    token_x_dtype: Optional[int],
):
    _resolve_do_rope(rope_sin, rope_cos)
    is_hifloat8 = _is_hifloat8(weight_quant_mode, token_x) and token_x_dtype is not None
    full_quant_kv = _is_full_quant_kv(weight_quant_mode, kv_cache_quant_mode)
    rope_dim = _resolve_rope_dim(token_x, rope_sin, weight_uk, weight_dkv_kr)

    query_dtype = (
        torch.uint8
        if (is_hifloat8 and full_quant_kv)
        else token_x.dtype
        if full_quant_kv
        else torch.bfloat16
    )
    query = torch.empty(
        _query_shape(token_x, weight_uk), dtype=query_dtype, device="meta"
    )

    query_rope_shape = _query_shape(token_x, weight_uk)[:-1] + [rope_dim]
    query_rope = torch.empty(query_rope_shape, dtype=torch.bfloat16, device="meta")

    if full_quant_kv:
        dsn0 = (
            token_x.size(0) * token_x.size(1)
            if token_x.dim() == DIM_3
            else token_x.size(0)
        )
        dequant_scale_q_nope = torch.empty(
            [dsn0, weight_uk.size(0), 1], dtype=torch.float32, device="meta"
        )
    else:
        dequant_scale_q_nope = torch.empty([0], dtype=torch.float32, device="meta")

    if query_norm_flag:
        qn_dtype = torch.uint8 if is_hifloat8 else weight_uq_qr.dtype
        if token_x.dim() == DIM_3:
            query_norm = torch.empty(
                [token_x.size(0), token_x.size(1), weight_dq.size(1)],
                dtype=qn_dtype,
                device="meta",
            )
        else:
            query_norm = torch.empty(
                [token_x.size(0), weight_dq.size(1)], dtype=qn_dtype, device="meta"
            )
    else:
        query_norm = torch.empty([0], dtype=torch.bfloat16, device="meta")

    if query_norm_flag and weight_quant_mode != 0:
        dsn0 = (
            token_x.size(0) * token_x.size(1)
            if token_x.dim() == DIM_3
            else token_x.size(0)
        )
        if weight_quant_mode == MODE_3:
            dtype = dequant_scale_x.dtype
            dim1 = weight_dq.size(1) // FP8_E4M3_BLOCK_SIZE
        else:
            dtype = torch.float32
            dim1 = 1
        dequant_scale_q_norm = torch.empty([dsn0, dim1], dtype=dtype, device="meta")
    else:
        dequant_scale_q_norm = torch.empty([0], dtype=torch.float32, device="meta")

    return query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm


class MlaPrologOpBuilder(OpBuilder):
    def __init__(self):
        super(MlaPrologOpBuilder, self).__init__(OP_NAME, category="attention")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/mla_prolog.cpp"]

    def schema(self) -> List[str]:
        """PyTorch operator signatures."""
        common_kw = (
            "*, Tensor? rope_sin=None, Tensor? rope_cos=None, "
            "Tensor? cache_index=None, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, "
            "Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, "
            "Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, Tensor? actual_seq_len=None, "
            "Tensor? k_nope_clip_alpha=None, "
            "float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, "
            'str cache_mode="PA_BSND", '
            "bool query_norm_flag=False, int weight_quant_mode=0, int kv_cache_quant_mode=0, int query_quant_mode=0, "
            "int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128, "
            "float qc_qr_scale=1.0, float kc_scale=1.0, "
            "int? token_x_dtype=None, int? weight_dq_dtype=None, int? weight_uq_qr_dtype=None, "
            "int? weight_dkv_kr_dtype=None, int? kv_cache_dtype=None"
        )
        inputs = (
            "Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, "
            "Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, "
            "Tensor(a!) kv_cache, Tensor(b!) kr_cache, "
        )
        return [
            OP_NAME
            + "("
            + inputs
            + common_kw
            + ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """Registers the Meta implementation (Shape/Dtype inference)."""

        @impl(get_as_library(), OP_NAME, "Meta")
        def mla_prolog_meta(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_uk,
            weight_dkv_kr,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            kv_cache,
            kr_cache,
            *,
            rope_sin=None,
            rope_cos=None,
            cache_index=None,
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=None,
            dequant_scale_w_dkv_kr=None,
            quant_scale_ckv=None,
            quant_scale_ckr=None,
            smooth_scales_cq=None,
            actual_seq_len=None,
            k_nope_clip_alpha=None,
            rmsnorm_epsilon_cq=1e-05,
            rmsnorm_epsilon_ckv=1e-05,
            cache_mode=DEFAULT_CACHE_MODE,
            query_norm_flag=False,
            weight_quant_mode=0,
            kv_cache_quant_mode=0,
            query_quant_mode=0,
            ckvkr_repo_mode=0,
            quant_scale_repo_mode=0,
            tile_size=128,
            qc_qr_scale=1.0,
            kc_scale=1.0,
            token_x_dtype=None,
            weight_dq_dtype=None,
            weight_uq_qr_dtype=None,
            weight_dkv_kr_dtype=None,
            kv_cache_dtype=None,
        ):
            if token_x.dim() not in (DIM_2, DIM_3):
                raise ValueError("token_x dim num should be 2 or 3")
            if weight_uk.dim() != DIM_3:
                raise ValueError("weight_uk dim num should be 3")
            _validate_shape_constraints(
                token_x,
                weight_dq,
                weight_uq_qr,
                weight_uk,
                weight_dkv_kr,
                rmsnorm_gamma_cq,
                rmsnorm_gamma_ckv,
                kv_cache,
                kr_cache,
                rope_sin,
                rope_cos,
                cache_index,
                cache_mode,
                kv_cache_quant_mode,
            )
            return _meta_outputs(
                token_x,
                weight_dq,
                weight_uq_qr,
                weight_uk,
                rope_sin,
                rope_cos,
                kr_cache,
                query_norm_flag,
                weight_quant_mode,
                kv_cache_quant_mode,
                dequant_scale_x,
                token_x_dtype,
            )


mla_prolog_op_builder = MlaPrologOpBuilder()
mla_prolog_op_builder._ensure_initialized()


@impl(get_as_library(), OP_NAME, "PrivateUse1")
def mla_prolog(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    *,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-05,
    rmsnorm_epsilon_ckv: float = 1e-05,
    cache_mode: str = DEFAULT_CACHE_MODE,
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLA Prolog 前向计算（原地更新 kv_cache/kr_cache），封装 aclnnMlaPrologV4WeightNz。

    RoPE 开关由 rope_sin/rope_cos 是否成对传入决定：二者同时为空（或 None）视为关闭，
    二者同时非空视为开启，一空一非空视为非法输入。

    Args:
        token_x (Tensor): 输入 token 特征，shape 支持 [T, He]（BS 合轴）或 [B, S, He]（BS 非合轴），dtype 支持
            float16/bfloat16（weight_quant_mode=5 的 hifloat8 场景为 torch.uint8 + token_x_dtype）。
        weight_dq (Tensor): Dq 权重，shape [He, Hcq]，dtype 与 token_x 一致（hifloat8 场景为 torch.uint8），
            必须为 FRACTAL_NZ 格式（npu_format_cast(t, 29)）。
        weight_uq_qr (Tensor): Uq/Qr 拼接权重，shape [Hcq, N*(D+Dr)]，weight_quant_mode=0 时 dtype 与 token_x 一致，
            mode=1 时为 int8，必须为 FRACTAL_NZ 格式。
        weight_uk (Tensor): Uk 权重，3 维 [N, D, Hckv]，dtype 为 bfloat16。
        weight_dkv_kr (Tensor): Dkv/Kr 拼接权重，shape [He, Hckv+Dr]，dtype 与 token_x 一致，
            必须为 FRACTAL_NZ 格式。
        rmsnorm_gamma_cq (Tensor): Cq 的 RMSNorm gamma，shape [Hcq]。
        rmsnorm_gamma_ckv (Tensor): Ckv 的 RMSNorm gamma，shape [Hckv]。
        kv_cache (Tensor(a!)): KV cache，原地更新。
        kr_cache (Tensor(b!)): Kr cache，原地更新。
        rope_sin (Tensor, optional): RoPE sin。与 rope_cos 同时为空（或 None）表示不执行 RoPE。
        rope_cos (Tensor, optional): RoPE cos，约束同 rope_sin。
        cache_index (Tensor, optional): cache 索引，默认 None。
        dequant_scale_x (Tensor, optional): X 反量化 scale，weight_quant_mode=3（float8_e8m0fnu）或 mode=5（float32）时必传。
        dequant_scale_w_dq (Tensor, optional): Dq 权重反量化 scale。
        dequant_scale_w_uq_qr (Tensor, optional): Uq/Qr 权重反量化 scale。
        dequant_scale_w_dkv_kr (Tensor, optional): Dkv/Kr 权重反量化 scale。
        quant_scale_ckv (Tensor, optional): Ckv 量化 scale。
        quant_scale_ckr (Tensor, optional): Ckr 量化 scale。
        smooth_scales_cq (Tensor, optional): Cq smooth scale。
        actual_seq_len (Tensor, optional): 实际序列长度，默认 None。
        k_nope_clip_alpha (Tensor, optional): Nope 裁剪 alpha，默认 None。
        rmsnorm_epsilon_cq (float): Cq RMSNorm epsilon，默认 1e-05。
        rmsnorm_epsilon_ckv (float): Ckv RMSNorm epsilon，默认 1e-05。
        cache_mode (str): cache 布局，默认 "PA_BSND"。
        query_norm_flag (bool): 是否计算 query_norm 输出，默认 False。
        weight_quant_mode (int): 权重量化模式，0-5，默认 0。
        kv_cache_quant_mode (int): KV cache 量化模式，0-3，默认 0。
        query_quant_mode (int): Query 量化模式，0/1，默认 0。
        ckvkr_repo_mode (int): Ckv/Kr 存储模式，默认 0。
        quant_scale_repo_mode (int): 量化 scale 存储模式，默认 0。
        tile_size (int): tile 大小，默认 128。
        qc_qr_scale (float): Qc/Qr scale，默认 1.0。
        kc_scale (float): Kc scale，默认 1.0。
        token_x_dtype (int, optional): hifloat8 场景 token_x 的 ACL dtype 枚举（290 = torch_npu.hifloat8）。
        weight_dq_dtype (int, optional): hifloat8 场景 weight_dq 的 ACL dtype 枚举。
        weight_uq_qr_dtype (int, optional): hifloat8 场景 weight_uq_qr 的 ACL dtype 枚举。
        weight_dkv_kr_dtype (int, optional): hifloat8 场景 weight_dkv_kr 的 ACL dtype 枚举。
        kv_cache_dtype (int, optional): hifloat8 场景 kv_cache 的 ACL dtype 枚举。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            query、query_rope、dequant_scale_q_nope、query_norm、dequant_scale_q_norm。
    """
    _resolve_do_rope(rope_sin, rope_cos)
    _validate_shape_constraints(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        kv_cache,
        kr_cache,
        rope_sin,
        rope_cos,
        cache_index,
        cache_mode,
        kv_cache_quant_mode,
    )
    op_module = mla_prolog_op_builder.load()
    return op_module.mla_prolog(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        kv_cache,
        kr_cache,
        rope_sin,
        rope_cos,
        cache_index,
        dequant_scale_x,
        dequant_scale_w_dq,
        dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr,
        quant_scale_ckv,
        quant_scale_ckr,
        smooth_scales_cq,
        actual_seq_len,
        k_nope_clip_alpha,
        rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv,
        cache_mode,
        query_norm_flag,
        weight_quant_mode,
        kv_cache_quant_mode,
        query_quant_mode,
        ckvkr_repo_mode,
        quant_scale_repo_mode,
        tile_size,
        qc_qr_scale,
        kc_scale,
        token_x_dtype,
        weight_dq_dtype,
        weight_uq_qr_dtype,
        weight_dkv_kr_dtype,
        kv_cache_dtype,
    )
