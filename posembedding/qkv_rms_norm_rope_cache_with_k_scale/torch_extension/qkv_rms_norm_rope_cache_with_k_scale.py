# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale"
INPLACE_OP_NAME = "qkv_rms_norm_rope_cache_with_k_scale_"
QKV_LAYOUT_TND = "TND"
QKV_LAYOUT_NTD = "NTD"
DEFAULT_Q_OUT_DTYPE = torch.float8_e4m3fn
Q_QUANT_PER_TOKEN_PER_HEAD = "PerTokenPerHead"
Q_QUANT_NO_QUANT = "NoQuant"
Q_QUANT_MX = "Mx"
K_QUANT_PER_TOKEN_PER_HEAD = "PerTokenPerHead"
K_QUANT_MX = "Mx"


class _MetaScene(NamedTuple):
    query_start_loc: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor]
    rotation: Optional[torch.Tensor]
    mrope_position: Optional[torch.Tensor]
    mrope_section: Optional[List[int]]
    layout_qkv: Optional[str]
    layout_q_out: Optional[str]
    q_quant_mode: str
    k_quant_mode: str


_EXTENSION_ARGUMENT_NAMES = (
    "qkv",
    "q_gamma",
    "k_gamma",
    "cos_sin",
    "slot_mapping",
    "k_cache",
    "v_cache",
    "k_scale_cache",
    "query_start_loc",
    "seq_lens",
    "head_nums",
    "layout_qkv",
    "layout_q_out",
    "rotation",
    "v_scale",
    "epsilon",
    "mrope_position",
    "mrope_section",
    "q_quant_mode",
    "k_quant_mode",
    "q_out_dtype",
)


def _meta_scene_from_locals(local_values):
    return _MetaScene(*(local_values[name] for name in _MetaScene._fields))


def _call_extension(op, local_values):
    return op(*(local_values[name] for name in _EXTENSION_ARGUMENT_NAMES))


class QkvRmsNormRopeCacheWithKScaleOpBuilder(OpBuilder):
    def __init__(self):
        super(QkvRmsNormRopeCacheWithKScaleOpBuilder, self).__init__(OP_NAME)

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/posembedding/qkv_rms_norm_rope_cache_with_k_scale.cpp"]

    def schema(self) -> List[str]:
        """PyTorch operator signatures."""
        return [
            "qkv_rms_norm_rope_cache_with_k_scale_("
            "Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, "
            "Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache, "
            "Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, "
            "*, "
            "Tensor? rotation=None, Tensor? v_scale=None, "
            'str? layout_qkv="TND", str? layout_q_out="NTD", '
            "float epsilon=0.000001, Tensor? mrope_position=None, int[]? mrope_section=None, "
            'str q_quant_mode="PerTokenPerHead", str k_quant_mode="PerTokenPerHead", '
            "ScalarType? q_out_dtype=None) -> (Tensor, Tensor?)",
            "qkv_rms_norm_rope_cache_with_k_scale("
            "Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping, "
            "Tensor k_cache, Tensor v_cache, Tensor k_scale_cache, "
            "Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, "
            "*, "
            "Tensor? rotation=None, Tensor? v_scale=None, "
            'str? layout_qkv="TND", str? layout_q_out="NTD", '
            "float epsilon=0.000001, Tensor? mrope_position=None, int[]? mrope_section=None, "
            'str q_quant_mode="PerTokenPerHead", str k_quant_mode="PerTokenPerHead", '
            "ScalarType? q_out_dtype=None) -> (Tensor, Tensor?, Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        def make_meta_impl(functional):
            def qkv_rms_norm_rope_cache_with_k_scale_meta(
                qkv,
                q_gamma,
                k_gamma,
                cos_sin,
                slot_mapping,
                k_cache,
                v_cache,
                k_scale_cache,
                query_start_loc,
                seq_lens,
                head_nums,
                *,
                rotation=None,
                v_scale=None,
                layout_qkv=QKV_LAYOUT_TND,
                layout_q_out=QKV_LAYOUT_NTD,
                epsilon=1e-6,
                mrope_position=None,
                mrope_section=None,
                q_quant_mode=Q_QUANT_PER_TOKEN_PER_HEAD,
                k_quant_mode=K_QUANT_PER_TOKEN_PER_HEAD,
                q_out_dtype=DEFAULT_Q_OUT_DTYPE,
            ):
                q_out, q_scale = _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
                    qkv,
                    head_nums,
                    _meta_scene_from_locals(locals()),
                    q_out_dtype,
                )
                if not functional:
                    return q_out, q_scale
                return (
                    q_out,
                    q_scale,
                    torch.empty_like(k_cache),
                    torch.empty_like(v_cache),
                    torch.empty_like(k_scale_cache),
                )

            return qkv_rms_norm_rope_cache_with_k_scale_meta

        impl(get_as_library(), INPLACE_OP_NAME, "Meta")(make_meta_impl(False))
        impl(get_as_library(), OP_NAME, "Meta")(make_meta_impl(True))


def _get_q_head_num_for_meta(head_nums):
    if head_nums is None:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: head_nums must not be None"
        )
    if len(head_nums) < 1:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: head_nums must contain q head num"
        )

    n_q = head_nums[0]
    return n_q


def _normalize_layout_for_meta(layout, default_layout, attr_name):
    layout = default_layout if layout is None or layout == "" else layout
    if layout not in (QKV_LAYOUT_TND, QKV_LAYOUT_NTD):
        raise RuntimeError(
            f"qkv_rms_norm_rope_cache_with_k_scale: {attr_name} must be TND or NTD"
        )
    return layout


def _normalize_meta_layouts(layout_qkv, layout_q_out):
    return (
        _normalize_layout_for_meta(layout_qkv, QKV_LAYOUT_TND, "layout_qkv"),
        _normalize_layout_for_meta(layout_q_out, QKV_LAYOUT_NTD, "layout_q_out"),
    )


def _normalize_q_out_dtype(q_out_dtype):
    if q_out_dtype is None:
        return DEFAULT_Q_OUT_DTYPE
    if not isinstance(q_out_dtype, torch.dtype):
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: q_out_dtype must be a torch.dtype"
        )
    return q_out_dtype


def _validate_q_out_dtype_for_scene(q_quant_mode, q_out_dtype):
    expected_dtype = (
        torch.bfloat16 if q_quant_mode == Q_QUANT_NO_QUANT else torch.float8_e4m3fn
    )
    if q_out_dtype != expected_dtype:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: q_out_dtype does not match "
            f"the resolved scene; expected {expected_dtype}, got {q_out_dtype}"
        )


def _resolve_scene_for_meta(scene):
    q_quant_mode = scene.q_quant_mode or Q_QUANT_PER_TOKEN_PER_HEAD
    k_quant_mode = scene.k_quant_mode or K_QUANT_PER_TOKEN_PER_HEAD
    has_mrope_position = scene.mrope_position is not None
    if has_mrope_position != bool(scene.mrope_section):
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: mrope_position and "
            "non-empty mrope_section must be provided together"
        )

    has_rope_inputs = all(
        value is not None
        for value in (scene.query_start_loc, scene.seq_lens, scene.rotation)
    )
    has_rope_quant_modes = (q_quant_mode, k_quant_mode) == (
        Q_QUANT_PER_TOKEN_PER_HEAD,
        K_QUANT_PER_TOKEN_PER_HEAD,
    )
    if not has_mrope_position and has_rope_inputs and has_rope_quant_modes:
        return q_quant_mode

    is_mrope_layout = scene.layout_qkv == scene.layout_q_out == QKV_LAYOUT_TND
    is_mrope = (
        has_mrope_position
        and scene.query_start_loc is None
        and scene.seq_lens is None
        and scene.rotation is not None
        and is_mrope_layout
        and q_quant_mode == Q_QUANT_NO_QUANT
        and k_quant_mode == K_QUANT_PER_TOKEN_PER_HEAD
    )
    if is_mrope:
        return q_quant_mode

    is_mrope_mx = (
        has_mrope_position
        and scene.query_start_loc is None
        and scene.seq_lens is None
        and scene.rotation is None
        and is_mrope_layout
        and q_quant_mode == Q_QUANT_MX
        and k_quant_mode == K_QUANT_MX
    )
    if is_mrope_mx:
        return q_quant_mode

    raise RuntimeError(
        "qkv_rms_norm_rope_cache_with_k_scale: unsupported scene; expected "
        "RoPE(positions present, rotation present, PerTokenPerHead/PerTokenPerHead), "
        "M-RoPE(TND/TND, M-RoPE position+section present, rotation present, "
        "NoQuant/PerTokenPerHead), or M-RoPE MX(TND/TND, M-RoPE position+section "
        "present, rotation absent, Mx/Mx)"
    )


def _get_qkv_shape_for_meta(qkv, layout_qkv):
    if qkv.dim() < 3:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: qkv must be at least 3D"
        )

    is_ntd = layout_qkv == QKV_LAYOUT_NTD
    token_axis = 1 if is_ntd else 0
    token_num = qkv.size(token_axis)
    head_size = qkv.size(2)
    return token_num, head_size


def _make_meta_output_tensors(
    n_q, token_num, head_size, layout_q_out, q_quant_mode, q_out_dtype
):
    if layout_q_out == QKV_LAYOUT_NTD:
        q_out_shape = (n_q, token_num, head_size)
        q_scale_shape = (n_q, token_num)
    else:
        q_out_shape = (token_num, n_q, head_size)
        q_scale_shape = (token_num, n_q)

    if q_quant_mode == Q_QUANT_NO_QUANT:
        q_scale = None
    elif q_quant_mode == Q_QUANT_MX:
        q_scale = torch.empty(
            (token_num, n_q, (head_size + 31) // 32),
            dtype=torch.float8_e8m0fnu,
            device="meta",
        )
    elif q_quant_mode == Q_QUANT_PER_TOKEN_PER_HEAD:
        q_scale = torch.empty(q_scale_shape, dtype=torch.float32, device="meta")
    else:
        raise RuntimeError(
            "qkv_rms_norm_rope_cache_with_k_scale: q_quant_mode must be "
            "PerTokenPerHead, NoQuant or Mx"
        )
    q_out = torch.empty(q_out_shape, dtype=q_out_dtype, device="meta")
    return q_out, q_scale


def _qkv_rms_norm_rope_cache_with_k_scale_meta_outputs(
    qkv,
    head_nums,
    scene,
    q_out_dtype,
):
    n_q = _get_q_head_num_for_meta(head_nums)
    q_out_dtype = _normalize_q_out_dtype(q_out_dtype)
    layout_qkv, layout_q_out = _normalize_meta_layouts(
        scene.layout_qkv, scene.layout_q_out
    )
    scene = scene._replace(
        layout_qkv=layout_qkv,
        layout_q_out=layout_q_out,
    )
    q_quant_mode = _resolve_scene_for_meta(scene)
    _validate_q_out_dtype_for_scene(q_quant_mode, q_out_dtype)
    token_num, head_size = _get_qkv_shape_for_meta(qkv, layout_qkv)
    return _make_meta_output_tensors(
        n_q, token_num, head_size, layout_q_out, q_quant_mode, q_out_dtype
    )


qkv_rms_norm_rope_cache_with_k_scale_op_builder = (
    QkvRmsNormRopeCacheWithKScaleOpBuilder()
)
qkv_rms_norm_rope_cache_with_k_scale_op_builder._ensure_initialized()


@impl(get_as_library(), INPLACE_OP_NAME, "PrivateUse1")
def qkv_rms_norm_rope_cache_with_k_scale_(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    head_nums: List[int],
    *,
    rotation: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    layout_qkv: Optional[str] = QKV_LAYOUT_TND,
    layout_q_out: Optional[str] = QKV_LAYOUT_NTD,
    epsilon: float = 1e-6,
    mrope_position: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    q_quant_mode: str = Q_QUANT_PER_TOKEN_PER_HEAD,
    k_quant_mode: str = K_QUANT_PER_TOKEN_PER_HEAD,
    q_out_dtype: Optional[torch.dtype] = DEFAULT_Q_OUT_DTYPE,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Run Q/K/V RMSNorm, RoPE, scene-specific quantization, and in-place KV cache update.

    In both M-RoPE scenes, ``mrope_position`` is an INT32 tensor with logical
    shape ``[T, 3]``: each row is one token and columns are ordered T/H/W.
    M-RoPE MX additionally requires ``q_quant_mode=k_quant_mode="Mx"``,
    ``rotation=None``, TND input/output layouts, and FP8 E4M3FN ``q_out_dtype``.
    """
    q_out_dtype = _normalize_q_out_dtype(q_out_dtype)
    op_module = qkv_rms_norm_rope_cache_with_k_scale_op_builder.load()
    return _call_extension(op_module.qkv_rms_norm_rope_cache_with_k_scale_, locals())


@impl(get_as_library(), OP_NAME, "PrivateUse1")
def qkv_rms_norm_rope_cache_with_k_scale(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    head_nums: List[int],
    *,
    rotation: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    layout_qkv: Optional[str] = QKV_LAYOUT_TND,
    layout_q_out: Optional[str] = QKV_LAYOUT_NTD,
    epsilon: float = 1e-6,
    mrope_position: Optional[torch.Tensor] = None,
    mrope_section: Optional[List[int]] = None,
    q_quant_mode: str = Q_QUANT_PER_TOKEN_PER_HEAD,
    k_quant_mode: str = K_QUANT_PER_TOKEN_PER_HEAD,
    q_out_dtype: Optional[torch.dtype] = DEFAULT_Q_OUT_DTYPE,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Functional variant returning cloned cache outputs instead of mutating caller-visible caches.

    In both M-RoPE scenes, ``mrope_position`` uses logical shape ``[T, 3]`` with
    T/H/W positions in columns 0/1/2 respectively. M-RoPE MX additionally
    requires ``q_quant_mode=k_quant_mode="Mx"``, ``rotation=None``, TND
    input/output layouts, and FP8 E4M3FN ``q_out_dtype``.
    """
    q_out_dtype = _normalize_q_out_dtype(q_out_dtype)
    op_module = qkv_rms_norm_rope_cache_with_k_scale_op_builder.load()
    return _call_extension(op_module.qkv_rms_norm_rope_cache_with_k_scale, locals())
