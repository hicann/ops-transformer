#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_ROOT = os.path.dirname(_ASSETS_DIR)
if _ASSETS_ROOT not in sys.path:
    sys.path.insert(0, _ASSETS_ROOT)
import quant_flash_attn_mxfp4_golden as golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(attrs):
    golden_mod._apply_golden_globals(attrs)
    golden_mod._apply_kwargs_globals(attrs)


def _is_numpy_arr(x):
    return isinstance(x, numpy.ndarray)


def _inplace_write(dst, src, slot_name):
    if dst is None:
        if src is not None and (not torch.is_tensor(src) or src.numel() > 0):
            raise ValueError(f"[INPUTS] {slot_name}: dst is None but src is not empty")
        return
    if src is None:
        if _is_numpy_arr(dst):
            dst[...] = 0
        else:
            dst.zero_()
        return
    src_t = src if torch.is_tensor(src) else torch.as_tensor(src)
    if tuple(dst.shape) != tuple(src_t.shape):
        raise ValueError(
            f"[INPUTS] {slot_name} shape mismatch: slot {tuple(dst.shape)} "
            f"!= computed {tuple(src_t.shape)}"
        )
    if _is_numpy_arr(dst):
        src_np = src_t.detach().cpu().contiguous().numpy()
        dst_str = str(dst.dtype)
        if "float4" in dst_str or "float8" in dst_str:
            src_np = src_np.view(dst.dtype)
        else:
            src_np = src_np.astype(dst.dtype)
        dst[...] = src_np
    else:
        dst_str = str(dst.dtype)
        if "float4" in dst_str or "float8" in dst_str:
            dst.copy_(src_t.view(dst.dtype))
        else:
            dst.copy_(src_t.to(dst.dtype))


def _write_int32_list(slot, values, slot_name):
    if slot is None:
        return
    if _is_numpy_arr(slot):
        if values:
            arr = numpy.array(list(values), dtype=slot.dtype)
            if tuple(slot.shape) != tuple(arr.shape):
                raise ValueError(
                    f"[INPUTS] {slot_name} shape mismatch: slot {tuple(slot.shape)} "
                    f"!= computed {tuple(arr.shape)}"
                )
            slot[...] = arr
        else:
            slot[...] = 0
        return
    if values:
        src = torch.tensor(list(values), dtype=torch.int32)
        if tuple(slot.shape) != tuple(src.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: slot {tuple(slot.shape)} "
                f"!= computed {tuple(src.shape)}"
            )
        slot.copy_(src.to(slot.dtype))
    else:
        slot.zero_()


def generate_qfa_mxfp4_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor = None,
    p_scale: torch.Tensor = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    seqused_kv: torch.Tensor = None,
    sinks: torch.Tensor = None,
    attn_mask: torch.Tensor = None,
    metadata: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    attrs = dict(kwargs)
    attrs.setdefault("quant_mode", quant_mode)
    attrs.setdefault("softmax_scale", softmax_scale)
    attrs.setdefault("mask_mode", mask_mode)
    attrs.setdefault("win_left", win_left)
    attrs.setdefault("win_right", win_right)
    attrs.setdefault("max_seqlen_q", max_seqlen_q)
    attrs.setdefault("max_seqlen_kv", max_seqlen_kv)
    attrs.setdefault("layout_q", layout_q)
    attrs.setdefault("layout_q_descale", layout_q_descale)
    attrs.setdefault("layout_kv", layout_kv)
    attrs.setdefault("layout_out", layout_out)
    attrs.setdefault("return_softmax_lse", return_softmax_lse)
    _apply_golden_globals(attrs)

    golden_mod._inject_physical_s_override(q, v, layout_q, layout_kv)
    try:
        data_dict = golden_mod.generate_data()
    finally:
        golden_mod._clear_physical_s_override()

    _inplace_write(q, data_dict["q"], "q (slot 0)")
    _inplace_write(k, data_dict["k"], "k (slot 1)")
    _inplace_write(v, data_dict["v"], "v (slot 2)")
    _inplace_write(q_descale, data_dict["q_descale"], "q_descale (slot 3)")
    _inplace_write(k_descale, data_dict["k_descale"], "k_descale (slot 4)")
    _inplace_write(v_descale, data_dict["v_descale"], "v_descale (slot 5)")
    _inplace_write(block_table, data_dict.get("block_table"), "block_table (slot 6)")
    _inplace_write(p_scale, data_dict.get("p_scale"), "p_scale (slot 7)")
    _write_int32_list(
        cu_seqlens_q, data_dict.get("cu_seqlens_q"), "cu_seqlens_q (slot 8)"
    )
    _write_int32_list(
        cu_seqlens_kv, data_dict.get("cu_seqlens_kv"), "cu_seqlens_kv (slot 9)"
    )
    _write_int32_list(seqused_q, data_dict.get("act_seq_lens_q"), "seqused_q (slot 10)")
    _write_int32_list(
        seqused_kv, data_dict.get("act_seq_lens_kv"), "seqused_kv (slot 11)"
    )
    _inplace_write(sinks, data_dict.get("sinks"), "sinks (slot 12)")
    _inplace_write(attn_mask, data_dict.get("attn_mask"), "attn_mask (slot 13)")

    logger.info(
        "[INPUTS] wrote MXFP4 q/k/v (q=%s), descale (dq=%s, dk=%s, dv=%s)",
        tuple(q.shape),
        tuple(q_descale.shape),
        tuple(k_descale.shape),
        tuple(v_descale.shape),
    )
