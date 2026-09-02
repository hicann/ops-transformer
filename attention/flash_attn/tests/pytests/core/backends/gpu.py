# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GPU 后端 — 三路径架构：paged / varlen / fixed。所有 layout 转换集中于此。"""

import torch
from typing import Any, Dict

from .base import Backend

try:
    from core.backends.gpu_impl import (
        gpu_fixed_attn,
        gpu_varlen_attn,
        gpu_paged_attn,
        FLASH_ATTN_AVAILABLE,
    )

    _HAS_GPU = FLASH_ATTN_AVAILABLE
except ImportError:
    _HAS_GPU = False


class GPUBackend(Backend):
    name = "gpu"

    def __init__(self, device_id: int = 0):
        self._device = torch.device(f"cuda:{device_id}")
        if _HAS_GPU:
            torch.cuda.set_device(device_id)

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_GPU and torch.cuda.is_available()

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute(
        self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)
        block_size = params.get("block_size", 256)
        mask_mode = params.get("mask_mode", 0)
        d = params.get("D", inputs["q"].shape[-1])
        softmax_scale = params.get("scale", d ** (-0.5))
        causal = int(mask_mode) == 3

        use_paged = (
            layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ") and block_size % 256 == 0
        )
        has_seqused = (
            params.get("seqused_kv") is not None
            or params.get("seqused_q") is not None
            or params.get("cu_seqlens_kv") is not None
        )
        use_varlen = (layout_q == "TND") or (has_seqused and not use_paged)

        if use_paged:
            out = self._compute_paged(inputs, params, softmax_scale, causal)
        elif use_varlen:
            out = self._compute_varlen(inputs, params, softmax_scale, causal)
        else:
            out = self._compute_fixed(inputs, params, softmax_scale, causal)

        torch.cuda.synchronize()
        return {"out": out}

    # ─── 路径 A: flash_attn_with_kvcache ──────────────────────────────────────

    def _compute_paged(self, inputs, params, softmax_scale, causal):
        layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)
        block_table = params.get("block_table")

        if isinstance(block_table, list):
            if isinstance(block_table[0], list):
                bt_tensor = torch.tensor(
                    block_table, dtype=torch.int32, device=self._device
                )
            else:
                bt_tensor = torch.tensor(
                    [block_table], dtype=torch.int32, device=self._device
                )
        else:
            bt_tensor = block_table.to(device=self._device, dtype=torch.int32)

        q = inputs["q"]
        k = inputs["k"]
        v = inputs["v"]

        if layout_kv == "PA_BBND":
            k_cache = k.contiguous().to(self._device)
            v_cache = v.contiguous().to(self._device)
        elif layout_kv == "PA_BNBD":
            k_cache = k.permute(0, 2, 1, 3).contiguous().to(self._device)
            v_cache = v.permute(0, 2, 1, 3).contiguous().to(self._device)
        elif layout_kv == "PA_NZ":
            k_cache = _pa_nz_to_bbnd(k, params).contiguous().to(self._device)
            v_cache = _pa_nz_to_bbnd(v, params).contiguous().to(self._device)
        else:
            raise ValueError(f"_compute_paged 不支持 layout_kv={layout_kv}")

        if layout_q == "TND":
            cu_seqlens_q = params.get("cu_seqlens_q")
            q = _unpack_tnd_to_bnsd(q, cu_seqlens_q).to(self._device)
            q_bshd = q.permute(0, 2, 1, 3).contiguous()
        elif layout_q == "BSND":
            q_bshd = q.to(self._device).contiguous()
        else:
            q = q.to(self._device) if q.device != self._device else q
            q_bshd = q.permute(0, 2, 1, 3).contiguous()

        seqused_kv = params.get("seqused_kv") or params.get("actual_seq_kvlen")
        B = q_bshd.shape[0]
        if seqused_kv is None:
            S_kv = k.shape[2] if layout_kv == "PA_BBND" else k.shape[1]
            cache_seqlens = torch.full(
                (B,), S_kv, dtype=torch.int32, device=self._device
            )
        elif isinstance(seqused_kv, (list, tuple)):
            cache_seqlens = torch.tensor(
                list(seqused_kv), dtype=torch.int32, device=self._device
            )
        else:
            cache_seqlens = seqused_kv.to(device=self._device, dtype=torch.int32)

        out_bshd = gpu_paged_attn(
            q_bshd, k_cache, v_cache, bt_tensor, cache_seqlens, softmax_scale, causal
        )
        layout_out = params.get("layout_out", layout_q)
        if layout_out == "BSND":
            return out_bshd.contiguous()
        elif layout_out == "TND":
            cu_seqlens_q = params.get("cu_seqlens_q")
            return _pack_bnsd_to_tnd(
                out_bshd.permute(0, 2, 1, 3), cu_seqlens_q
            ).contiguous()
        else:
            return out_bshd.permute(0, 2, 1, 3).contiguous()

    # ─── 路径 B: flash_attn_varlen_func ───────────────────────────────────────

    def _compute_varlen(self, inputs, params, softmax_scale, causal):
        layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        q = inputs["q"].to(self._device)
        if q.dim() == 3:
            q_tnd = q.contiguous()
        elif layout_q == "TND":
            q_tnd = q.squeeze(0).permute(1, 0, 2).contiguous()
        else:
            q_bnsd = q if layout_q == "BNSD" else q.permute(0, 2, 1, 3)
            q_tnd, cu_q_from_pack = self._bnsd_to_tnd_pack_q(q_bnsd, params)
            q_tnd = q_tnd.contiguous()
            if params.get("cu_seqlens_q") is None:
                params = dict(params)
                params["cu_seqlens_q"] = cu_q_from_pack

        if layout_kv.startswith("PA_"):
            k_bnsd, v_bnsd = self._pa_to_bnsd_pair(inputs, layout_kv, params)
            k_tnd, v_tnd, cu_kv = self._bnsd_to_tnd_packed(k_bnsd, v_bnsd, params)
            k_tnd = k_tnd.to(self._device)
            v_tnd = v_tnd.to(self._device)
        else:
            k = inputs["k"].to(self._device)
            v = inputs["v"].to(self._device)
            if k.dim() == 3:
                k_tnd = k.contiguous()
                v_tnd = v.contiguous()
                cu_kv = params.get("cu_seqlens_kv")
            else:
                if layout_kv == "BSND":
                    k_bnsd = k.permute(0, 2, 1, 3)
                    v_bnsd = v.permute(0, 2, 1, 3)
                else:
                    k_bnsd = k
                    v_bnsd = v
                k_tnd, v_tnd, cu_kv = self._bnsd_to_tnd_packed(k_bnsd, v_bnsd, params)

        cu_q = params.get("cu_seqlens_q")
        if cu_q is None:
            raise ValueError("TND layout 需要 cu_seqlens_q")
        if cu_kv is None:
            raise ValueError("TND layout 需要 cu_seqlens_kv")

        seqlens_q = [int(cu_q[i + 1]) - int(cu_q[i]) for i in range(len(cu_q) - 1)]
        seqlens_kv = [int(cu_kv[i + 1]) - int(cu_kv[i]) for i in range(len(cu_kv) - 1)]
        max_sq = max(seqlens_q) if seqlens_q else 0
        max_skv = max(seqlens_kv) if seqlens_kv else 0

        out_tnd = gpu_varlen_attn(
            q_tnd,
            k_tnd,
            v_tnd,
            cu_q,
            cu_kv,
            max_sq,
            max_skv,
            softmax_scale,
            causal,
            self._device,
        )
        layout_out = params.get("layout_out", layout_q)
        if layout_out == "TND" or (layout_q == "TND" and layout_out == "TND"):
            return out_tnd.contiguous()
        elif layout_out in ("BNSD", "BSND"):
            cu = (
                cu_q
                if isinstance(cu_q, list)
                else (
                    cu_q.cpu().tolist()
                    if isinstance(cu_q, torch.Tensor)
                    else list(cu_q)
                )
            )
            B = len(cu) - 1
            N = out_tnd.shape[1]
            D = out_tnd.shape[2]
            max_s = max(int(cu[i + 1]) - int(cu[i]) for i in range(B))
            out_bnsd = torch.zeros(
                B, N, max_s, D, dtype=out_tnd.dtype, device=out_tnd.device
            )
            for i in range(B):
                start, end = int(cu[i]), int(cu[i + 1])
                out_bnsd[i, :, : end - start, :] = out_tnd[start:end, :, :].permute(
                    1, 0, 2
                )
            if layout_out == "BSND":
                return out_bnsd.permute(0, 2, 1, 3).contiguous()
            return out_bnsd.contiguous()
        else:
            return out_tnd.contiguous()

    # ─── 路径 C: flash_attn_func ──────────────────────────────────────────────

    def _compute_fixed(self, inputs, params, softmax_scale, causal):
        layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        q = inputs["q"].to(self._device)

        if layout_kv.startswith("PA_"):
            k_bnsd, v_bnsd = self._pa_to_bnsd_pair(inputs, layout_kv, params)
            k = k_bnsd.to(self._device)
            v = v_bnsd.to(self._device)
        else:
            k = inputs["k"].to(self._device)
            v = inputs["v"].to(self._device)

        if layout_q == "BSND":
            q_bshd = q.contiguous()
        else:
            q_bshd = q.permute(0, 2, 1, 3).contiguous()

        kv_layout = "BNSD" if layout_kv.startswith("PA_") else layout_kv
        if kv_layout == "BSND":
            k_bshd = k.contiguous()
            v_bshd = v.contiguous()
        else:
            k_bshd = k.permute(0, 2, 1, 3).contiguous()
            v_bshd = v.permute(0, 2, 1, 3).contiguous()

        out_bshd = gpu_fixed_attn(q_bshd, k_bshd, v_bshd, softmax_scale, causal)
        layout_out = params.get("layout_out", layout_kv)
        if layout_out == "BSND":
            return out_bshd.contiguous()
        else:
            return out_bshd.permute(0, 2, 1, 3).contiguous()

    # ─── 工具方法 ─────────────────────────────────────────────────────────────

    def _bnsd_to_tnd_pack_q(self, q_bnsd, params):
        """BNSD Q [B, N, S, D] → TND [total_q, N, D] + cu_seqlens_q"""
        B, N, S, D = q_bnsd.shape
        seqused_q = params.get("seqused_q") or params.get("actual_seq_qlen")
        slices = []
        cu_q = [0]
        for i in range(B):
            s = int(seqused_q[i]) if seqused_q else S
            slices.append(q_bnsd[i, :, :s, :].permute(1, 0, 2))
            cu_q.append(cu_q[-1] + s)
        q_tnd = torch.cat(slices, dim=0)
        return q_tnd, cu_q

    def _pa_to_bnsd_pair(self, inputs, layout_kv, params):
        from core.backends.cpu import _pa_to_bnsd

        k_cpu = (
            inputs["k"].cpu()
            if inputs["k"].device != torch.device("cpu")
            else inputs["k"]
        )
        v_cpu = (
            inputs["v"].cpu()
            if inputs["v"].device != torch.device("cpu")
            else inputs["v"]
        )
        k_bnsd = _pa_to_bnsd(k_cpu, layout_kv, params)
        v_bnsd = _pa_to_bnsd(v_cpu, layout_kv, params)
        return k_bnsd, v_bnsd

    def _bnsd_to_tnd_packed(self, k_bnsd, v_bnsd, params):
        seqused_kv = params.get("seqused_kv") or params.get("actual_seq_kvlen")
        B_kv = k_bnsd.shape[0]
        k_slices, v_slices = [], []
        for i in range(B_kv):
            s = int(seqused_kv[i]) if seqused_kv else k_bnsd.shape[2]
            k_slices.append(k_bnsd[i, :, :s, :])
            v_slices.append(v_bnsd[i, :, :s, :])
        k_tnd = torch.cat(k_slices, dim=1).permute(1, 0, 2).contiguous()
        v_tnd = torch.cat(v_slices, dim=1).permute(1, 0, 2).contiguous()

        cu_kv = params.get("cu_seqlens_kv")
        if cu_kv is None:
            cu_kv = [0]
            for i in range(B_kv):
                s = int(seqused_kv[i]) if seqused_kv else k_bnsd.shape[2]
                cu_kv.append(cu_kv[-1] + s)
        return k_tnd, v_tnd, cu_kv


def _pa_nz_to_bbnd(t, params):
    """PA_NZ [num_blocks, N, D//blk_elem, block_size, blk_elem] → PA_BBND [num_blocks, block_size, N, D]"""
    dtype_str = params.get("Dtype", "fp16")
    blk_elem = 16 if dtype_str in ("fp16", "bf16") else 32
    num_blocks, N, d_frac, bs, _ = t.shape
    t_perm = t.permute(0, 1, 3, 2, 4).contiguous()
    t_reshaped = t_perm.reshape(num_blocks, N, bs, d_frac * blk_elem)
    return t_reshaped.permute(0, 2, 1, 3)


def _unpack_tnd_to_bnsd(q_tnd, cu_seqlens_q):
    """TND Q [T, N, D] 或 [1, N, T, D] → BNSD [B, N, max_sq, D]"""
    if isinstance(cu_seqlens_q, torch.Tensor):
        cu = cu_seqlens_q.cpu().tolist()
    else:
        cu = list(cu_seqlens_q)
    B = len(cu) - 1

    if q_tnd.dim() == 3:
        T, N, D = q_tnd.shape
        max_sq = max(int(cu[i + 1]) - int(cu[i]) for i in range(B))
        out = torch.zeros(B, N, max_sq, D, dtype=q_tnd.dtype, device=q_tnd.device)
        for i in range(B):
            start, end = int(cu[i]), int(cu[i + 1])
            out[i, :, : end - start, :] = q_tnd[start:end, :, :].permute(1, 0, 2)
    else:
        N, D = q_tnd.shape[1], q_tnd.shape[3]
        max_sq = max(int(cu[i + 1]) - int(cu[i]) for i in range(B))
        out = torch.zeros(B, N, max_sq, D, dtype=q_tnd.dtype, device=q_tnd.device)
        for i in range(B):
            start, end = int(cu[i]), int(cu[i + 1])
            out[i, :, : end - start, :] = q_tnd[0, :, start:end, :]
    return out


def _pack_bnsd_to_tnd(out_bnsd, cu_seqlens_q):
    """BNSD [B, N, max_sq, D] → TND [T, N, D]"""
    if isinstance(cu_seqlens_q, torch.Tensor):
        cu = cu_seqlens_q.cpu().tolist()
    else:
        cu = list(cu_seqlens_q)
    B = len(cu) - 1
    slices = []
    for i in range(B):
        seq_len = int(cu[i + 1]) - int(cu[i])
        slices.append(out_bnsd[i, :, :seq_len, :].permute(1, 0, 2))
    return torch.cat(slices, dim=0)
