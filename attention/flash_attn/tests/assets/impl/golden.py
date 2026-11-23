#!/usr/bin/python
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

"""FlashAttn golden reference implementation and TTK adapter."""

import math
from typing import Optional, Tuple

import torch

from .dropout import DropoutMask


class FlashAttnGolden:
    """FlashAttn Golden reference implementation (non-quantized).

    Supports two computation modes:
      - "full":    per-batch bmm(Q,K) -> softmax -> bmm(P,V)
      - "tiled":   block-wise online softmax, mirroring operator internals

    Both modes support device_mode="cpu" and "npu".
    """

    TILE_SIZE = 2048
    EPSILON = 1e-20
    UNBOUNDED_TOKENS = 2147483647
    TILED_CROSSOVER = 3 * 1024 * 1024  # S1*S2 > this -> tiled saves memory

    def __init__(
        self,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int = 128,
        softmax_scale: Optional[float] = None,
        tile_size: Optional[int] = None,
    ):
        self._num_heads_q = num_heads_q
        self._num_heads_kv = num_heads_kv
        self._head_dim = head_dim
        self._softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )
        self._tile_size = tile_size or self.TILE_SIZE

    def forward_golden(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = 1.0,
        mask_mode: int = 0,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_kv: str = "BSND",
        layout_out: str = "BSND",
        return_softmax_lse: bool = False,
        dropout_p: float = 0.0,
        seed: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        device_mode: str = "cpu",
        compute_mode: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device_mode not in ("cpu", "npu"):
            raise ValueError(f"device_mode must be 'cpu' or 'npu', got {device_mode}")
        if compute_mode not in ("auto", "tiled", "full"):
            raise ValueError(
                f"compute_mode must be 'auto', 'tiled' or 'full', got {compute_mode}"
            )

        # golden 始终根据 mask_mode/win_left/win_right 自行构造 mask,
        # 不使用传入的 attn_mask tensor (该 tensor 仅供 NPU 算子使用)
        attn_mask = None

        if device_mode == "npu":
            for name, t in [("q", q), ("k", k), ("v", v)]:
                if t.device.type != "npu":
                    raise ValueError(
                        f"device_mode='npu' requires {name} to be on NPU, "
                        f"got {t.device}"
                    )

        allowed = {
            "BSND": (("BSND", "PA_BBND", "PA_BNBD", "PA_NZ"), ("BSND",)),
            "BNSD": (("BNSD", "PA_BBND", "PA_BNBD", "PA_NZ"), ("BNSD", "BSND")),
            "TND": (("TND", "PA_BBND", "PA_BNBD", "PA_NZ"), ("TND",)),
        }
        if (
            layout_q not in allowed
            or layout_kv not in allowed[layout_q][0]
            or layout_out not in allowed[layout_q][1]
        ):
            raise ValueError(
                f"unsupported layouts: {layout_q}, {layout_kv}, {layout_out}"
            )
        if mask_mode not in (0, 3, 4):
            raise ValueError(f"unsupported mask_mode: {mask_mode}")
        if (
            self._num_heads_kv <= 0
            or self._num_heads_q < self._num_heads_kv
            or self._num_heads_q % self._num_heads_kv
        ):
            raise ValueError("num_heads_q must be a positive multiple of num_heads_kv")
        if sinks is not None and sinks.shape == (0,):
            sinks = None
        if sinks is not None and (
            sinks.dtype != torch.float32 or sinks.shape != (self._num_heads_q,)
        ):
            raise ValueError("sinks must be float32 with shape (num_heads_q,)")
        orig_dtype = q.dtype
        # P 的量化精度(算子 vector->Cube 交接时 exp 权重 cast 到输入精度)
        self._p_dtype = orig_dtype
        scale = softmax_scale if softmax_scale is not None else self._softmax_scale

        if device_mode == "cpu":
            if block_table is not None:
                block_table = block_table.cpu()
            if cu_seqlens_q is not None:
                cu_seqlens_q = cu_seqlens_q.cpu()
            if cu_seqlens_kv is not None:
                cu_seqlens_kv = cu_seqlens_kv.cpu()
            if seqused_q is not None:
                seqused_q = seqused_q.cpu()
            if seqused_kv is not None:
                seqused_kv = seqused_kv.cpu()
            if sinks is not None:
                sinks = sinks.cpu()

        is_tnd_q = layout_q == "TND"
        is_tnd_kv = layout_kv == "TND"

        cu_q_list = cu_seqlens_q.tolist() if cu_seqlens_q is not None else None
        cu_kv_list = cu_seqlens_kv.tolist() if cu_seqlens_kv is not None else None

        if is_tnd_q and cu_q_list is None:
            raise ValueError("TND query requires cu_seqlens_q")
        if is_tnd_kv and cu_kv_list is None:
            raise ValueError("TND KV requires cu_seqlens_kv")
        batch = len(cu_q_list) - 1 if is_tnd_q else q.shape[0]
        for name, tensor, prefix, layout, total in (
            ("q", seqused_q, cu_q_list, layout_q, q.shape[0]),
            ("kv", seqused_kv, cu_kv_list, layout_kv, k.shape[0]),
        ):
            if layout == "TND" and (
                len(prefix) != batch + 1
                or prefix[0] != 0
                or prefix[-1] != total
                or any(x > y for x, y in zip(prefix, prefix[1:]))
            ):
                raise ValueError(f"cu_seqlens_{name} must cover the TND tensor from 0")
            if tensor is not None and (tensor.ndim != 1 or tensor.numel() != batch):
                raise ValueError(f"seqused_{name} must contain {batch} lengths")
        q_physical = (
            [cu_q_list[b + 1] - cu_q_list[b] for b in range(batch)]
            if is_tnd_q
            else [q.shape[2 if layout_q == "BNSD" else 1]] * batch
        )
        q_lens = seqused_q.tolist() if seqused_q is not None else q_physical
        if layout_kv.startswith("PA_"):
            if seqused_kv is None or block_table is None:
                raise ValueError("paged KV requires seqused_kv and block_table")
            bs = k.shape[
                3 if layout_kv == "PA_NZ" else 1 if layout_kv == "PA_BBND" else 2
            ]
            if block_table.ndim != 2 or block_table.shape[0] != batch:
                raise ValueError("block_table must have one row per batch")
            kv_physical = [block_table.shape[1] * bs] * batch
        elif is_tnd_kv:
            kv_physical = [cu_kv_list[b + 1] - cu_kv_list[b] for b in range(batch)]
        else:
            if k.shape[0] != batch or v.shape[0] != batch:
                raise ValueError("query and KV batch dimensions must match")
            kv_physical = [k.shape[2 if layout_kv == "BNSD" else 1]] * batch
        kv_lens = seqused_kv.tolist() if seqused_kv is not None else kv_physical
        for name, used, physical in (
            ("q", q_lens, q_physical),
            ("kv", kv_lens, kv_physical),
        ):
            if any(
                length < 0 or length > limit for length, limit in zip(used, physical)
            ):
                raise ValueError(f"seqused_{name} exceeds physical sequence capacity")

        dropout = DropoutMask(
            dropout_p,
            seed,
            offset,
            self._num_heads_q,
            q_lens,
            kv_lens,
            layout_q,
            self._num_heads_kv,
        )
        if dropout.dropout_p == 0.0:
            dropout = None

        if compute_mode == "auto":
            if q_lens is not None and kv_lens is not None:
                max_sq = max(q_lens)
                max_product = max(sq * skv for sq, skv in zip(q_lens, kv_lens))
            else:
                max_sq = q.shape[2] if layout_q == "BNSD" else q.shape[1]
                if block_table is not None:
                    if kv_lens is not None:
                        max_skv = max(kv_lens)
                    else:
                        max_skv = k.shape[2] if layout_kv == "BNSD" else k.shape[1]
                else:
                    max_skv = k.shape[2] if layout_kv == "BNSD" else k.shape[1]
                max_product = max_sq * max_skv
            compute_mode = "tiled" if (max_product > self.TILED_CROSSOVER) else "full"
        self.last_compute_mode = compute_mode

        out_f32, lse_f32 = self._compute_per_batch(
            q,
            k,
            v,
            q_lens,
            kv_lens,
            layout_q,
            layout_kv,
            cu_q_list,
            cu_kv_list,
            block_table,
            mask_mode,
            attn_mask,
            win_left,
            win_right,
            scale,
            sinks,
            compute_mode,
            dropout,
        )

        if layout_out in ("TND", "NTD"):
            attn_out = self._cast_round(out_f32, orig_dtype).to(orig_dtype)
            if layout_out == "NTD":
                attn_out = attn_out.permute(1, 0, 2).contiguous()
        else:
            out_bnsd = self._cast_round(out_f32, orig_dtype).to(orig_dtype)
            if layout_out == "BNSD":
                attn_out = out_bnsd
            elif layout_out == "BSND":
                attn_out = out_bnsd.permute(0, 2, 1, 3).contiguous()
            else:
                raise ValueError(f"Unsupported layout_out: {layout_out}")

        if not return_softmax_lse:
            softmax_lse = torch.zeros(1, dtype=torch.float32, device=attn_out.device)
        else:
            softmax_lse = lse_f32

        return attn_out, softmax_lse

    def _compute_per_batch(
        self,
        q,
        k,
        v,
        q_lens,
        kv_lens,
        layout_q,
        layout_kv,
        cu_q,
        cu_kv,
        block_table,
        mask_mode,
        attn_mask,
        win_left,
        win_right,
        scale,
        sinks,
        compute_mode="tiled",
        dropout=None,
    ):
        N_q = self._num_heads_q
        N_kv = self._num_heads_kv
        if layout_kv == "PA_NZ":
            dv = v.shape[2] * v.shape[4]
        else:
            dv = v.shape[-1]
        group = N_q // N_kv

        if layout_q == "TND":
            B = len(cu_q) - 1
        elif layout_q == "BNSD":
            B = q.shape[0]
        else:
            B = q.shape[0]

        if q_lens is None:
            if layout_q == "TND":
                q_lens = [cu_q[b + 1] - cu_q[b] for b in range(B)]
            elif layout_q == "BNSD":
                q_lens = [q.shape[2]] * B
            else:
                q_lens = [q.shape[1]] * B

        if kv_lens is None:
            if layout_kv == "TND" and cu_kv is not None:
                kv_lens = [cu_kv[b + 1] - cu_kv[b] for b in range(B)]
            elif block_table is not None:
                raise ValueError("paged KV requires seqused_kv")
            elif layout_kv == "BNSD":
                kv_lens = [k.shape[2]] * B
            else:
                kv_lens = [k.shape[1]] * B

        is_tnd_out = layout_q == "TND"
        if is_tnd_out:
            total_q = q.shape[0]
            out_shape = (total_q, N_q, dv)
            lse_shape = (N_q, total_q)
        else:
            # out/lse 的 S 维度用张量全量(算子按 q 张量形状输出),
            # 不能取 max(seqused_q) — 否则 seqused < S1 时 shape mismatch
            if layout_q == "BSND":
                max_sq = q.shape[1]
            else:
                max_sq = q.shape[2]
            out_shape = (B, N_q, max_sq, dv)
            lse_shape = (B, N_q, max_sq)

        out = torch.zeros(out_shape, dtype=torch.float32, device=q.device)
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
        # Skipped batches and Q padding keep the per-head initial value.
        if sinks is None:
            lse.fill_(float("inf"))
        else:
            lse.copy_(sinks[:, None])

        for b in range(B):
            sq = q_lens[b]
            skv = kv_lens[b]
            if is_tnd_out:
                q_start = cu_q[b]
            if sq <= 0 or skv <= 0:
                continue

            q_b, k_b, v_b = self._extract_batch_bnsd_f32(
                q,
                k,
                v,
                b,
                sq,
                skv,
                layout_q,
                layout_kv,
                cu_q,
                cu_kv,
                block_table,
            )

            compute_fn = (
                self._compute_tiled_single
                if compute_mode == "tiled"
                else self._compute_full_single
            )
            mask_b = (
                None
                if (compute_mode == "tiled" and mask_mode in (3, 4))
                else self._build_batch_mask(
                    sq,
                    skv,
                    mask_mode,
                    attn_mask,
                    q.device,
                    b,
                    win_left,
                    win_right,
                )
            )

            delta = skv - sq
            for kv_h in range(N_kv):
                k_bh = k_b[0, kv_h, :skv, :]
                v_bh = v_b[0, kv_h, :skv, :]
                for g in range(group):
                    qh = kv_h * group + g
                    q_bh = q_b[0, qh, :sq, :]
                    sink = sinks[qh] if sinks is not None else None
                    drop = dropout.for_head(b, qh) if dropout is not None else None
                    if compute_mode == "tiled":
                        o, l = compute_fn(
                            q_bh,
                            k_bh,
                            v_bh,
                            mask_b,
                            scale,
                            sink,
                            mask_mode,
                            delta,
                            win_left,
                            win_right,
                            drop,
                        )
                    else:
                        o, l = compute_fn(q_bh, k_bh, v_bh, mask_b, scale, sink, drop)
                    if is_tnd_out:
                        out[q_start : q_start + sq, qh, :] = o
                        lse[qh, q_start : q_start + sq] = l
                    else:
                        out[b, qh, :sq, :] = o
                        lse[b, qh, :sq] = l

            del q_b, k_b, v_b

        return out, lse

    def _extract_batch_bnsd_f32(
        self,
        q,
        k,
        v,
        b,
        sq,
        skv,
        layout_q,
        layout_kv,
        cu_q,
        cu_kv,
        block_table,
    ):
        if layout_q == "BNSD":
            q_b = q[b : b + 1, :, :sq, :].float()
        elif layout_q == "BSND":
            q_b = q[b : b + 1, :sq, :, :].permute(0, 2, 1, 3).contiguous().float()
        elif layout_q == "TND":
            s, e = cu_q[b], cu_q[b + 1]
            q_b = q[s:e].unsqueeze(0).permute(0, 2, 1, 3).contiguous().float()
        else:
            raise ValueError(f"Unsupported layout_q: {layout_q}")

        if layout_kv == "BNSD":
            k_b = k[b : b + 1, :, :skv, :].float()
            v_b = v[b : b + 1, :, :skv, :].float()
        elif layout_kv == "BSND":
            k_b = k[b : b + 1, :skv, :, :].permute(0, 2, 1, 3).contiguous().float()
            v_b = v[b : b + 1, :skv, :, :].permute(0, 2, 1, 3).contiguous().float()
        elif layout_kv == "TND":
            s, e = cu_kv[b], cu_kv[b + 1]
            k_b = k[s:e].unsqueeze(0).permute(0, 2, 1, 3).contiguous().float()
            v_b = v[s:e].unsqueeze(0).permute(0, 2, 1, 3).contiguous().float()
        elif layout_kv in ("PA_BBND", "PA_BNBD"):
            N_kv = k.shape[1] if layout_kv == "PA_BNBD" else k.shape[2]
            D = k.shape[-1]
            dv = v.shape[-1]
            bs = k.shape[2] if layout_kv == "PA_BNBD" else k.shape[1]
            k_b = torch.zeros(1, N_kv, skv, D, dtype=torch.float32, device=k.device)
            v_b = torch.zeros(1, N_kv, skv, dv, dtype=torch.float32, device=v.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            ids = bt[b, :num_blocks_b]
            if (
                ids.numel() != num_blocks_b
                or ((ids < 0) | (ids >= k.shape[0]) | (ids >= v.shape[0])).any()
            ):
                raise ValueError("block_table references a page outside the KV cache")
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                if layout_kv == "PA_BBND":
                    k_b[0, :, offset : offset + valid, :] = (
                        k[block_id, :valid, :, :].permute(1, 0, 2).float()
                    )
                    v_b[0, :, offset : offset + valid, :] = (
                        v[block_id, :valid, :, :].permute(1, 0, 2).float()
                    )
                else:
                    k_b[0, :, offset : offset + valid, :] = k[
                        block_id, :, :valid, :
                    ].float()
                    v_b[0, :, offset : offset + valid, :] = v[
                        block_id, :, :valid, :
                    ].float()
        elif layout_kv == "PA_NZ":
            N_kv = k.shape[1]
            D = k.shape[2] * k.shape[4]
            dv = v.shape[2] * v.shape[4]
            bs = k.shape[3]
            k_b = torch.zeros(1, N_kv, skv, D, dtype=torch.float32, device=k.device)
            v_b = torch.zeros(1, N_kv, skv, dv, dtype=torch.float32, device=v.device)
            bt = block_table.to(torch.int64)
            num_blocks_b = (skv + bs - 1) // bs
            ids = bt[b, :num_blocks_b]
            if (
                ids.numel() != num_blocks_b
                or ((ids < 0) | (ids >= k.shape[0]) | (ids >= v.shape[0])).any()
            ):
                raise ValueError("block_table references a page outside the KV cache")
            for blk in range(num_blocks_b):
                block_id = bt[b, blk].item()
                offset = blk * bs
                valid = min(bs, skv - offset)
                if valid <= 0:
                    continue
                k_blk = (
                    k[block_id, :, :, :valid, :]
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(N_kv, valid, D)
                )
                v_blk = (
                    v[block_id, :, :, :valid, :]
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(N_kv, valid, dv)
                )
                k_b[0, :, offset : offset + valid, :] = k_blk.float()
                v_b[0, :, offset : offset + valid, :] = v_blk.float()
        else:
            raise ValueError(f"Unsupported layout_kv: {layout_kv}")

        return q_b, k_b, v_b

    def _compute_tiled_single(
        self,
        q,
        k,
        v,
        mask,
        scale,
        sink,
        mask_mode=0,
        delta=0,
        win_left=-1,
        win_right=-1,
        dropout=None,
    ):
        def mask_fn(qs, qe, ks, ke):
            if mask_mode in (3, 4):
                return self._build_mask_mode_mask(
                    torch.arange(qs, qe, device=q.device),
                    torch.arange(ks, ke, device=q.device),
                    mask_mode,
                    delta,
                    win_left,
                    win_right,
                )
            return None if mask is None else mask[qs:qe, ks:ke]

        return self._attention(q, k, v, scale, self._tile_size, mask_fn, sink, dropout)

    def _compute_full_single(self, q, k, v, mask, scale, sink, dropout=None):
        return self._attention(
            q,
            k,
            v,
            scale,
            max(q.shape[0], k.shape[0], 1),
            lambda *args: mask,
            sink,
            dropout,
        )

    @staticmethod
    def _cast_round(value, dtype):
        """Match the arch35 vector-to-Cube P cast: round to nearest, ties away from zero."""
        rounded = value.to(dtype)
        low = rounded.float()
        direction = torch.where(value > low, float("inf"), float("-inf"))
        other = torch.nextafter(rounded, direction.to(dtype)).float()
        tie = (value == (low + other) * 0.5) & (value != low)
        return torch.where(tie, torch.where(value.abs() > low.abs(), other, low), low)

    def _attention(self, q, k, v, scale, tile_size, mask_fn, sink, dropout=None):
        rows, length = q.shape[0], k.shape[0]
        out = torch.zeros((rows, v.shape[-1]), dtype=torch.float32, device=q.device)
        lse = torch.full((rows,), float("inf"), dtype=torch.float32, device=q.device)
        for qs in range(0, rows, tile_size):
            qe = min(qs + tile_size, rows)
            maximum = torch.full((qe - qs, 1), float("-inf"), device=q.device)
            total = torch.zeros_like(maximum)
            if sink is not None:
                maximum.fill_(float(sink))
                total.fill_(1.0)
            acc = torch.zeros((qe - qs, v.shape[-1]), device=q.device)
            for ks in range(0, length, tile_size):
                ke = min(ks + tile_size, length)
                scores = (q[qs:qe] @ k[ks:ke].T) * scale
                mask = mask_fn(qs, qe, ks, ke)
                if mask is not None:
                    scores.masked_fill_(mask, float("-inf"))
                new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
                correction = (maximum - safe_max).exp()
                weights = (scores - safe_max).exp()
                total = total * correction + weights.sum(-1, keepdim=True)
                if dropout is not None:
                    weights = dropout(weights, qs, qe, ks, ke)
                acc = (
                    acc * correction
                    + self._cast_round(weights, self._p_dtype) @ v[ks:ke]
                )
                maximum = new_max
            empty = total == 0
            output = acc / torch.where(empty, 1.0, total)
            row_lse = (total.log() + maximum).squeeze(-1)
            out[qs:qe] = output.masked_fill(empty, 0.0)
            lse[qs:qe] = (
                row_lse
                if sink is not None
                else row_lse.masked_fill(empty.squeeze(-1), float("inf"))
            )
        return out, lse

    @staticmethod
    def _build_mask_mode_mask(
        q_idx,
        k_idx,
        mask_mode,
        delta,
        win_left,
        win_right,
    ):
        q_pos = q_idx.unsqueeze(1) + delta
        k_pos = k_idx.unsqueeze(0)

        if mask_mode == 3:
            return k_pos > q_pos
        if mask_mode != 4:
            return None

        left_bound = -win_left if win_left >= 0 else -FlashAttnGolden.UNBOUNDED_TOKENS
        right_bound = win_right if win_right >= 0 else FlashAttnGolden.UNBOUNDED_TOKENS
        return (k_pos < (q_pos + left_bound)) | (k_pos > (q_pos + right_bound))

    def _build_batch_mask(
        self,
        sq,
        skv,
        mask_mode,
        attn_mask,
        device,
        batch_index,
        win_left=-1,
        win_right=-1,
    ):
        if mask_mode in (3, 4):
            q_idx = torch.arange(sq, device=device)
            k_idx = torch.arange(skv, device=device)
            return self._build_mask_mode_mask(
                q_idx,
                k_idx,
                mask_mode,
                skv - sq,
                win_left,
                win_right,
            )

        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask[batch_index, 0]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[batch_index]
            return attn_mask[:sq, :skv].to(torch.bool)

        return None


# ── TTK adapter ──


def _to_cpu(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _derive_head_dims(q, k, layout_q, layout_kv):
    head_dim = int(q.shape[-1])
    if layout_q == "TND":
        num_heads_q = int(q.shape[1])
    elif layout_q == "BNSD":
        num_heads_q = int(q.shape[1])
    else:
        num_heads_q = int(q.shape[2])

    if layout_kv == "TND":
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "BNSD":
        num_heads_kv = int(k.shape[1])
    elif layout_kv in ("PA_BNBD", "PA_NZ"):
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "PA_BBND":
        num_heads_kv = int(k.shape[2])
    else:
        num_heads_kv = int(k.shape[2])
    return num_heads_q, num_heads_kv, head_dim


def _run_reference(golden_cls, arguments):
    """Shared CPU adapter: derive heads, run forward_golden, drop suppressed LSE."""
    layout_q = arguments["layout_q"]
    layout_kv = arguments["layout_kv"]
    q_cpu = _to_cpu(arguments["q"])
    k_cpu = _to_cpu(arguments["k"])
    v_cpu = _to_cpu(arguments["v"])

    num_heads_q, num_heads_kv, head_dim = _derive_head_dims(
        q_cpu, k_cpu, layout_q, layout_kv
    )

    golden = golden_cls(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        softmax_scale=float(arguments["softmax_scale"]),
    )

    out, lse = golden.forward_golden(
        q_cpu,
        k_cpu,
        v_cpu,
        metadata=_to_cpu(arguments.get("metadata")),
        block_table=_to_cpu(arguments.get("block_table")),
        cu_seqlens_q=_to_cpu(arguments.get("cu_seqlens_q")),
        cu_seqlens_kv=_to_cpu(arguments.get("cu_seqlens_kv")),
        seqused_q=_to_cpu(arguments.get("seqused_q")),
        seqused_kv=_to_cpu(arguments.get("seqused_kv")),
        sinks=_to_cpu(arguments.get("sinks")),
        softmax_scale=float(arguments["softmax_scale"]),
        mask_mode=int(arguments["mask_mode"]),
        win_left=int(arguments["win_left"]),
        win_right=int(arguments["win_right"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=arguments["layout_out"],
        return_softmax_lse=bool(arguments["return_softmax_lse"]),
        dropout_p=arguments.get("dropout_p", 0.0),
        seed=_to_cpu(arguments.get("seed")),
        offset=_to_cpu(arguments.get("offset")),
        device_mode="cpu",
        compute_mode="auto",
    )

    if not arguments["return_softmax_lse"]:
        lse = None
    return out, lse


def cpu_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    dropout_p: float = 0.0,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    **_unused,
):
    """Compute CPU golden reference using FlashAttnGolden."""
    from . import cross_check

    if cross_check.enabled():
        return cross_check.get_case(_unused["testcase_name"])[0]
    arguments = dict(locals())
    arguments.pop("_unused")
    arguments.pop("cross_check")
    return _run_reference(FlashAttnGolden, arguments)
