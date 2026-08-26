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

        orig_dtype = q.dtype
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

        if is_tnd_q:
            if seqused_q is not None:
                q_lens = seqused_q.tolist()
            else:
                B = len(cu_q_list) - 1
                q_lens = [cu_q_list[b + 1] - cu_q_list[b] for b in range(B)]
        else:
            q_lens = seqused_q.tolist() if seqused_q is not None else None

        if is_tnd_kv:
            if seqused_kv is not None:
                kv_lens = seqused_kv.tolist()
            else:
                B = len(cu_kv_list) - 1
                kv_lens = [cu_kv_list[b + 1] - cu_kv_list[b] for b in range(B)]
        else:
            kv_lens = seqused_kv.tolist() if seqused_kv is not None else None

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
            compute_mode = (
                "tiled"
                if (max_sq >= self._tile_size and max_product > self.TILED_CROSSOVER)
                else "full"
            )
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
        )

        if layout_out in ("TND", "NTD"):
            attn_out = out_f32.to(orig_dtype)
            if layout_out == "NTD":
                attn_out = attn_out.permute(1, 0, 2).contiguous()
        else:
            out_bnsd = out_f32.to(orig_dtype)
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
                kv_lens = self._get_kv_lens_from_block_table(block_table, B)
            elif layout_kv == "BNSD":
                kv_lens = [k.shape[2]] * B
            else:
                kv_lens = [k.shape[1]] * B

        is_tnd_out = layout_q == "TND"
        if is_tnd_out:
            total_q = q.shape[0]
            out_tnd = torch.zeros(
                total_q, N_q, dv, dtype=torch.float32, device=q.device
            )
            # 算子 lse 输出为 (N, T), 与算子对齐
            lse_tnd = torch.zeros(N_q, total_q, dtype=torch.float32, device=q.device)
            out = None
            lse = None
        else:
            # out/lse 的 S 维度用张量全量(算子按 q 张量形状输出),
            # 不能取 max(seqused_q) — 否则 seqused < S1 时 shape mismatch
            if layout_q == "BSND":
                max_sq = q.shape[1]
            else:
                max_sq = q.shape[2]
            out = torch.zeros(B, N_q, max_sq, dv, dtype=torch.float32, device=q.device)
            lse = torch.zeros(B, N_q, max_sq, dtype=torch.float32, device=q.device)
            out_tnd = None
            lse_tnd = None

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
                        )
                    else:
                        o, l = compute_fn(q_bh, k_bh, v_bh, mask_b, scale, sink)
                    if is_tnd_out:
                        out_tnd[q_start : q_start + sq, qh, :] = o
                        lse_tnd[qh, q_start : q_start + sq] = l
                    else:
                        out[b, qh, :sq, :] = o
                        lse[b, qh, :sq] = l

            del q_b, k_b, v_b

        # 无效行对齐算子: 有效行之后未计算的行(/空序列) LSE 刷 +inf
        # (整行被 mask 的行已在 _compute_*_single 中置 +inf; 此处只覆盖
        #  seqused_q < S1 的 padding 行与 sq<=0 的空序列行)
        if is_tnd_out:
            for b in range(B):
                sq = q_lens[b]
                seg_start = cu_q[b]
                seg_end = cu_q[b + 1]
                if sq < seg_end - seg_start:
                    lse_tnd[:, seg_start + sq : seg_end] = float("inf")
        else:
            for b in range(B):
                sq = q_lens[b]
                if sq < max_sq:
                    lse[b, :, sq:] = float("inf")

        if is_tnd_out:
            return out_tnd, lse_tnd
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
    ):
        Sq, D = q.shape
        Skv = k.shape[0]
        dv = v.shape[-1]
        BS = self._tile_size

        TILES_Q = (Sq + BS - 1) // BS
        TILES_KV = (Skv + BS - 1) // BS

        out = torch.zeros(Sq, dv, dtype=torch.float32, device=q.device)
        o_sum = torch.zeros(Sq, 1, dtype=torch.float32, device=q.device)
        o_max = torch.full(
            (Sq, 1),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=q.device,
        )

        Q_BLOCKS = list(torch.split(q, BS, dim=0))
        K_BLOCKS = list(torch.split(k, BS, dim=0))
        V_BLOCKS = list(torch.split(v, BS, dim=0))
        o_BLOCKS = list(torch.split(out, BS, dim=0))
        s_BLOCKS = list(torch.split(o_sum, BS, dim=0))
        m_BLOCKS = list(torch.split(o_max, BS, dim=0))

        for i in range(TILES_Q):
            Qi = Q_BLOCKS[i]
            Sq_start = i * BS
            Sq_end = min(Sq_start + BS, Sq)

            for j in range(TILES_KV):
                oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

                Kj = K_BLOCKS[j]
                Vj = V_BLOCKS[j]
                Sk_start = j * BS
                Sk_end = min(Sk_start + BS, Skv)

                S_ij = torch.matmul(Qi, Kj.t()) * scale

                if mask_mode in (3, 4):
                    q_idx = torch.arange(Sq_start, Sq_end, device=q.device)
                    k_idx = torch.arange(Sk_start, Sk_end, device=q.device)
                    tile_mask = self._build_mask_mode_mask(
                        q_idx,
                        k_idx,
                        mask_mode,
                        delta,
                        win_left,
                        win_right,
                    )
                    if tile_mask is not None:
                        S_ij = S_ij.masked_fill(tile_mask, float("-inf"))
                elif mask is not None:
                    S_ij = S_ij.masked_fill(
                        mask[Sq_start:Sq_end, Sk_start:Sk_end], float("-inf")
                    )

                m_block = S_ij.amax(dim=-1, keepdim=True)
                m_new = torch.maximum(mi, m_block)

                P_ij = torch.exp(S_ij - m_new)
                s_block = P_ij.sum(dim=-1, keepdim=True)

                update_old = torch.exp(mi - m_new)
                P_V = torch.matmul(P_ij, Vj[: Kj.shape[0], :])

                o_BLOCKS[i] = update_old * oi + P_V
                s_BLOCKS[i] = update_old * si + s_block
                m_BLOCKS[i] = m_new

        out = torch.cat(o_BLOCKS, dim=0)
        out_sum = torch.cat(s_BLOCKS, dim=0)
        out_max = torch.cat(m_BLOCKS, dim=0)

        out = out / (out_sum + self.EPSILON)

        lse = torch.log(out_sum + self.EPSILON) + out_max
        lse = lse.squeeze(-1)
        lse = torch.where(
            out_sum.squeeze(-1) <= 0, torch.full_like(lse, float("inf")), lse
        )

        return out, lse

    def _compute_full_single(self, q, k, v, mask, scale, sink):
        scores = torch.matmul(q, k.t()) * scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        row_max = scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )

        if sink is not None:
            row_max = torch.maximum(row_max, torch.full_like(row_max, sink))

        exp_scores = torch.exp(scores - row_max)
        row_sum = exp_scores.sum(dim=-1, keepdim=True)

        if sink is not None:
            row_sum = row_sum + math.exp(sink - row_max)

        probs = exp_scores / (row_sum + self.EPSILON)
        out = torch.matmul(probs, v)

        lse = torch.log(row_sum + self.EPSILON) + row_max
        lse = lse.squeeze(-1)
        lse = torch.where(
            row_sum.squeeze(-1) <= 0, torch.full_like(lse, float("inf")), lse
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

    @staticmethod
    def _get_kv_lens_from_block_table(block_table, B):
        bt = block_table.to(torch.int64)
        lens = []
        for b in range(B):
            row = bt[b]
            valid = (row >= 0).sum().item()
            lens.append(int(valid))
        return lens


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


def cpu_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **_unused,
):
    """Compute CPU golden reference using FlashAttnGolden."""
    if sinks is not None:
        message = "FlashAttn golden does not support sinks yet"
        print(message)
        raise NotImplementedError(message)

    q_cpu = _to_cpu(q)
    k_cpu = _to_cpu(k)
    v_cpu = _to_cpu(v)
    bt_cpu = _to_cpu(block_table)
    cu_q_cpu = _to_cpu(cu_seqlens_q)
    cu_kv_cpu = _to_cpu(cu_seqlens_kv)
    sq_cpu = _to_cpu(seqused_q)
    skv_cpu = _to_cpu(seqused_kv)
    sinks_cpu = _to_cpu(sinks)

    num_heads_q, num_heads_kv, head_dim = _derive_head_dims(
        q_cpu, k_cpu, layout_q, layout_kv
    )

    golden = FlashAttnGolden(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        softmax_scale=float(softmax_scale),
    )

    out, lse = golden.forward_golden(
        q_cpu,
        k_cpu,
        v_cpu,
        block_table=bt_cpu,
        cu_seqlens_q=cu_q_cpu,
        cu_seqlens_kv=cu_kv_cpu,
        seqused_q=sq_cpu,
        seqused_kv=skv_cpu,
        sinks=sinks_cpu,
        softmax_scale=float(softmax_scale),
        mask_mode=int(mask_mode),
        win_left=int(win_left),
        win_right=int(win_right),
        max_seqlen_q=int(max_seqlen_q) if max_seqlen_q is not None else -1,
        max_seqlen_kv=int(max_seqlen_kv) if max_seqlen_kv is not None else -1,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=bool(return_softmax_lse),
        device_mode="cpu",
        compute_mode="full",
    )

    if not return_softmax_lse:
        lse = None
    return out, lse
