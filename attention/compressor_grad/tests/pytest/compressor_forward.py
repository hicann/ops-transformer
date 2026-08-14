#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
Compressor 算子核心实现：前向 + 反向（纯 PyTorch autograd）。

计算流程（参考 compressor_golden.py 中的 cpu_compressor）：
  1. new_kv     = x @ wkv.T          # [*, coff * head_dim]
  2. new_score  = x @ wgate.T        # [*, coff * head_dim]
  3. new_score += ape（按位置，位置索引 = 全局序列位置 % cmp_ratio）
  4. 写入 page cache（kv_state / score_state，score 已带 ape）
  5. 对每个可压缩块（cmp_ratio 个 token）：
       收集 coff * cmp_ratio 个 token（当前块 + 可选的前一块）
       对 score 按列做 softmax → 权重
       compressed = sum(kv * weight, dim=0) → (1, head_dim)
  6. 返回堆叠后的 compressed KV

反向传播链：
  - 通过逐元素乘法和求和
  - 通过按列 softmax
  - 通过 page-cache 读取 → new_kv / new_score / 旧 state
  - 通过 matmul → x, wkv, wgate
  - 通过加法 → ape

精度标准（来自 compressor_golden.py 的 check_result）：
  bfloat16: rtol=0.0078125, atol=0.0001, 通过率≥99.5%
  默认:     rtol=0.005,     atol=0.000025, 通过率≥99.5%
"""

import torch
from typing import Optional, List, Tuple


# ===========================================================================
# 工具函数
# ===========================================================================


def _softmax_by_column(z: torch.Tensor) -> torch.Tensor:
    """
    按列计算 softmax（沿 dim=0，每列独立）。
    输入:  (N, D)
    输出:  (N, D)，每列之和为 1。
    """
    z_max = z.max(dim=0, keepdim=True).values
    exp_z = torch.exp(z - z_max)
    return exp_z / exp_z.sum(dim=0, keepdim=True)


class _QuantGrad(torch.autograd.Function):
    """前向恒等、反向梯度量化：模拟 kernel 反向中间量（dkv/dsb）经
    FP16/BF16 存储后回 FP32（与 compressor_grad_cpu_golden 的
    d_new_kv.to(iodtype).float() 对齐）。"""

    @staticmethod
    def forward(ctx, x, quant_dtype):
        ctx.quant_dtype = quant_dtype
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(ctx.quant_dtype).float(), None


def _read_cache(
    state: torch.Tensor,  # (block_num, block_size, coff * head_dim)
    block_table: torch.Tensor,  # (B, max_blocks)  int32
    b_idx: int,  # 当前 batch 索引
    start_seq: int,  # 起始序列位置
    end_seq: int,  # 结束序列位置
    d_start: int,  # 特征维度起始
    d_end: int,  # 特征维度结束
    cache_mode: int,  # 1=分页  2=每 batch 一个 block
) -> torch.Tensor:
    """
    从 page cache 中读取 [start_seq, end_seq) × [d_start, d_end) 的数据。
    返回 (end_seq - start_seq, d_end - d_start)。
    """
    block_size = state.shape[1]
    n = end_seq - start_seq
    chunks: list[torch.Tensor] = []
    done = 0
    while done < n:
        sid = start_seq + done
        if cache_mode == 1:
            bid = int(block_table[b_idx, sid // block_size].item())
        else:
            bid = int(block_table[b_idx].item())
        off = sid % block_size
        cnt = block_size - off
        if cnt > n - done:
            cnt = n - done
        chunks.append(
            state[bid : bid + 1, off : off + cnt, d_start:d_end].reshape(
                cnt, d_end - d_start
            )
        )
        done += cnt
    return torch.cat(chunks, dim=0)


def _write_cache(
    state: torch.Tensor,
    block_table: torch.Tensor,
    data: torch.Tensor,  # (seq_len, coff * head_dim)
    b_idx: int,
    start_seq: int,
    end_seq: int,
    cache_mode: int,
    update_mask: Optional[
        torch.Tensor
    ] = None,  # 与 state 同 shape 的 bool 张量，记录写入位置
) -> torch.Tensor:
    """
    向 page cache 写入数据。为避免破坏 autograd 计算图，返回新的 state 张量。
    传入 update_mask 时同步标记实际写入的位置（block_id 为 0 的无效块不标记）。
    """
    state = state.clone()
    block_size = state.shape[1]
    n = end_seq - start_seq
    done = 0
    while done < n:
        sid = start_seq + done
        if cache_mode == 1:
            bid = int(block_table[b_idx, sid // block_size].item())
            if bid == 0:
                # block_id 为 0 表示无效 block，跳过
                done += block_size - (sid % block_size)
                continue
        else:
            bid = int(block_table[b_idx].item())
        off = sid % block_size
        cnt = block_size - off
        if cnt > n - done:
            cnt = n - done
        state[bid : bid + 1, off : off + cnt, :] = data[done : done + cnt]
        if update_mask is not None:
            update_mask[bid : bid + 1, off : off + cnt, :] = True
        done += cnt
    return state


def _build_ape_for_flat_sequence(
    total_len: int,
    B: int,
    start_pos: List[int],
    ape: torch.Tensor,
    cmp_ratio: int,
    bs_combine: bool,
    cu_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    为展平序列的每个位置预计算 ape。
    返回 (total_len, coff*head_dim)，
    其中第 p 行 = ape[全局序列位置 % cmp_ratio, :]。
    """
    ape_out = torch.zeros(
        total_len, ape.shape[1], dtype=torch.float32, device=ape.device
    )
    for b in range(B):
        sp = start_pos[b]
        if bs_combine:
            fs = int(cu_seqlens[b].item())
            fe = int(cu_seqlens[b + 1].item())
        else:
            fs = b * (total_len // B) if B > 0 else 0
            fe = (b + 1) * (total_len // B) if B > 0 else total_len
        L = fe - fs
        if L <= 0:
            continue
        idx = (torch.arange(sp, sp + L, device=ape.device) % cmp_ratio).long()
        ape_out[fs:fe, :] = ape[idx, :]
    return ape_out


# ===========================================================================
# 前向传播（纯 PyTorch，autograd 自动处理反向）
# ===========================================================================


def compressor_forward(
    x: torch.Tensor,  # [B, S, H] 或 [T, H]（bs_combine 模式）
    wkv: torch.Tensor,  # (coff * head_dim, H)
    wgate: torch.Tensor,  # (coff * head_dim, H)
    kv_state: torch.Tensor,  # (block_num, block_size, coff * head_dim)
    score_state: torch.Tensor,  # (block_num, block_size, coff * head_dim)
    ape: torch.Tensor,  # (cmp_ratio, coff * head_dim)  位置编码
    block_table: torch.Tensor,  # (B, max_blocks) int32  page 映射表
    start_pos: List[int],  # 每个 batch 的起始序列位置
    cmp_ratio: int = 4,  # 压缩率：每 cmp_ratio 个 token 压缩为 1 个
    coff: int = 1,  # 1=无重叠  2=相邻块重叠
    cache_mode: int = 1,  # 1=分页 cache  2=每 batch 一个 block
    cu_seqlens: Optional[torch.Tensor] = None,  # bs_combine 模式下的累积序列长度
    seqused: Optional[List[int]] = None,  # 每个 batch 实际使用的序列长度
    return_intermediates: bool = False,  # 是否返回 d_kv, d_score, d_sc2
    return_update_mask: bool = False,  # 是否返回 kv/score state 的写入位置 mask（与 kv_state_out/score_state_out 同 shape）
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    matmul_mode: str = "two",
) -> Tuple:
    """
    执行 compressor 前向计算。

    返回:
      cmp_kv:      压缩后的 KV，(压缩块数, head_dim) 或 (B, 最大块数, head_dim)
      cmp_kv_mask: 同形状 bool 张量，标记有效位置
      kv_state:    更新后的 kv_state
      score_state: 更新后的 score_state

    matmul_mode:
      "two"（两方，现状）：x/wkv/wgate 提升 compute_dtype 后 FP32 matmul + _QuantGrad 量化模拟；
      "same"（三方 B）：输入不提升，BF16×BF16 → BF16 出，matmul→vec 前 cast 回 compute_dtype；
      "high"（三方 C）：全程 compute_dtype（float64），无量化。
    """
    orig_dtype = x.dtype
    # 统一计算设备（三方 B 在 NPU 上跑 / 两方与三方 C 在 CPU 上跑）
    dev = x.device if device == "cpu" else device
    x = x.to(dev)
    wkv = wkv.to(dev)
    wgate = wgate.to(dev)
    kv_state = kv_state.to(dev)
    score_state = score_state.to(dev)
    ape = ape.to(dev)
    head_dim = wkv.shape[0] // coff
    B = len(start_pos)
    bs_combine = cu_seqlens is not None

    # ---- 第一步：matmul ----
    ape_f32 = ape.to(compute_dtype)
    if matmul_mode == "same":
        # 三方 B：输入不提升（BF16 进），matmul 输出 BF16
        x_mm, wkv_mm, wgate_mm = x, wkv, wgate
    else:
        # two/high：提升到 compute_dtype
        x_mm = x.to(compute_dtype)
        wkv_mm = wkv.to(compute_dtype)
        wgate_mm = wgate.to(compute_dtype)

    if bs_combine:
        # x 形状为 (T, H)，直接做矩阵乘法
        flat_kv = x_mm @ wkv_mm.T  # (T, coff*head_dim)
        flat_score = x_mm @ wgate_mm.T
        T = x.shape[0]
    else:
        # x 形状为 (B, S, H)，先展平再乘
        B_in, S_in, H_in = x.shape
        flat_kv = x_mm.reshape(B_in * S_in, H_in) @ wkv_mm.T
        flat_score = x_mm.reshape(B_in * S_in, H_in) @ wgate_mm.T
        T = B_in * S_in

    if matmul_mode == "same":
        # matmul→vec cast：BF16 出 → FP32（vec 侧 +ape/softmax/加权求和需 FP32 输入）
        flat_kv = flat_kv.to(compute_dtype)
        flat_score = flat_score.to(compute_dtype)
    elif matmul_mode == "two":
        # 模拟 kernel 反向中间精度：dkv/dsb 梯度经 FP16/BF16 存储（golden 同款量化）
        flat_kv = _QuantGrad.apply(flat_kv, orig_dtype)
        flat_score = _QuantGrad.apply(flat_score, orig_dtype)
    # high：保持 compute_dtype（float64）
    # ---- 第二步：全局添加 ape（与 golden 中 in-place 加法等价）----
    ape_add = _build_ape_for_flat_sequence(
        T, B, start_pos, ape_f32, cmp_ratio, bs_combine, cu_seqlens
    )
    flat_score = flat_score + ape_add  # (T, coff*head_dim)，ape 已应用

    # ---- 第三步：分配输出缓冲区 ----
    # 与 golden 的对齐策略：TH 布局时预分配 (min(T, T//cmp_ratio+B), head_dim)
    # BSH 布局时预分配 (B, ceil(S/cmp_ratio), head_dim)
    if not bs_combine:
        S_val = x.shape[1]
        max_blocks = (S_val + cmp_ratio - 1) // cmp_ratio
        cmp_kv = torch.zeros(
            B, max_blocks, head_dim, dtype=torch.float32, device=x.device
        )
        cmp_mask = torch.zeros(
            B, max_blocks, head_dim, dtype=torch.bool, device=x.device
        )
    else:
        T_val = x.shape[0]
        out_rows = min(T_val, T_val // cmp_ratio + B)
        cmp_kv = torch.zeros(out_rows, head_dim, dtype=torch.float32, device=x.device)
        cmp_mask = torch.zeros(out_rows, head_dim, dtype=torch.bool, device=x.device)
        out_idx_flat = 0  # 当前写入行索引

    kv_state_out = kv_state.clone()
    score_state_out = score_state.clone()
    kv_update_mask = torch.zeros_like(kv_state_out, dtype=torch.bool)
    score_update_mask = torch.zeros_like(score_state_out, dtype=torch.bool)

    # 中间变量：按 output layout 预分配（与 cmp_kv 同 shape，最后维为 N=coff*cmpRatio）
    N_inter = coff * cmp_ratio
    if return_intermediates:
        if not bs_combine:
            kvIntermediate = torch.zeros(
                B, max_blocks, N_inter, head_dim, dtype=torch.float32, device=x.device
            )
            softmaxScoreIntermediate = torch.zeros(
                B, max_blocks, N_inter, head_dim, dtype=torch.float32, device=x.device
            )
        else:
            kvIntermediate = torch.zeros(
                out_rows, N_inter, head_dim, dtype=torch.float32, device=x.device
            )
            softmaxScoreIntermediate = torch.zeros(
                out_rows, N_inter, head_dim, dtype=torch.float32, device=x.device
            )

    if bs_combine:
        out_idx_flat = 0

    # ---- 第四步：逐 batch 逐块处理 ----
    for b_idx in range(B):
        b_start = start_pos[b_idx]

        # 确定当前 batch 的序列长度
        if seqused is not None:
            b_seq_used = seqused[b_idx]
        elif bs_combine:
            b_seq_used = int(cu_seqlens[b_idx + 1].item() - cu_seqlens[b_idx].item())
        else:
            b_seq_used = x.shape[1]

        # compress_seq_id：小于此位置的需要压缩
        compress_limit = (b_start + b_seq_used) // cmp_ratio * cmp_ratio
        batch_out_sc_id = 0
        batch_seq_idx = 0

        while batch_seq_idx < b_seq_used:
            # 当前块的全局起始/结束序列位置
            s_start = b_start + batch_seq_idx
            s_end = s_start // cmp_ratio * cmp_ratio + cmp_ratio
            if s_end > b_start + b_seq_used:
                s_end = b_start + b_seq_used

            # 展平数组中的偏移
            base = int(cu_seqlens[b_idx].item()) if bs_combine else b_idx * x.shape[1]
            off_s = base + (s_start - b_start)
            off_e = base + (s_end - b_start)

            # 是否保存到 cache / 是否压缩
            save_flag = (cache_mode == 1) or (
                s_start >= (compress_limit - (coff - 1) * cmp_ratio)
            )
            compress_flag = s_start < compress_limit

            # ---- 写入 page cache（kv 无 ape，score 已带 ape）----
            if save_flag:
                kv_state_out = _write_cache(
                    kv_state_out,
                    block_table,
                    flat_kv[off_s:off_e, :],
                    b_idx,
                    s_start,
                    s_end,
                    cache_mode,
                    kv_update_mask if return_update_mask else None,
                )
                score_state_out = _write_cache(
                    score_state_out,
                    block_table,
                    flat_score[off_s:off_e, :],
                    b_idx,
                    s_start,
                    s_end,
                    cache_mode,
                    score_update_mask if return_update_mask else None,
                )

            # ---- 执行压缩 ----
            if compress_flag:
                # 初始化 sc 缓冲区
                kvLocal = torch.zeros(coff, cmp_ratio, head_dim, device=x.device)
                sc_score = torch.full(
                    (coff, cmp_ratio, head_dim), float("-inf"), device=x.device
                )

                # --- 填充当前数据（coff_id = coff-1）---
                cur_cid = coff - 1
                ds_cur = cur_cid * head_dim
                de_cur = (cur_cid + 1) * head_dim
                cfs = 0  # 从 state cache 读取的数量
                if b_start == s_start:
                    cfs = b_start % cmp_ratio
                    if cfs > 0:
                        kvLocal[cur_cid, 0:cfs, :] = _read_cache(
                            kv_state_out,
                            block_table,
                            b_idx,
                            b_start - cfs,
                            b_start,
                            ds_cur,
                            de_cur,
                            cache_mode,
                        )
                        sc_score[cur_cid, 0:cfs, :] = _read_cache(
                            score_state_out,
                            block_table,
                            b_idx,
                            b_start - cfs,
                            b_start,
                            ds_cur,
                            de_cur,
                            cache_mode,
                        )

                # 从展平数组获取当前数据（kv 无 ape，score 已预加 ape）
                kvLocal[cur_cid, cfs:cmp_ratio, :] = flat_kv[off_s:off_e, ds_cur:de_cur]
                sc_score[cur_cid, cfs:cmp_ratio, :] = flat_score[
                    off_s:off_e, ds_cur:de_cur
                ]

                # --- 填充前一块数据（coff_id = 0），仅 coff == 2 时有效 ---
                if coff == 2:
                    prev_cid = 0
                    ds_pr = prev_cid * head_dim
                    de_pr = (prev_cid + 1) * head_dim
                    cfs_p = 0
                    if b_start == s_start:
                        # 第一个块：前一块数据全部来自 cache
                        cfs_p = cmp_ratio
                        if b_start >= cmp_ratio:
                            cs2 = b_start - b_start % cmp_ratio - cmp_ratio
                            kvLocal[prev_cid, 0:cfs_p, :] = _read_cache(
                                kv_state_out,
                                block_table,
                                b_idx,
                                cs2,
                                cs2 + cfs_p,
                                ds_pr,
                                de_pr,
                                cache_mode,
                            )
                            sc_score[prev_cid, 0:cfs_p, :] = _read_cache(
                                score_state_out,
                                block_table,
                                b_idx,
                                cs2,
                                cs2 + cfs_p,
                                ds_pr,
                                de_pr,
                                cache_mode,
                            )
                    elif s_start - cmp_ratio < b_start:
                        # 第二个块：部分来自 cache，部分来自当前数据的前半段
                        cfs_p = b_start % cmp_ratio
                        if cfs_p > 0:
                            cs2 = b_start - cfs_p
                            kvLocal[prev_cid, 0:cfs_p, :] = _read_cache(
                                kv_state_out,
                                block_table,
                                b_idx,
                                cs2,
                                b_start,
                                ds_pr,
                                de_pr,
                                cache_mode,
                            )
                            sc_score[prev_cid, 0:cfs_p, :] = _read_cache(
                                score_state_out,
                                block_table,
                                b_idx,
                                cs2,
                                b_start,
                                ds_pr,
                                de_pr,
                                cache_mode,
                            )
                    if cfs_p < cmp_ratio:
                        po_s = off_s - (cmp_ratio - cfs_p)
                        po_e = off_s
                        kvLocal[prev_cid, cfs_p:cmp_ratio, :] = flat_kv[
                            po_s:po_e, ds_pr:de_pr
                        ]
                        sc_score[prev_cid, cfs_p:cmp_ratio, :] = flat_score[
                            po_s:po_e, ds_pr:de_pr
                        ]

                # ---- 核心压缩：softmax + 加权求和 ----
                kv_2d = kvLocal.reshape(coff * cmp_ratio, head_dim)
                sc_score_2d = sc_score.reshape(coff * cmp_ratio, head_dim)
                # 将 -inf 替换为有限值以保证 softmax 正确
                sc_score_2d = torch.where(
                    torch.isinf(sc_score_2d),
                    torch.tensor(-1e9, device=x.device),
                    sc_score_2d,
                )
                # 按列 softmax → 权重
                sm = torch.softmax(sc_score_2d, dim=0)
                # 逐元素乘 + 沿 token 维求和
                compressed = (kv_2d * sm).sum(dim=0, keepdim=True)  # (1, head_dim)

                if return_intermediates:
                    if not bs_combine:
                        kvIntermediate[b_idx, batch_out_sc_id] = kv_2d.detach()
                        softmaxScoreIntermediate[b_idx, batch_out_sc_id] = sm.detach()
                    else:
                        kvIntermediate[out_idx_flat] = kv_2d.detach()
                        softmaxScoreIntermediate[out_idx_flat] = sm.detach()

                if not bs_combine:
                    cmp_kv[b_idx, batch_out_sc_id, :] = compressed
                    cmp_mask[b_idx, batch_out_sc_id, :] = True
                else:
                    cmp_kv[out_idx_flat, :] = compressed
                    cmp_mask[out_idx_flat, :] = True
                    out_idx_flat += 1

                batch_out_sc_id += 1

            batch_seq_idx = s_end - b_start

    # ---- 组装最终输出 ----
    # TH 和 BSH 都使用预分配缓冲区，无需额外拼接

    cmp_kv = cmp_kv.to(orig_dtype)
    if return_intermediates and coff == 2:
        # 中间量输出与 NPU kernel 一致的交错布局（偶行=prev、奇行=cur）：
        # 顺序布局 [prev 半区; cur 半区] 重排为 [p0,c0,p1,c1,...]，比对侧无需再 interleave
        half = cmp_ratio
        softmaxScoreIntermediate = torch.stack(
            [
                softmaxScoreIntermediate[..., :half, :],
                softmaxScoreIntermediate[..., half:, :],
            ],
            dim=-2,
        ).reshape(softmaxScoreIntermediate.shape)
        kvIntermediate = torch.stack(
            [kvIntermediate[..., :half, :], kvIntermediate[..., half:, :]], dim=-2
        ).reshape(kvIntermediate.shape)
    if return_intermediates and return_update_mask:
        return (
            cmp_kv,
            cmp_mask,
            kv_state_out,
            score_state_out,
            softmaxScoreIntermediate,
            kvIntermediate,
            kv_update_mask,
            score_update_mask,
        )
    if return_update_mask:
        return (
            cmp_kv,
            cmp_mask,
            kv_state_out,
            score_state_out,
            kv_update_mask,
            score_update_mask,
        )
    if return_intermediates:
        return (
            cmp_kv,
            cmp_mask,
            kv_state_out,
            score_state_out,
            softmaxScoreIntermediate,
            kvIntermediate,
        )
    return cmp_kv, cmp_mask, kv_state_out, score_state_out


# ===========================================================================
# 便捷包装：可直接在模型中调用的 Compressor 模块
# ===========================================================================


class Compressor(torch.nn.Module):
    """
    Compressor 算子模块，支持前向 + 反向。

    用法:
        compressor = Compressor(cmp_ratio=4, coff=1, cache_mode=1)
        cmp_kv, cmp_mask, kv_state, score_state = compressor(
            x, wkv, wgate, kv_state, score_state, ape,
            block_table, start_pos, cu_seqlens, seqused)
    """

    def __init__(self, cmp_ratio: int = 4, coff: int = 1, cache_mode: int = 1):
        super().__init__()
        self.cmp_ratio = cmp_ratio
        self.coff = coff
        self.cache_mode = cache_mode

    def forward(
        self,
        x: torch.Tensor,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        ape: torch.Tensor,
        block_table: torch.Tensor,
        start_pos: List[int],
        cu_seqlens: Optional[torch.Tensor] = None,
        seqused: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return compressor_forward(
            x,
            wkv,
            wgate,
            kv_state,
            score_state,
            ape,
            block_table,
            start_pos,
            cmp_ratio=self.cmp_ratio,
            coff=self.coff,
            cache_mode=self.cache_mode,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
        )
