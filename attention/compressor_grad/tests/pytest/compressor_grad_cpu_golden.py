#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


import torch
from typing import Optional, List, Tuple


def compressor_grad_golden(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    d_cpm_kv: torch.Tensor,
    softmax_score: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    coff: int = 1,
    device: str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    matmul_mode: str = "two",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compressor 算子的反向传播。

    参数:
        x:            输入张量 [B, S, H] 或 [T, H]（BSH 布局或 TH 布局）
        wkv:          KV 投影权重 (coff * head_dim, H)
        wgate:        Gate 投影权重 (coff * head_dim, H)
        d_cpm_kv:     前向输出 compressed KV 的上游梯度 ∂Loss/∂cmp_kv
                      BSH: (B, ceil(S/cmp_ratio), head_dim)
                      TH:  (min(T, T//cmp_ratio+B), head_dim)
        kv:           前向保存的中间变量（reshape 后的 KV 数据）
                      BSH: (B, ceil(S/cmp_ratio), coff*cmp_ratio, head_dim)
                      TH:  (min(T, T//cmp_ratio+B), coff*cmp_ratio, head_dim)
        softmax_score: 前向保存的 softmax 权重
                      形状同 kv
        start_pos:    每个 batch 的起始全局序列位置 (B,) int32。
                      None 时默认全为 0（即所有 batch 从头开始）
        cmp_ratio:    压缩率：每 cmp_ratio 个 token 压缩为 1 个
        coff:         重叠因子（1=无重叠, 2=相邻块有 50% 重叠）
        cu_seqlens:   TH 布局下的累积序列长度 (B+1,) int32。
                      BSH 布局时传 None，函数内部不依赖此参数定位 token
        seqused:      每个 batch 实际参与压缩的 token 数 (B,) int32。
                      None 表示使用全部 sequence length

    返回:
        d_x:      输入 x 的梯度 ∂Loss/∂x
        d_wkv:    权重 wkv 的梯度 ∂Loss/∂wkv
        d_wgate:  权重 wgate 的梯度 ∂Loss/∂wgate
        d_ape:    APE 位置编码的梯度 ∂Loss/∂ape, (cmp_ratio, coff * head_dim)
    """
    # ══════════════════════════════════════════════════════════════════════
    # 阶段 A: 解析输入参数，确定数据布局
    # ══════════════════════════════════════════════════════════════════════

    # head_dim: 每个注意力头的维度 D。
    # wkv 的 shape 是 (coff * head_dim, H)，所以 head_dim = wkv.shape[0] // coff
    head_dim = wkv.shape[0] // coff

    # 两种数据布局：
    #   TH 布局 (is_th_layout=True):  x shape (T, H)，B 和 S 合并到 T 维
    #       此时 cu_seqlens 提供每个 batch 在 T 维的起止边界
    #   BSH 布局 (is_th_layout=False): x shape (B, S, H)，batch 独立存储
    #       此时不需要 cu_seqlens，直接用 b_idx * S 定位
    is_th_layout = cu_seqlens is not None

    # B: batch 数量。start_pos 为 None 时默认全 0
    if start_pos is None:
        if is_th_layout:
            num_batches = cu_seqlens.shape[0] - 1
        else:
            num_batches = x.shape[0]
        start_positions = [0] * num_batches
    else:
        num_batches = start_pos.shape[0]
        start_positions = start_pos.tolist()

    # 统一计算设备（三方 B 在 NPU 上跑 / 两方与三方 C 在 CPU 上跑）
    dev = x.device if device == "cpu" else device
    x = x.to(dev)
    wkv = wkv.to(dev)
    wgate = wgate.to(dev)
    d_cpm_kv = d_cpm_kv.to(dev)
    softmax_score = softmax_score.to(dev)
    kv = kv.to(dev)
    # matmul 输入：two/high 口径提升到 compute_dtype；same 口径保持 io_dtype 不提升
    x_f32 = x.to(compute_dtype) if matmul_mode != "same" else x
    wkv_f32 = wkv.to(compute_dtype) if matmul_mode != "same" else wkv
    wgate_f32 = wgate.to(compute_dtype) if matmul_mode != "same" else wgate

    # 确定总 token 数和 batch/seq 维度（仅 BSH 布局需要保留 shape 信息用于最后 reshape）
    if is_th_layout:
        total_tokens = x.shape[0]
        batch_size = seq_len = None  # TH 布局下不需要恢复 BSH shape
    else:
        batch_size, seq_len, hidden_size = x.shape
        total_tokens = batch_size * seq_len

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 B: 分配梯度累积缓冲区
    # ══════════════════════════════════════════════════════════════════════
    #
    # 前向计算中有三个"源头"张量会产生梯度：
    #   (a) new_kv    = x @ wkv.T         — 仅仅来自 matmul
    #   (b) new_score = x @ wgate.T + APE — 来自 matmul + 加法
    #   (c) APE 本身                        — 位置编码参数
    #
    # 在反向过程中，我们先将每个压缩块的梯度反向传播到 (a)(b)(c) 的对应
    # token/位置，得到 d_new_kv、d_new_score、d_ape。最后一步再将这些
    # 累积梯度通过 matmul 反向得到 d_x, d_wkv, d_wgate。

    # 维度约定: T=total_tokens, C=coff*head_dim, H=hidden_size
    #
    # d_new_kv:    (T, C)  ∂Loss/∂new_kv — 仅来自 matmul 反向
    d_new_kv = torch.zeros(
        total_tokens, coff * head_dim, device=dev, dtype=compute_dtype
    )

    # d_new_score: (T, C)  ∂Loss/∂new_score — 同时流入 matmul 反向和 APE 反向
    d_new_score = torch.zeros(
        total_tokens, coff * head_dim, device=dev, dtype=compute_dtype
    )

    # d_ape:       (cmp_ratio, C)  ∂Loss/∂ape
    d_ape = torch.zeros(cmp_ratio, coff * head_dim, device=dev, dtype=compute_dtype)

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 C: 逐 batch、逐压缩块进行反向传播
    # ══════════════════════════════════════════════════════════════════════
    #
    # 遍历顺序与前向完全一致：先遍历 batch，再在每个 batch 内按 cmp_ratio
    # 步长遍历压缩块。对于每个压缩块，执行反步 1→3（参见文件头部推导）。
    #
    # grad_output_index / bsh_block_idx: 按压缩块顺序消费三个输入（见 C.4a）
    grad_output_index = 0

    for batch_idx in range(num_batches):
        # ── C.1  确定当前 batch 的序列范围 ────────────────────────────────
        # batch_start: 当前 batch 在全局序列中的绝对起始位置
        batch_start = start_positions[batch_idx]

        # batch_seq_used: 当前 batch 实际参与计算的 token 数
        # seqused 非 None 时优先使用，否则用 sequence length
        if seqused is not None:
            batch_seq_used = seqused[batch_idx].item()
        elif is_th_layout:
            batch_seq_used = int(
                cu_seqlens[batch_idx + 1].item() - cu_seqlens[batch_idx].item()
            )
        else:
            batch_seq_used = x.shape[1]

        # compress_limit: 需要压缩的序列位置上限（向下对齐到 cmp_ratio 的整数倍）
        # 例如: batch_start=5, seq_used=10, cmp_ratio=4 → limit = (5+10)//4*4 = 12
        #       位置 5,6,7,8,9,10,11 会被压缩，12,13,14 不压缩（不足一个完整块）
        compress_limit = (batch_start + batch_seq_used) // cmp_ratio * cmp_ratio

        # batch_seq_idx: 当前在 batch 内已处理到的相对位置（相对于 batch_start）
        batch_seq_idx = 0
        # bsh_block_idx: BSH 布局下当前 batch 的压缩块序号（从 0 开始）
        bsh_block_idx = 0

        while batch_seq_idx < batch_seq_used:
            # ── C.2  当前压缩块的全局序列范围 ─────────────────────────────
            # block_seq_start: 当前块在全局序列中的起始位置
            block_seq_start = batch_start + batch_seq_idx

            # block_seq_end: 当前块在全局序列中的结束位置（向前对齐到 cmp_ratio 整数倍）
            # 对于最后一个不完整块，end 会被截断到 batch 结尾
            block_seq_end = (block_seq_start // cmp_ratio) * cmp_ratio + cmp_ratio
            if block_seq_end > batch_start + batch_seq_used:
                block_seq_end = batch_start + batch_seq_used

            # 推进到下一块（在 guard clause 之前更新，避免重复）
            batch_seq_idx = block_seq_end - batch_start

            # ── C.4  尾块跳过（不足 cmp_ratio 个 token，仅写 cache）──
            if block_seq_start >= compress_limit:
                break

            # ═══════════ 反向传播：反步 1→3 ═══════════

            # ── 获取当前块的上游梯度和前向中间变量 ──
            if is_th_layout:
                d_compressed_kv = d_cpm_kv[grad_output_index].to(compute_dtype)
                saved_kv = kv[grad_output_index].to(compute_dtype)
                saved_softmax = softmax_score[grad_output_index].to(compute_dtype)
                grad_output_index += 1
            else:
                d_compressed_kv = d_cpm_kv[batch_idx, bsh_block_idx].to(compute_dtype)
                saved_kv = kv[batch_idx, bsh_block_idx].to(compute_dtype)
                saved_softmax = softmax_score[batch_idx, bsh_block_idx].to(
                    compute_dtype
                )
                bsh_block_idx += 1

            # ══════════════════════════════════════════════════════════
            # 【反步 1】element-wise mul + reduce_sum 反向
            # ══════════════════════════════════════════════════════════
            #
            # 前向: C = Σ_i (K_i ⊙ W_i)   K=(N, hd), W=(N, hd), C=(1, hd)
            #   其中 N=coff*cmp_ratio, hd=head_dim
            # 反向: dK_i = dC ⊙ W_i,  dW_i = dC ⊙ K_i  (广播: dC (1,hd)→(N,hd))

            d_compressed_kv_2d = d_compressed_kv.unsqueeze(0)  # (1,hd) → (N,hd)
            d_kv_block = d_compressed_kv_2d * saved_softmax  # (N,hd)
            d_score_weighted = d_compressed_kv_2d * saved_kv  # (N,hd)
            # ══════════════════════════════════════════════════════════
            # 【反步 2】softmax 反向 (dim=0, 每列独立)
            # ══════════════════════════════════════════════════════════
            #
            # 前向: S = softmax(Z, dim=0)   S,Z ∈ (N,hd)
            # 反向: dZ = S ⊙ (dS - column_sum(S ⊙ dS))
            #   S ⊙ dS → (N,hd)
            #   column_sum → (1,hd)
            #   dZ → (N,hd)

            softmax_backward_sum = (saved_softmax * d_score_weighted).sum(
                dim=0, keepdim=True
            )  # (1,hd)
            d_score_block = saved_softmax * (
                d_score_weighted - softmax_backward_sum
            )  # (N,hd)
            d_score_block = d_score_block.view(cmp_ratio, coff, head_dim)
            d_kv_block = d_kv_block.view(cmp_ratio, coff, head_dim)

            # ══════════════════════════════════════════════════════════
            # 【反步 3】梯度路由 → 映射回 flat new_kv/new_score 数组
            # ══════════════════════════════════════════════════════════
            #
            # d_kv_block, d_score_block ∈ (N, hd)  其中 N=coff*cmp_ratio
            # 路由目标: d_new_kv, d_new_score ∈ (T, C)  其中 C=coff*hd
            #
            # kv_2d 的 N 行按 overlap_id 分组:
            #   overlap_id=0:       行 [0, cmp_ratio)       → 列 [0, hd)
            #   overlap_id=coff-1:  行 [(coff-1)*cr, N)    → 列 [(coff-1)*hd, C)

            # flat 偏移: 当前块在展平 (T, C) 数组中的 token 行范围
            flat_offset_base = (
                int(cu_seqlens[batch_idx].item())
                if is_th_layout
                else batch_idx * x.shape[1]
            )
            flat_offset_start = flat_offset_base + (block_seq_start - batch_start)
            flat_offset_end = flat_offset_base + (block_seq_end - batch_start)

            # ── 3a  当前 overlap 槽位 (overlap_id = coff - 1) ──
            current_overlap_id = coff - 1
            current_dim_start = current_overlap_id * head_dim
            current_dim_end = (current_overlap_id + 1) * head_dim

            current_tokens_from_cache = 0
            if batch_start == block_seq_start:
                current_tokens_from_cache = batch_start % cmp_ratio

            block_row_start = current_overlap_id * cmp_ratio + current_tokens_from_cache
            block_row_end = current_overlap_id * cmp_ratio + cmp_ratio

            # d_kv_block[rows, :] → d_new_kv[tokens, cols]   (N,hd)子块 → (T,C)子块
            d_new_kv[
                flat_offset_start:flat_offset_end,
                current_dim_start:current_dim_end,
            ] += d_kv_block[current_tokens_from_cache:cmp_ratio, coff - 1, :]

            d_new_score[
                flat_offset_start:flat_offset_end,
                current_dim_start:current_dim_end,
            ] += d_score_block[current_tokens_from_cache:cmp_ratio, coff - 1, :]

            # d_ape: (cmp_ratio, C)  按 pos%cmp_ratio 累加 d_score_block 对应行
            ape_offset = block_seq_start % cmp_ratio
            n_pos = block_seq_end - block_seq_start

            d_ape[
                ape_offset : ape_offset + n_pos, current_dim_start:current_dim_end
            ] += d_score_block[current_tokens_from_cache:cmp_ratio, coff - 1, :]

            # ── 3b  前一个 overlap 槽位 (overlap_id = 0, 仅 coff == 2) ──
            if coff == 2:
                prev_tokens_from_cache = 0
                if batch_start == block_seq_start:
                    prev_tokens_from_cache = cmp_ratio
                elif block_seq_start - cmp_ratio < batch_start:
                    prev_tokens_from_cache = batch_start % cmp_ratio

                # 全来自 cache，无需路由
                if prev_tokens_from_cache >= cmp_ratio:
                    continue

                prev_dim_start = 0
                prev_dim_end = head_dim
                block_row_start = prev_tokens_from_cache
                block_row_end = cmp_ratio

                prev_flat_offset_start = flat_offset_start - (
                    cmp_ratio - prev_tokens_from_cache
                )
                prev_flat_offset_end = flat_offset_start

                # (N,hd)子块 → (T,C)子块，前一块列偏移 [0, hd)
                d_new_kv[
                    prev_flat_offset_start:prev_flat_offset_end,
                    prev_dim_start:prev_dim_end,
                ] += d_kv_block[prev_tokens_from_cache:cmp_ratio, 0, :]

                d_new_score[
                    prev_flat_offset_start:prev_flat_offset_end,
                    prev_dim_start:prev_dim_end,
                ] += d_score_block[prev_tokens_from_cache:cmp_ratio, 0, :]

                n_pos = cmp_ratio - prev_tokens_from_cache
                d_ape[
                    prev_tokens_from_cache:cmp_ratio, prev_dim_start:prev_dim_end
                ] += d_score_block[prev_tokens_from_cache:cmp_ratio, 0, :]

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 D: Matmul 反向
    # ══════════════════════════════════════════════════════════════════════
    #
    # 维度约定: T=total_tokens, C=coff*head_dim, H=hidden_size
    #
    # 前向有两条独立的 matmul:
    #   new_kv    = x @ wkv.T      → (T, H) @ (H, C) = (T, C)   不含 APE
    #   new_score = x @ wgate.T    → (T, H) @ (H, C) = (T, C)   APE 后加
    #
    # Matmul 反向公式 (Y = X @ W.T):
    #   dX = dY @ W       → (T, C) @ (C, H) = (T, H)
    #   dW = dY.T @ X     → (C, T) @ (T, H) = (C, H)
    #
    # x 同时流入两条路径，d_x 是两条之和:
    #   d_x     = d_new_kv @ wkv   + d_new_score @ wgate
    #             (T,C)@(C,H)→(T,H)   (T,C)@(C,H)→(T,H)
    #   d_wkv   = d_new_kv.T @ x_flat   → (C, T) @ (T, H) = (C, H)
    #   d_wgate = d_new_score.T @ x_flat → (C, T) @ (T, H) = (C, H)

    x_flat = (
        x_f32 if is_th_layout else x_f32.reshape(total_tokens, x.shape[-1])
    )  # (T, H)
    iodtype = x.dtype
    if matmul_mode == "two":
        # 两方口径（现状）：d_new_kv 量化到 iodtype 后提升回 compute_dtype（模拟 kernel
        # 中间量 FP16/BF16 存储）；wkv/wgate/x 已提升 compute_dtype → FP32 matmul
        d_new_kv_q = d_new_kv.to(iodtype).to(compute_dtype)
        d_new_score_q = d_new_score.to(iodtype).to(compute_dtype)
    elif matmul_mode == "same":
        # 三方 B 口径（same）：仅量化到 iodtype（模拟中间存储），权重不提升 → BF16 进 BF16 出
        d_new_kv_q = d_new_kv.to(iodtype)
        d_new_score_q = d_new_score.to(iodtype)
    else:  # "high"
        # 三方 C 口径：全程 compute_dtype（float64），无量化、无提升
        d_new_kv_q = d_new_kv
        d_new_score_q = d_new_score
    if matmul_mode == "same":
        # BF16 进 BF16 出（torch: bf16@bf16 → bf16），golden 内不再 cast，输出即 io_dtype
        d_x_flat = d_new_kv_q @ wkv + d_new_score_q @ wgate  # (T, H)
        d_wkv = d_new_kv_q.T @ x_flat  # (C, H)
        d_wgate = d_new_score_q.T @ x_flat  # (C, H)
    else:
        d_x_flat = d_new_kv_q @ wkv_f32 + d_new_score_q @ wgate_f32  # (T, H)
        d_wkv = d_new_kv_q.T @ x_flat  # (C, H)
        d_wgate = d_new_score_q.T @ x_flat  # (C, H)

    # 根据原始输入布局恢复 d_x 的形状，并转换回输入的 dtype；统一回 CPU 返回
    if is_th_layout:
        d_x = d_x_flat.to(x.dtype).cpu()
    else:
        d_x = d_x_flat.reshape(batch_size, seq_len, hidden_size).to(x.dtype).cpu()

    return (
        d_x,
        d_wkv.to(wkv.dtype).cpu(),
        d_wgate.to(wgate.dtype).cpu(),
        d_ape.float().cpu(),
    )
