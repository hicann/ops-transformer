#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MXFP4 Flash Attention NPU 拼接 Golden

用 torch_npu.npu_quant_matmul (float4_e2m1fn_x2 + e8m0, group_sizes=[1,1,32]) 拼接
mm1 (Q@K^T) / mm2 (P@V), 复刻 NPU kernel mmad 的加法树与舍入顺序, 消除
CPU FP32 matmul 与 NPU 的数值 diverge (极端场景: 全上界/全下界/0 等).

计算流程与 flash_attention_cpu_golden.py 的 blockwise_snap_local 路径逐行对齐:
  - Q/K: per-token-group (D 维, 32/group) MXFP4 量化
  - V:   per-channel-group (S 维, 32/group) MXFP4 量化 (group 不跨 batch)
  - P:   snap-to-log2-grid blockwise MXFP4 量化, p_scale = e8m0 2^(K_diff-2)
  - online softmax: m snap 到 ln2 整数网格, alpha = exp2(整数) 精确
  - mm1/mm2 走 npu_quant_matmul, S/O/l 累加 FP32

npu_quant_matmul scale 形状约定 (已在 test_npu_quant_matmul_fp4.py 上板验证):
  pertoken_scale: [M, K/64, 2] 连续 (尾维 2 = 相邻两个 32-group 的 e8m0 打包)
  input_scale:    x2 连续 [K, N/2] 时为 [K/64, N, 2] 连续;
                  x2 转置视图 (逻辑 [N,K] 沿 K 打包) 时为
                  [N, K/64, 2] 先上卡再 transpose(0,1) 的转置视图
  注意: fp4/e8m0 tensor 在 CPU 侧 transpose 后 .npu() 会触发 AICPU Transpose
  kernel 失败 (E39999), 必须先 .npu() 再 transpose.
"""

import math
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

from mx_quant_fp4_tool import mxfp4_quantize_last, pack_fp4_to_uint8
from flash_attention_cpu_golden import (
    _normalize_cu_seqlens,
    _resolve_s_dtype,
    _to_e8m0_uint8,
    _FP4_POS_LEVELS_FP32,
    _FP4_POS_MIDPOINTS_FP32,
)

# ==============================================================================
# npu_quant_matmul 拼接算子
# ==============================================================================


def _pair_scale_rows(scale_e8m0):
    """[M, G] e8m0 -> [M, G/2, 2] 连续 (pertoken_scale 约定)."""
    m, g = scale_e8m0.shape
    if g % 2 != 0:
        raise ValueError(f"scale group 数 ({g}) 必须为偶数 (64 对齐)")
    return scale_e8m0.contiguous().reshape(m, g // 2, 2)


def _pair_scale_cols(scale_e8m0):
    """[G, N] e8m0 -> [G/2, N, 2] dense (input_scale 约定, x2 连续模式).

    [kg, n, p] = scale[2*kg + p, n] (相邻两个 K-group 配对), 语义与
    mxfp8 golden c2 的 reshape(N,-1,2).permute(1,0,2).contiguous() 一致,
    已在 test_npu_quant_matmul_fp4.py case2 验证 (rel=0).

    显式构造而非 permute+contiguous: G/2==1 时 permute 结果是伪连续
    (stride (2,2,1)), C++ 侧 stride 检查会误判为 transpose 视图,
    与 dense x2 组合报 "transpose are not same".
    """
    g, n = scale_e8m0.shape
    if g % 2 != 0:
        raise ValueError(f"scale group 数 ({g}) 必须为偶数 (64 对齐)")
    out = torch.empty(g // 2, n, 2, dtype=scale_e8m0.dtype)
    out[..., 0] = scale_e8m0[0::2, :]
    out[..., 1] = scale_e8m0[1::2, :]
    return out


def _quant_matmul(x1_packed, pertoken_rows, x2_packed, input_scale_rows, x2_transposed):
    """npu_quant_matmul 包装.

    x1_packed:        [M, K/2] uint8 (fp4 打包, 逻辑 [M, K])
    pertoken_rows:    [M, K/32] e8m0 uint8
    x2_packed:        x2_transposed=False: [K, N/2] (逻辑 [K, N], 沿 N 打包)
                      x2_transposed=True:  [N, K/2] (逻辑 [N, K], 沿 K 打包),
                                           传入前需先 .npu() 再 .t()
    input_scale_rows: [K/32, N] (x2 连续) 或 [N, K/32] (x2 转置) e8m0 uint8
    返回: [M, N] FP32 (CPU tensor)
    """
    pt = _pair_scale_rows(pertoken_rows).view(torch.float8_e8m0fnu).npu()
    if x2_transposed:
        # x2 上卡后 transpose 视图; scale 同为转置视图 (转置性必须一致)
        it = _pair_scale_rows(input_scale_rows).view(torch.float8_e8m0fnu).npu()
        it = it.transpose(0, 1)
        # reshape (而非 contiguous): GQA 下 head 切片首维可能为 1, contiguous
        # 对 size-1 维是 no-op, 保留脏 stride (320,...), .t() 后被框架判为
        # 非真转置 -> "x2 and scale transpose are not same" (0105)
        x2 = x2_packed.reshape(x2_packed.shape).view(torch.float4_e2m1fn_x2).npu().t()
    else:
        it = _pair_scale_cols(input_scale_rows).view(torch.float8_e8m0fnu).npu()
        x2 = x2_packed.contiguous().view(torch.float4_e2m1fn_x2).npu()
    x1 = x1_packed.contiguous().view(torch.float4_e2m1fn_x2).npu()

    res = torch_npu.npu_quant_matmul(
        x1,
        x2,
        it,
        pertoken_scale=pt,
        output_dtype=torch.float32,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32],
    )
    torch.npu.synchronize()
    return res.cpu()


# ==============================================================================
# P 量化 (blockwise_snap_local 变体: 输出 fp4 codes + e8m0 p_scale)
# 计算路径与 flash_attention_cpu_golden._blockwise_snap_local_quantize_p_eff
# 逐行一致, 仅把 P_eff 的组装替换为 codes/p_scale 输出 (供 npu_quant_matmul).
# ==============================================================================


def _snap_local_p_quantize_codes(
    S: torch.Tensor,
    m_running: Optional[torch.Tensor],
    mx_block_size: int = 32,
    softmax_scale: float = 1.0,
):
    """S [bq, bkv] (s_dtype) -> (p_codes, p_scale_e8m0, m_new, P_eff).

    p_codes:      [bq, bkv_pad32] uint8 (0-7, fp4 正数编码, 已 pad 0)
    p_scale_e8m0: [bq, n_blk] uint8 (biased, 2^(K_diff-2))
    m_new:        FP32 K 网格整数 (running max)
    P_eff:        [bq, bkv] s_dtype, 与 CPU golden 完全一致 (供 l_i 求和)
    """
    NEG_INF = float("-inf")
    device = S.device
    src_dtype = S.dtype
    seqk_len = S.shape[-1]
    LN2 = torch.tensor(math.log(2.0), dtype=torch.float16, device=device)
    INV_LN2 = torch.tensor(1.0 / math.log(2.0), dtype=torch.float16, device=device)

    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pad_tensor = torch.full(
            (*S.shape[:-1], pad), NEG_INF, dtype=src_dtype, device=device
        )
        S_padded = torch.cat([S, pad_tensor], dim=-1)
    else:
        S_padded = S

    other_shape = S_padded.shape[:-1]
    n_blk = S_padded.shape[-1] // mx_block_size
    S_reshape = S_padded.reshape(*other_shape, n_blk, mx_block_size)

    # S 已处于 scaled 域 (主循环 fp32*scale -> fp16 单舍入, 对齐 kernel
    # FixpipeMm1 QF322F16_PRE deqScalar), 此处不再 Muls(dScale).
    # 与 kernel 行为对齐: S 全 -inf 时 max 保持 -inf (kernel 侧 MIN_VALUE*INV_LN2 也溢出 -inf),
    # P = exp(-inf - (-inf)) = NaN 传播到输出, 不再用 zeros_like 钳位兜底.
    m_block_scaled = S_reshape.max(dim=-1).values  # max of SCALED S (fp16)
    # fp16 域乘法: -inf 保持 -inf, 正常值与 kernel 舍入一致 (不加多余 cast)
    K_block = (m_block_scaled * INV_LN2).floor()
    m_block_snap = (K_block - 2) * LN2  # [..., n_blk], on log2 grid

    # kernel 指令链复刻 (vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32.h):
    #   Sub(s_fp16, snap_fp16) -> Exp(硬件 fp16 exp) -> Cast fp16->bf16
    x_fp16 = S_reshape - m_block_snap.unsqueeze(-1)  # fp16 Sub (scaled 域)
    P_fp16 = torch.exp(x_fp16)  # fp16 硬件 exp 等价
    P_local_r = P_fp16

    # fp4 量化: bf16 round-trip + bucketize(round-half-up)
    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    P_local_x4_bf16 = P_local_r.to(torch.bfloat16)
    P_local_x4 = P_local_x4_bf16.to(P_local_r.dtype)
    idx_padded = torch.bucketize(P_local_x4.contiguous(), midpoints, right=True)
    # flatten [.., n_blk, 32] -> [.., bkv_pad] (2 维 x1 供 npu_quant_matmul)
    p_codes = idx_padded.reshape(*S_padded.shape).to(torch.uint8)  # 0-7 fp4 正编码

    # running max (K 网格整数, FP32)
    K_ij = K_block.max(dim=-1).values  # s_dtype, 整数
    K_ij_fp32 = K_ij.float()
    if m_running is None:
        m_new = K_ij_fp32
    else:
        m_new = torch.maximum(m_running, K_ij_fp32)

    K_new_s = m_new.to(src_dtype)
    K_diff = K_block - K_new_s.unsqueeze(-1)  # <= 0, s_dtype 整数

    # p_scale: e8m0 biased (K_diff - 2), clamp [0, 254]
    p_scale_e8m0 = _to_e8m0_uint8(K_diff.to(torch.int32) - 2)

    # P_eff (与 CPU golden 数值一致, 供 l_i 求和)
    corr = torch.exp2(K_diff - 2.0)  # 2^(K_diff - 2)
    P_q_padded = levels[idx_padded]
    P_q = P_q_padded.reshape(*other_shape, -1)[..., :seqk_len]
    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len]
    P_eff = P_q * corr_broad

    return p_codes, p_scale_e8m0, m_new, P_eff


# ==============================================================================
# 主入口
# ==============================================================================


def flash_attention_npu_golden_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_kv,
    seq_used_q=None,
    seq_used_kv=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    s_dtype: str = "fp32",
) -> torch.Tensor:
    """MXFP4 Flash Attention NPU 拼接 golden (TND packed varlen).

    对齐 flash_attention_cpu_golden.flash_attention_cpu_golden_varlen 的
    quantize=True + quantize_p=True + quantize_p_mode='blockwise_snap_local'
    + v_quant_axis='seq_k' + s_layout='ND' 路径, 输出 FP32.

    约束: head_dim 与 d_v 需为 64 的倍数 (scale pairing 硬约束).
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("Varlen 期望 3D packed 输入 [total_seq, num_heads, head_dim]")
    if block_kv % 64 != 0:
        raise ValueError(f"block_kv ({block_kv}) 需为 64 的倍数")

    s_torch_dtype = _resolve_s_dtype(s_dtype)

    total_q, num_heads_q, head_dim = q.shape
    total_kv, num_heads_kv, _ = k.shape
    d_v = v.shape[-1]
    device = q.device

    if head_dim % 64 != 0:
        raise ValueError(f"head_dim ({head_dim}) 需为 64 的倍数")
    if d_v % 2 != 0:
        raise ValueError(f"d_v ({d_v}) 需为偶数 (fp4 打包)")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError(
            f"GQA: num_heads_q ({num_heads_q}) 必须能被 "
            f"num_heads_kv ({num_heads_kv}) 整除"
        )
    gs = num_heads_q // num_heads_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    batch_size = (
        cu_seqlens_q.numel() - 1
        if isinstance(cu_seqlens_q, torch.Tensor)
        else len(cu_seqlens_q) - 1
    )
    cu_q, used_q = _normalize_cu_seqlens(cu_seqlens_q, seq_used_q, batch_size, "q")
    cu_kv, used_kv = _normalize_cu_seqlens(cu_seqlens_kv, seq_used_kv, batch_size, "kv")
    # deqScalar 硬件通路: 标量以低位截断的 fp32 参与 (低 13 位清零, & -8192)
    hw_scale = float(
        torch.tensor(softmax_scale, dtype=torch.float32)
        .view(torch.int32)
        .__and__(-8192)
        .view(torch.float32)
    )
    if cu_q[-1] > total_q:
        raise ValueError(f"cu_seqlens_q[-1]={cu_q[-1]} > total_seq_q={total_q}")
    if cu_kv[-1] > total_kv:
        raise ValueError(f"cu_seqlens_kv[-1]={cu_kv[-1]} > total_seq_kv={total_kv}")

    # ---- MXFP4 量化: Q/K per-token-group (D 维), V per-channel-group (S 维) ----
    # q_codes [T, H, D], q_scale [T, H, D/32] (e8m0 biased uint8)
    q_codes, q_scale = mxfp4_quantize_last(
        q, quant_axis=-1, block_size=mx_block_size, mode=mx_mode
    )
    k_codes, k_scale = mxfp4_quantize_last(
        k, quant_axis=-1, block_size=mx_block_size, mode=mx_mode
    )

    # V 沿 S 维量化, group 不跨 batch (与 CPU golden v_quant_axis='seq_k' 一致)
    v_codes = torch.zeros_like(v, dtype=torch.uint8)
    v_scale_parts = []
    v_group_offsets = [0]
    for off, sk in zip(cu_kv, used_kv):
        if sk == 0:
            v_group_offsets.append(v_group_offsets[-1])
            continue
        codes_b, scale_b = mxfp4_quantize_last(
            v[off : off + sk], quant_axis=0, block_size=mx_block_size, mode=mx_mode
        )
        v_codes[off : off + sk] = codes_b
        v_scale_parts.append(scale_b)  # [ceil(sk/32), H, D]
        v_group_offsets.append(v_group_offsets[-1] + scale_b.shape[0])
    if v_scale_parts:
        v_scale = torch.cat(v_scale_parts, dim=0)  # [G_total, H, D]
    else:
        v_scale = torch.zeros(0, num_heads_kv, d_v, dtype=torch.uint8)

    # ---- GQA 展开 (量化后, codes/scale 同步 repeat) ----
    if gs > 1:
        k_codes = k_codes.repeat_interleave(gs, dim=1)
        k_scale = k_scale.repeat_interleave(gs, dim=1)
        v_codes = v_codes.repeat_interleave(gs, dim=1)
        v_scale = v_scale.repeat_interleave(gs, dim=1)

    # ---- fp4 打包 (尾维减半) ----
    q_packed = pack_fp4_to_uint8(q_codes)  # [T, H, D/2]
    k_packed = pack_fp4_to_uint8(k_codes)  # [T_kv, H, D/2]
    v_packed = pack_fp4_to_uint8(v_codes)  # [T_kv, H, d_v/2]

    # ---- 输出 buffer ----
    out = torch.zeros(total_q, num_heads_q, d_v, dtype=torch.float32, device=device)
    NEG_INF = float("-inf")

    # ---- 逐 batch / head / tile ----
    for b in range(batch_size):
        q_off, sq = cu_q[b], used_q[b]
        kv_off, sk = cu_kv[b], used_kv[b]
        if sq == 0 or sk == 0:
            continue

        v_gstart = v_group_offsets[b]
        causal_offset = (sk - sq) if causal else 0
        num_q_tiles = (sq + block_q - 1) // block_q
        num_kv_tiles = (sk + block_kv - 1) // block_kv

        for h in range(num_heads_q):
            for i in range(num_q_tiles):
                q_lo = i * block_q
                q_hi = min(q_lo + block_q, sq)
                bq = q_hi - q_lo

                Q_p = q_packed[q_off + q_lo : q_off + q_hi, h]  # [bq, D/2]
                Q_s = q_scale[q_off + q_lo : q_off + q_hi, h]  # [bq, D/32]

                m_i = torch.full((bq,), NEG_INF, dtype=torch.float32, device=device)
                l_i = torch.zeros(bq, dtype=torch.float32, device=device)
                O_i = torch.zeros(bq, d_v, dtype=torch.float32, device=device)

                for j in range(num_kv_tiles):
                    k_lo = j * block_kv
                    k_hi = min(k_lo + block_kv, sk)
                    bkv = k_hi - k_lo

                    if causal and k_lo > (q_hi - 1) + causal_offset:
                        break

                    K_p = k_packed[kv_off + k_lo : kv_off + k_hi, h]  # [bkv, D/2]
                    K_s = k_scale[kv_off + k_lo : kv_off + k_hi, h]  # [bkv, D/32]

                    # (1) mm1: S = Q @ K^T (npu_quant_matmul, FP32 累加树=NPU)
                    # kernel: FixpipeMm1 QF322F16_PRE deqScalar 随路将 fp32 累加
                    # 结果 * softmax_scale 后单次 cast fp16 (单舍入);
                    # golden 复刻该顺序: fp32 乘 scale -> fp16.
                    # NOTE: deqScalar 走 fixpipe 硬件通路, 标量以低位截断的
                    # fp32 参与 (低 13 位 mantissa 清零, & -8192), golden 复刻
                    # 该截断后再乘, 否则边界元素档位翻转 (0183 case).
                    S_ij = _quant_matmul(
                        Q_p, Q_s, K_p, K_s, x2_transposed=True
                    )  # [bq, bkv] FP32
                    S_ij = (S_ij * hw_scale).to(s_torch_dtype)  # scaled 域 fp16

                    # (2) causal mask
                    if causal:
                        q_pos_1d = (
                            torch.arange(q_lo, q_hi, device=device) + causal_offset
                        )
                        k_pos_1d = torch.arange(k_lo, k_hi, device=device)
                        mask = k_pos_1d.unsqueeze(0) > q_pos_1d.unsqueeze(1)
                        S_ij = S_ij.masked_fill(mask, NEG_INF)

                    # (3) P 量化 (snap_local): codes + p_scale + running max
                    p_codes, p_scale, m_new, P_eff = _snap_local_p_quantize_codes(
                        S_ij, m_i, mx_block_size, softmax_scale
                    )
                    P_fp32 = P_eff.float()

                    # (4) alpha = exp2(m_i - m_new), K 网格整数差, 精确 2 的幂
                    # NaN (S 全 -inf 派生) 不钳位, 让 alpha/l/O 自然 NaN 与 kernel 对齐
                    first_tile = torch.isinf(m_i) & (m_i < 0)
                    K_diff_run = m_i - m_new
                    alpha = torch.where(
                        first_tile, torch.zeros_like(K_diff_run), torch.exp2(K_diff_run)
                    )

                    # (5) mm2: O += P @ V (npu_quant_matmul)
                    # K 维 (seqk) pad 到 64 倍数 (scale pairing 硬约束),
                    # P pad 0 (code 0 = +0.0), scale pad 127 (=2^0), 贡献为 0
                    k_phys = ((bkv + 63) // 64) * 64
                    if p_codes.shape[-1] < k_phys:
                        p_codes = F.pad(p_codes, (0, k_phys - p_codes.shape[-1]))
                    if p_scale.shape[-1] < k_phys // mx_block_size:
                        p_scale = F.pad(
                            p_scale,
                            (0, k_phys // mx_block_size - p_scale.shape[-1]),
                            value=127,
                        )
                    V_p = v_packed[kv_off + k_lo : kv_off + k_hi, h]  # [bkv, d_v/2]
                    g_lo = v_gstart + k_lo // mx_block_size
                    g_hi = v_gstart + (k_hi + mx_block_size - 1) // mx_block_size
                    V_s = v_scale[g_lo:g_hi, h]  # [bkv/32, d_v]
                    if V_p.shape[0] < k_phys:
                        V_p = F.pad(V_p, (0, 0, 0, k_phys - V_p.shape[0]))
                    if V_s.shape[0] < k_phys // mx_block_size:
                        V_s = F.pad(
                            V_s,
                            (0, 0, 0, k_phys // mx_block_size - V_s.shape[0]),
                            value=127,
                        )
                    p_packed = pack_fp4_to_uint8(p_codes)  # [bq, k_phys/2]

                    mm2_res = _quant_matmul(
                        p_packed, p_scale, V_p, V_s, x2_transposed=False
                    )  # [bq, d_v] FP32
                    # (6) online 更新 (FP32)
                    l_i = alpha * l_i + P_fp32.sum(dim=-1)
                    O_i = alpha.unsqueeze(-1) * O_i + mm2_res
                    m_i = m_new

                l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
                # 与 kernel 对齐: rsum 含 NaN 时直接除 (输出 NaN), 不兜底 (kernel Div 无保护)
                O_i = O_i / torch.where(torch.isnan(l_i), l_i, l_safe).unsqueeze(-1)
                out[q_off + q_lo : q_off + q_hi, h] = O_i

    return out
