# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import NamedTuple

import torch
import torch.nn.functional as F


CHUNK_SIZE = 64
INVERSE_BLOCK_SIZE = CHUNK_SIZE // 2
# 与 BasicBlock::UseStreamingExact 的算术量选路模型一致。
CUBE_DIM = 128  # Cube 的物理 Dk/Dv，包含窄维 padding。
# S:[Dv,Dk]，alpha=exp(g)，r=beta*(v-alpha*(S@k))；乘和加分别计数。
STREAM_ARITHMETIC_FACTOR = (
    2  # p=S@k，约 2*Dk*Dv。
    + 1  # S_decay=alpha*S，Dk*Dv。
    + 2  # S_next=S_decay+r*k.T，2*Dk*Dv。
    + 2  # o=S_next@(scale*q)，约 2*Dk*Dv。
)
# 仅计主项：省略 residual/exp、scale*q 的实际执行、转换、规约/广播与 padding。
# 256 是固定选路权重，不是硬件吞吐量；模型也省略 chunk 路径的 Vector 计算和搬运/同步。
STREAM_VECTOR_COST = 256
# [M,K]@[K,N] 计 M*N*K 个 MAC；步骤编号对应 arch35.h，C=64、D=128、H=32。
CUBE_MAC_PER_CHUNK = (
    2 * CHUNK_SIZE * CHUNK_SIZE * CUBE_DIM  # [1] KKT/QKT：[2C,D]@[D,C]。
    + 2 * INVERSE_BLOCK_SIZE**3  # [4a/b] tmp=(-L10)@A00，A10=A11@tmp；两次 [H,H] 乘。
    + 2 * CHUNK_SIZE * CHUNK_SIZE * CUBE_DIM  # [5a/b] Kcd/U：两次 [C,C]@[C,D]。
    + 2 * CHUNK_SIZE * CUBE_DIM * CUBE_DIM  # [6a/b] R/O_inter：两次 [C,D]@[D,D]。
    + CHUNK_SIZE * CUBE_DIM * CUBE_DIM  # [7] delta=R16.T@Kg：[D,C]@[C,D]。
    + CHUNK_SIZE * CHUNK_SIZE * CUBE_DIM  # [8] O=O_inter+M@R16：[C,C]@[C,D]。
)
# =5*C*C*D + 3*C*D*D + 2*H**3 =2621440+3145728+65536 =5832704 MAC/chunk。
# 选路时再乘 2，统一为乘/加操作数；展开表达式不改变原有选路边界。


class ChunkPrefix(NamedTuple):
    """与状态无关的 chunk 中间量；P=Nv*chunk_count，下面省略共同前缀 [P]。

    数学符号与 C++ 对应：u=U，kcd=Kcd，
    q_prime=Q'，kg=Kg，masked_qkt=M，chunk_decay=exp(G_last)。
    除 U/decay 外，成员均已在对应发布点舍入 BF16，再升为 FP32 供小算子计算。
    """

    u: torch.Tensor  # U=A@(beta*V), [C,Dv]，FP32 L0C 累加量
    kcd: torch.Tensor  # Kcd=A@(-beta*exp(G)*K), [C,Dk]，已包含负号
    q_prime: torch.Tensor  # Q'=scale*exp(G)*Q, [C,Dk]
    kg: torch.Tensor  # Kg=exp(G_last-G)*K, [C,Dk]
    masked_qkt: torch.Tensor  # M_ij=scale*(Q_i K_j^T)*exp(G_i-G_j), j<=i, [C,C]
    chunk_decay: torch.Tensor  # exp(G_last), 标量


def _bf16(x):
    """显式落地 BF16，再升 FP32 供 NPU 运算；升精度不恢复丢失的位。"""
    return x.bfloat16().float()


def uses_streaming(seq_len, dk, dv, C=CHUNK_SIZE):
    """优化方案：小工作量用纯 AIV 递推，省去 Cube 固定 tile 的计算和搬运。

    比较 7*seq_len*Dk*Dv*256 与 ceil(seq_len/C)*2*5832704；分数不是实测耗时。
    保留 C 关键字兼容现有调用；实际融合路径固定 C=64，空序列只复制初态。
    """
    chunk_count = (seq_len + C - 1) // C
    cube_score = chunk_count * 2 * CUBE_MAC_PER_CHUNK
    streaming_score = seq_len * dk * dv * STREAM_ARITHMETIC_FACTOR * STREAM_VECTOR_COST
    return streaming_score < cube_score or seq_len == 0


def _inclusive_log_decay_scan(log_decay):
    """G_i=sum(g_0..g_i)，输入/输出均为 [P,C] FP32。

    优化方案：使用 Hillis–Steele 倍增顺序
    C = 64, 64 轮 -> 6 轮并行前缀和；
    改成串行 cumsum 会改变浮点加法顺序，clone 保证每轮读取更新前的值。
    """
    g_cum = log_decay.clone()
    for stride in (1, 2, 4, 8, 16, 32):
        g_cum[..., stride:] = g_cum[..., stride:] + g_cum[..., :-stride].clone()
    return g_cum


def _invert_unit_lower_chunk(lower):
    """求 A=(I+L)^(-1) FP32路径求逆，最后转换为bf16。

    输入 lower 是严格下三角 L:[P,C,C] FP32，不含单位阵；C=64。
    输出同 shape，以 FP32 存储舍入 BF16 后的值。

    先在 FP32 中对整个 64x64单位下三角矩阵做一次三角求解，再统一转为 BF16。
    该路径不模拟 kernel将 64x64 拆为两个32x32 对角块、
    且左下块矩阵乘中间量逐次落 BF16 的AIV+AIC 协作过程。
    """
    eye = torch.eye(CHUNK_SIZE, dtype=lower.dtype, device=lower.device)
    inverse = torch.linalg.solve_triangular(eye + lower, eye, upper=False)
    return _bf16(inverse)


def _prepare_chunk_prefix(
    q,  # [P,C,Dk]
    k,  # [P,C,Dk]
    v,  # [P,C,Dv]
    beta,  # [P,C]
    g,  # [P,C]
    scale,
    nv,
    chunk_count,
):
    """
    准备不依赖 S 的前缀。

    kernel优化方案：kernel 用一次合并矩阵乘获得 KKT/QKT，并提前准备下一 chunk；
    这里分别计算两者以展示公式，在与 FIX->UB 相同的位置舍入 BF16。
    """

    kkt = _bf16(k @ k.transpose(-1, -2))  # KKT:[P,C,C]
    qkt = _bf16(q @ k.transpose(-1, -2))  # QKT:[P,C,C]

    g_cum = _inclusive_log_decay_scan(g)  # G:[P,C]
    g_exp = g_cum.exp()
    log_decay_difference = g_cum[..., :, None] - g_cum[..., None, :]
    # 上三角不参与后续计算，先置 0 再 exp，避免无效位置溢出；下三角保留 exp(G_i-G_j)。
    pairwise_decay = torch.tril(log_decay_difference).exp()

    # L_ij=beta_i*KKT_ij*exp(G_i-G_j)，仅 j<i 非零；L/A 均为 [P,C,C]。
    lower = torch.tril(kkt * pairwise_decay, diagonal=-1) * beta[..., None]
    inverse = _invert_unit_lower_chunk(lower)

    raw_kcd = _bf16(k * (-(beta * g_exp))[..., None])
    v_beta = _bf16(v * beta[..., None])
    q_prime = _bf16((q * scale) * g_exp[..., None])
    kg = _bf16(k * (g_cum[..., -1, None] - g_cum).exp()[..., None])
    kcd = _bf16(inverse @ raw_kcd)
    u = inverse @ v_beta  # U 保留 L0C FP32，尚未发布残差 R。
    masked_qkt = _bf16(torch.tril((qkt * scale) * pairwise_decay))

    prefix = u, kcd, q_prime, kg, masked_qkt, g_cum[..., -1].exp()
    # [Nv*chunk_count,...] -> [Nv,chunk_count,...]，下一步按 chunk 取出所有头。
    prefix = ChunkPrefix(
        *(tensor.reshape(nv, chunk_count, *tensor.shape[1:]) for tensor in prefix)
    )
    return prefix


def _stream_sequence(
    q,  # (Si, Nv, Dk)
    k,  # (Si, Nv, Dk)
    v,  # (Si, Nv, Dv)
    beta,  # (Si, Nv)
    state,  # (Nv, Dv, Dk)
    g,  # (Si, Nv)
    scale,
):
    """
    共享 QK 头已展开；state:[Nv,Dv,Dk] 始终为 FP32，输出为 BF16 [Si,Nv,Dv]。
    以下 r/S/o 公式按单个头书写，代码通过 Nv 维并行处理全部 value 头。
    """

    def dot(a, b):
        return (a * b).sum(dim=-1)

    output = torch.empty_like(v)
    # 逐token计算
    for token in range(len(q)):
        decay = g[token].exp()  # alpha:[Nv]
        token_beta = beta[token]  # [Nv]
        token_query = q[token]  # [Nv,Dk]
        token_key = k[token]  # [Nv,Dk]
        token_value = v[token]  # [Nv,Dv]

        # r=beta*(v-alpha*S_prev@k)，沿 Dk 点积：[Nv,Dv,Dk] x [Nv,1,Dk] -> [Nv,Dv]。
        residual = (
            token_value - decay[:, None] * dot(state, token_key[:, None, :])
        ) * token_beta[:, None]
        # S=alpha*S_prev+r[:,None]*k[None,:]，外积广播为 [Nv,Dv,Dk]。
        state = (
            state * decay[:, None, None] + token_key[:, None, :] * residual[..., None]
        )
        # o=S@(scale*q)，读取更新后的 S，结果为 [Nv,Dv]。
        output[token] = dot(state, (token_query * scale)[:, None, :])
    return output.bfloat16(), state


def _pack_sequence_chunks(tensor, chunk_count):
    """[Si,Nv,*features] -> [Nv*chunk_count,C,*features]，先按头再按 chunk 排列。

    仅沿序列维度补 0，Q/K/V需要跳过feature维度
    """
    head_major = tensor.transpose(0, 1)
    # feature维度 无padding + S轴补0
    padding = [0, 0] * (head_major.ndim - 2) + [
        0,
        chunk_count * CHUNK_SIZE - len(tensor),
    ]
    # 补0并拆分chunk
    return F.pad(head_major, padding).reshape(
        -1, CHUNK_SIZE, *head_major.shape[2:]
    )  # [Nv*chunk_count, C, *feature]


def _advance_chunk(prefix, state, state_dtype):
    """
    单个 chunk 的状态相关计算：prefix 各成员以前缀 [Nv] 索引，State:[Nv,Dv,Dk]。

    R=U+Kcd@S^T
    O=Q'@S^T+M@R
    S_next=exp(G_last)*S+R^T@Kg。
    返回 BF16 O:[Nv,C,Dv] 和 FP32 S_next:[Nv,Dv,Dk]。
    """
    # 优化方案：两次状态矩阵乘共享同一份 BF16 S^T；kernel 只搬一次 L1->L0B。
    published_state = _bf16(state).transpose(-1, -2)  # [Nv,Dk,Dv]
    # U 的 FP32 累加量直接叠加 Kcd@S^T，再写 UB
    residual_fp32 = torch.baddbmm(prefix.u, prefix.kcd, published_state)
    # R 转 BF16 进行 Cube 计算
    residual_bf16 = _bf16(residual_fp32)  # [Nv,C,Dv]
    inter_chunk_output = prefix.q_prime @ published_state  # [Nv,C,Dv]
    output = torch.baddbmm(
        inter_chunk_output, prefix.masked_qkt, residual_bf16
    ).bfloat16()
    state_delta = residual_bf16.transpose(-1, -2) @ prefix.kg  # [Nv,Dv,Dk]
    if state_dtype == torch.bfloat16:
        # BF16 接口保留 delta 的既有 BF16 舍入；主状态的 FP32 精度与此独立。
        state_delta = _bf16(state_delta)
    next_state = state * prefix.chunk_decay[:, None, None] + state_delta
    return output, next_state


def _chunk_sequence(
    q,  # (Si, Nv, Dk)
    k,  # (Si, Nv, Dk)
    v,  # (Si, Nv, Dv)
    beta,  # (Si, Nv)
    state,  # (Nv, Dv, Dk)
    g,  # (Si, Nv)
    scale,
    state_dtype,
):
    """按 chunk 顺序递推一个序列；输入 shape 同 _stream_sequence，主状态跨块保留 FP32。"""
    seq_len, value_heads = v.shape[:2]
    chunk_count = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    pad_q, pad_k, pad_v, pad_beta, pad_g = [
        _pack_sequence_chunks(tensor, chunk_count) for tensor in (q, k, v, beta, g)
    ]

    prefix = _prepare_chunk_prefix(
        pad_q, pad_k, pad_v, pad_beta, pad_g, scale, value_heads, chunk_count
    )
    output = torch.empty_like(v, dtype=torch.bfloat16)
    # 逐chunk计算，[Nv, chunk_count, C, D]
    for chunk in range(chunk_count):
        chunk_prefix = ChunkPrefix(*(tensor[:, chunk] for tensor in prefix))
        chunk_output, state = _advance_chunk(chunk_prefix, state, state_dtype)

        token_start = chunk * CHUNK_SIZE
        valid_tokens = min(CHUNK_SIZE, seq_len - token_start)
        output[token_start : token_start + valid_tokens] = chunk_output[
            :, :valid_tokens
        ].transpose(0, 1)
    return output, state


@torch.no_grad()
def chunk_gdn_benchmark_opt(
    q,  # (T, Nk, Dk)
    k,  # (T, Nk, Dk)
    v,  # (T, Nv, Dv)
    beta,  # (T, Nv)
    scale,  # float
    initial_state,  # (B, Nv, Dv, Dk)
    actual_seq_lengths,  # (B,)
    g=None,  # (T, Nv)
    chunk_size=CHUNK_SIZE,
):
    """
    actual_seq_lengths 保存各序列长度（非累积终点），总和等于 T，可含空序列。
    Nv 是 Nk 的整数倍；scale 为调用方给定的 query 缩放，chunk_size 固定为 64。
    返回 output:[T,Nv,Dv] BF16 和 states:[B,Nv,Dv,Dk]（与初态 dtype 一致）。
    """
    if chunk_size != CHUNK_SIZE:
        raise ValueError("融合模型只支持 chunk_size=64")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, beta)):
        raise ValueError("q/k/v/beta 必须是接口要求的 BF16，不在模型入口隐式改写输入")
    if initial_state.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("initial_state 必须是 BF16 或 FP32")
    if q.device.type not in ("npu", "cpu"):
        raise ValueError("benchmark 只支持 NPU 或 CPU tensor")
    if any(t.device != q.device for t in (k, v, beta, initial_state)) or (
        g is not None and g.device != q.device
    ):
        raise ValueError("benchmark 输入必须位于同一设备")
    lengths = (
        actual_seq_lengths.detach().cpu().tolist()
        if torch.is_tensor(actual_seq_lengths)
        else list(actual_seq_lengths)
    )
    if any(length < 0 for length in lengths) or sum(lengths) != len(q):
        raise ValueError("各 batch 长度必须非负且总和等于 T")

    state_dtype = initial_state.dtype
    q, k, v, beta, initial_state = [
        t.detach().float() for t in (q, k, v, beta, initial_state)
    ]
    g = torch.zeros(v.shape[:2], device=v.device) if g is None else g.detach().float()
    nv, nk = v.shape[1], q.shape[1]
    # QK 共享：value_head h 读取 qk_head=floor(h/group_size)，展开后 Q/K:[T,Nv,Dk]。
    head_group_size = nv // nk
    q = q.repeat_interleave(head_group_size, dim=1)
    k = k.repeat_interleave(head_group_size, dim=1)
    output = torch.empty_like(v, dtype=torch.bfloat16)
    states = torch.empty_like(initial_state, dtype=state_dtype)

    token_start = 0
    for batch, seq_len in enumerate(lengths):
        token_end = token_start + seq_len
        state = initial_state[batch].clone()  # [Nv,Dv,Dk]，空序列原样返回
        cur_q, cur_k, cur_v, cur_beta, cur_g = [
            tensor[token_start:token_end] for tensor in (q, k, v, beta, g)
        ]
        if uses_streaming(seq_len, k.shape[-1], v.shape[-1]):
            # 小数据量走逐token计算分支
            sequence_output, state = _stream_sequence(
                cur_q,
                cur_k,
                cur_v,
                cur_beta,
                state,
                cur_g,
                scale,
            )
        else:
            sequence_output, state = _chunk_sequence(
                cur_q,
                cur_k,
                cur_v,
                cur_beta,
                state,
                cur_g,
                scale,
                state_dtype,
            )
        output[token_start:token_end] = sequence_output
        states[batch] = state.to(state_dtype)
        token_start = token_end
    return output, states
