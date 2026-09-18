# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


def stage1_opt(
    query,  # (S, Nk, Dk)
    key,  # (S, Nk, Dk)
    value,  # (S, Nv, Dv)
    g,  # (S, Nv)
    beta,  # (S, Nv)
    scale,
    C,  # chunk size
    v_inner_dtype=torch.bfloat16,
):
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device = query.device

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded_seq_len = (S + C - 1) // C * C
    n_chunks = padded_seq_len // C
    pad_size = padded_seq_len - S

    # Pre-pad all inputs along seq dim (dim 0)
    def pad_seq(t):
        if pad_size > 0:
            pad_pattern = [0, 0] * (t.dim() - 1) + [0, pad_size]
            return F.pad(t, pad_pattern)
        return t

    q_pad = pad_seq(query)  # (Sp, Nv, Dk)
    k_pad = pad_seq(key)  # (Sp, Nv, Dk)
    v_pad = pad_seq(value)  # (Sp, Nv, Dv)
    g_pad = pad_seq(g)  # (Sp, Nv)
    b_pad = pad_seq(beta)  # (Sp, Nv)

    # Reshape to (Nv*nc, C, D) — batch over all heads & chunks, replacing double Python loop
    q_all = q_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dk)
    k_all = k_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dk)
    v_all = v_pad.permute(1, 0, 2).reshape(Nv * n_chunks, C, Dv)
    g_all = g_pad.permute(1, 0).reshape(Nv * n_chunks, C)
    b_all = b_pad.permute(1, 0).reshape(Nv * n_chunks, C)

    # Batched computation
    k_f = k_all.float()
    b_f = b_all.float().unsqueeze(-1)

    kkt = (k_all @ k_all.transpose(-1, -2)).float() * b_f  # (Nv*nc, C, C)
    qkt = q_all @ k_all.transpose(-1, -2)  # (Nv*nc, C, C)

    g_cum = g_all.cumsum(dim=-1)  # (Nv*nc, C)
    g_cum_exp = g_cum.exp()  # (Nv*nc, C)

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)
    diff = g_cum[:, :, None] - g_cum[:, None, :]  # (Nv*nc, C, C)
    attn = (diff * lower).exp() * lower  # (Nv*nc, C, C)

    eye = torch.eye(C, dtype=torch.float32, device=device)
    attn_1 = torch.linalg.solve_triangular(eye + kkt * attn, eye, upper=False)
    attn_1 = attn_1.to(torch.bfloat16)  # (Nv*nc, C, C)

    kg = k_f * (g_cum[:, -1, None] - g_cum).exp()[..., None]  # (Nv*nc, C, Dk)
    k_cumdecay = b_f * g_cum_exp[:, :, None] * (-1) * k_f  # (Nv*nc, C, Dk)

    v_beta = (v_all.float() * b_f).to(torch.bfloat16)  # (Nv*nc, C, Dv)
    q_prime = (q_all.float() * scale * g_cum_exp[:, :, None]).to(
        torch.bfloat16
    )  # (Nv*nc, C, Dk)

    # Match the Vinner accumulator precision of the selected state path.
    # Convert operands before matmul so FP32 results are not rounded through BF16.
    v_inner = attn_1.to(v_inner_dtype) @ v_beta.to(v_inner_dtype)  # (Nv*nc, C, Dv)
    k_cumdecay = attn_1 @ k_cumdecay.to(torch.bfloat16)  # (Nv*nc, C, Dk)
    kg = kg.to(torch.bfloat16)  # (Nv*nc, C, Dk)

    # Reshape back to (Nv, Sp, ...)
    g_cum = g_cum.reshape(Nv, padded_seq_len)
    k_cumdecay = k_cumdecay.reshape(Nv, padded_seq_len, Dk)
    v_inner = v_inner.reshape(Nv, padded_seq_len, Dv)
    q_prime = q_prime.reshape(Nv, padded_seq_len, Dk)
    kg = kg.reshape(Nv, padded_seq_len, Dk)
    qkt = qkt.reshape(Nv, padded_seq_len, C)

    return g_cum, k_cumdecay, v_inner, q_prime, kg, qkt


def stage2_opt(
    q_prime,  # (Nv, Sp, Dk)
    v_inner,  # (Nv, Sp, Dv)
    g_cum,  # (Nv, Sp)
    k_cumdecay,  # (Nv, Sp, Dk)
    state,  # (Nv, Dv, Dk)
    kg,  # (Nv, Sp, Dk)
    C,  # chunk size
):
    Nv, Sp, Dv = v_inner.shape
    _, _, Dk = q_prime.shape
    n_chunks = Sp // C
    device = q_prime.device

    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=device)

    # Batch over Nv, loop only over chunks (sequential state dependency between chunks)
    cur_state = state  # (Nv, Dv, Dk)
    for idx in range(n_chunks):
        s = idx * C
        e = s + C

        qg = q_prime[:, s:e, :]  # (Nv, C, Dk)
        vi = v_inner[:, s:e, :]  # (Nv, C, Dv)
        gc = g_cum[:, s:e]  # (Nv, C)
        kcd = k_cumdecay[:, s:e, :]  # (Nv, C, Dk)
        kgc = kg[:, s:e, :]  # (Nv, C, Dk)

        # Batched stage2 computation over all heads
        state_f = cur_state.float()  # (Nv, Dv, Dk)
        state_fT = state_f.transpose(-1, -2)  # (Nv, Dk, Dv)

        attn_inter_chunk = (qg.float() @ state_fT).to(torch.bfloat16)  # (Nv, C, Dv)
        v_prime = (kcd.float() @ state_fT).to(v_inner.dtype)  # (Nv, C, Dv)
        # FP32 Vinner/Vprime are summed before the single BF16 cast used by stage3.
        v_new_chunk = (vi + v_prime).to(torch.bfloat16)  # (Nv, C, Dv)

        state_out = v_new_chunk.transpose(-1, -2) @ kgc  # (Nv, Dv, Dk)
        decay = gc.exp()[:, -1][:, None, None]  # (Nv, 1, 1)
        state_old = (state_f * decay).to(torch.bfloat16)  # (Nv, Dv, Dk)
        cur_state = state_old + state_out  # (Nv, Dv, Dk)

        attn_inter[:, s:e, :] = attn_inter_chunk
        v_new[:, s:e, :] = v_new_chunk

    final_state = cur_state.to(torch.bfloat16)
    return final_state, attn_inter, v_new


def stage3_opt(
    qkt,  # (Nv, Sp, C)
    scale,  # float
    g_cum,  # (Nv, Sp)
    attn_inter,  # (Nv, Sp, Dv)
    v_new,  # (Nv, Sp, Dv)
    C,  # chunk size
):
    Nv, Sp, Dv = attn_inter.shape
    assert Sp % C == 0
    n_chunks = Sp // C
    device = qkt.device

    # Reshape all inputs to (Nv*nc, C, ...) — batch over all heads & chunks
    qkt_all = qkt.reshape(Nv * n_chunks, C, C)
    g_all = g_cum.reshape(Nv * n_chunks, C)
    ai_all = attn_inter.reshape(Nv * n_chunks, C, Dv)
    vn_all = v_new.reshape(Nv * n_chunks, C, Dv)

    # Batched computation (replaces double loop over Nv and chunks)
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    diff = g_all[:, :, None] - g_all[:, None, :]  # (Nv*nc, C, C)
    decay = (diff * lower).exp() * lower.float()  # (Nv*nc, C, C)
    masked_qkt = qkt_all.float() * scale * decay  # (Nv*nc, C, C)
    attn_inner = masked_qkt.to(torch.bfloat16) @ vn_all  # (Nv*nc, C, Dv)
    core_attn_out = (ai_all + attn_inner).to(torch.bfloat16)  # (Nv*nc, C, Dv)

    # Reshape back: (Nv*nc, C, Dv) -> (Nv, Sp, Dv) -> (Sp, Nv, Dv)
    attn_out = core_attn_out.reshape(Nv, Sp, Dv).permute(1, 0, 2)

    return attn_out


def chunk_gdn_benchmark_opt(
    query,  # (T, Nk, Dk)
    key,  # (T, Nk, Dk)
    value,  # (T, Nv, Dv)
    beta,  # (T, Nv)
    scale,  # float
    initial_state,  # (B, Nv, Dv, Dk)
    actual_seq_lengths,  # (B,)
    g=None,  # (T, Nv)
    chunk_size=64,
    v_inner_dtype=torch.bfloat16,
):
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device = query.device

    if g is None:
        g = torch.zeros((T, Nv), dtype=torch.float32, device=device)
    attn_out = torch.empty((T, Nv, Dv), dtype=torch.bfloat16, device=device)
    final_state = torch.empty_like(initial_state).to(torch.bfloat16)

    start = 0
    C = chunk_size
    for bid in range(B):
        cur_state = initial_state[bid].clone()
        S = actual_seq_lengths[bid]
        end = start + S

        g_cum, k_cum_decay, v_inner, q_prime, kg, qkt = stage1_opt(
            query[start:end],
            key[start:end],
            value[start:end],
            g[start:end],
            beta[start:end],
            scale,
            C,
            v_inner_dtype=v_inner_dtype,
        )

        cur_state, attn_inter, v_new = stage2_opt(
            q_prime, v_inner, g_cum, k_cum_decay, cur_state, kg, C
        )
        final_state[bid] = cur_state.to(torch.bfloat16)
        attn_out_padded = stage3_opt(qkt, scale, g_cum, attn_inter, v_new, C)

        attn_out[start:end, ...] = attn_out_padded[:S]
        start = end

    return attn_out, final_state
