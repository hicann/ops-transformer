# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""CPU references and native TTK TestSpec. See ttk/README.md for the reference implementation source."""

import dataclasses

import torch
import torch.nn.functional as F
from kda_ttk_ops import load_asset


def effective(c):
    return load_asset("inputs").effective(c)


def sequences(*args, **kwargs):
    return load_asset("inputs").sequences(*args, **kwargs)


CHUNK_SIZE = 64
REFERENCE_BLOCK_SIZE = 16
LOW_DTYPE = torch.bfloat16
SUPPORTED_HEAD_DIM = 128
PERSISTENT_TEMPLATE_ID = 2  # metadata ABI 4


@dataclasses.dataclass
class KdaInputs:
    K: torch.Tensor  # (B, Nk, S, D) bf16
    V: torch.Tensor  # (B, Nv, S, D) bf16
    Q: torch.Tensor  # (B, Nk, S, D) bf16
    beta_brc: torch.Tensor  # (B, Nv, S, 1) raw BF16 logits
    g: torch.Tensor  # (B, Nv, S, D) raw BF16 logits
    initial_state: torch.Tensor  # (B, Nv, Dk, Dv) fp32
    scale_value: float
    A_log: torch.Tensor  # (Nv,) fp32
    dt_bias: torch.Tensor  # (Nv, D) fp32


def _chunk_major(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    chunks = seq_len // CHUNK_SIZE
    return x.reshape(batch, heads, chunks, CHUNK_SIZE, dim).contiguous()


@dataclasses.dataclass
class _StageOneGolden:
    Gamma: torch.Tensor
    K_decayed_beta: torch.Tensor
    K_decayed: torch.Tensor
    Q_decayed: torch.Tensor
    inverse: torch.Tensor
    U_pre: torch.Tensor
    W: torch.Tensor
    Mqk: torch.Tensor
    K_restored: torch.Tensor
    gamma_C: torch.Tensor


@dataclasses.dataclass
class _StageTwoGolden:
    U: torch.Tensor
    O_state: torch.Tensor
    S: torch.Tensor
    H: torch.Tensor


@dataclasses.dataclass
class _StageThreeGolden:
    output: torch.Tensor


@dataclasses.dataclass
class _ChainGolden:
    stage1: _StageOneGolden
    stage2: _StageTwoGolden
    stage3: _StageThreeGolden


def _to_fp16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """近似 Stage1 FP16 packed/NZ L1 交接。"""
    return x.to(torch.float16).to(torch.float32)


def _to_bf16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """近似 BF16 GM 读写精度损失。"""
    return x.to(LOW_DTYPE).to(torch.float32)


def _reference_lower_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    cumulative_decay: torch.Tensor,
    *,
    diagonal: int,
    cast_operand=None,
    safe_diagonal: bool = True,
) -> torch.Tensor:
    """Compute causal QK/KKT tiles with 16-token relative-decay references."""
    result = torch.zeros(
        *left.shape[:-1],
        right.shape[-2],
        dtype=torch.float32,
        device=left.device,
    )
    cast = (lambda x: x) if cast_operand is None else cast_operand
    for row_start in range(0, CHUNK_SIZE, REFERENCE_BLOCK_SIZE):
        row_end = row_start + REFERENCE_BLOCK_SIZE
        reference = cumulative_decay[..., row_start : row_start + 1, :]
        left_rows = cast(
            left[..., row_start:row_end, :]
            * (cumulative_decay[..., row_start:row_end, :] - reference).exp()
        )
        for col_start in range(0, row_end, REFERENCE_BLOCK_SIZE):
            col_end = col_start + REFERENCE_BLOCK_SIZE
            if safe_diagonal and col_start == row_start:
                for row in range(row_start, row_end):
                    relative_decay = (
                        cumulative_decay[..., row : row + 1, :]
                        - cumulative_decay[..., row_start : row + 1, :]
                    ).exp()
                    right_cols = cast(
                        right[..., row_start : row + 1, :] * relative_decay
                    )
                    result[..., row : row + 1, row_start : row + 1] = cast(
                        left[..., row : row + 1, :]
                    ) @ right_cols.transpose(-1, -2)
                continue
            right_cols = cast(
                right[..., col_start:col_end, :]
                * (reference - cumulative_decay[..., col_start:col_end, :]).exp()
            )
            result[..., row_start:row_end, col_start:col_end] = (
                left_rows @ right_cols.transpose(-1, -2)
            )

    return result * torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=left.device),
        diagonal=diagonal,
    )


def _npu_neumann_inverse_from_negative_t(negative_t: torch.Tensor) -> torch.Tensor:
    """从 NPU ``-T`` 交接重建驻留逆矩阵（packed64：16→32→64 块下三角组合）。"""
    inverse = torch.zeros_like(negative_t, dtype=torch.float32)
    for block_idx in range(0, CHUNK_SIZE, 16):
        block = _to_fp16_to_fp32(
            negative_t[block_idx : block_idx + 16, block_idx : block_idx + 16]
        )
        diag_inv = torch.eye(16, dtype=torch.float32, device=negative_t.device) + block
        diag_power2 = block @ block
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power2)
        diag_power4 = _to_fp16_to_fp32(diag_power2) @ _to_fp16_to_fp32(diag_power2)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power4)
        diag_power8 = _to_fp16_to_fp32(diag_power4) @ _to_fp16_to_fp32(diag_power4)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power8)
        inverse[block_idx : block_idx + 16, block_idx : block_idx + 16] = diag_inv

    for block_idx in range(0, CHUNK_SIZE, 32):
        even = slice(block_idx, block_idx + 16)
        odd = slice(block_idx + 16, block_idx + 32)
        odd_lower_even = _to_fp16_to_fp32(negative_t[odd, even])
        lower_tmp = odd_lower_even @ _to_fp16_to_fp32(inverse[even, even])
        inverse[odd, even] = _to_fp16_to_fp32(inverse[odd, odd]) @ _to_fp16_to_fp32(
            lower_tmp
        )

    even32 = slice(0, 32)
    odd32 = slice(32, 64)
    odd32_lower_even32 = _to_fp16_to_fp32(negative_t[odd32, even32])
    lower32_tmp = odd32_lower_even32 @ _to_fp16_to_fp32(inverse[even32, even32])
    inverse[odd32, even32] = _to_fp16_to_fp32(inverse[odd32, odd32]) @ _to_fp16_to_fp32(
        lower32_tmp
    )
    return _to_bf16_to_fp32(inverse)


def _stage1_cpu(K, V, Q, beta_brc, g, *, scale_value) -> _StageOneGolden:
    """Stage1 CPU 公式（全 fp32）：生成不依赖状态的 chunk 工作区。"""
    if V.shape[1] != K.shape[1]:  # GQA repeat 到 Nv
        repeat = V.shape[1] // K.shape[1]
        K = K.repeat_interleave(repeat, dim=1)
        Q = Q.repeat_interleave(repeat, dim=1)
    K_i = _chunk_major(K).to(torch.float32)
    V_i = _chunk_major(V).to(torch.float32)
    Q_i = _chunk_major(Q).to(torch.float32)
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    cumulative_decay = torch.cumsum(g_i, dim=-2)
    gamma = torch.exp(cumulative_decay)
    K_decayed = gamma * K_i
    K_decayed_beta = beta * K_decayed
    Q_decayed = gamma * Q_i * scale_value

    T = _reference_lower_matmul(beta * K_i, K_i, cumulative_decay, diagonal=-1)

    identity = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=K.device)
    inverse = torch.linalg.solve_triangular(
        identity + T, identity.expand_as(T), upper=False, unitriangular=True
    )
    M = inverse * beta.transpose(-1, -2)
    U_pre = M @ V_i
    W = M @ K_decayed
    Mqk = _reference_lower_matmul(Q_i * scale_value, K_i, cumulative_decay, diagonal=0)
    gamma_C = gamma[..., -1, :, None].contiguous()
    K_restored = K_i * (cumulative_decay[..., -1:, :] - cumulative_decay).exp()
    return _StageOneGolden(
        gamma,
        K_decayed_beta,
        K_decayed,
        Q_decayed,
        inverse,
        U_pre,
        W,
        Mqk,
        K_restored,
        gamma_C,
    )


def _stage1_npu(
    K, V, Q, beta_brc, g, *, scale_value, safe_diagonal=True
) -> _StageOneGolden:
    """Stage1 NPU-oriented golden：显式 batch/head/chunk 循环 + bf16/fp16 交接。"""
    if V.shape[1] != K.shape[1]:
        rep = V.shape[1] // K.shape[1]
        K = K.repeat_interleave(rep, dim=1)
        Q = Q.repeat_interleave(rep, dim=1)
    K_i = _to_bf16_to_fp32(_chunk_major(K))
    V_i = _chunk_major(V).to(torch.float32)
    Q_i = _to_bf16_to_fp32(_chunk_major(Q))
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    batch, heads, chunks, _, dim = K_i.shape
    gamma = torch.empty_like(K_i)
    K_decayed_beta = torch.empty_like(K_i)
    K_decayed = torch.empty_like(K_i)
    Q_decayed = torch.empty_like(K_i)
    inverse_out = torch.empty(
        batch,
        heads,
        chunks,
        CHUNK_SIZE,
        CHUNK_SIZE,
        dtype=torch.float32,
        device=K.device,
    )
    U_pre = torch.empty_like(K_i)
    W = torch.empty_like(K_i)
    Mqk = torch.empty_like(inverse_out)
    K_restored = torch.empty_like(K_i)
    gamma_C = torch.empty(
        batch, heads, chunks, dim, 1, dtype=torch.float32, device=K.device
    )
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                K_c = K_i[b, h, c]
                V_c = V_i[b, h, c]
                Q_c = Q_i[b, h, c]
                beta_c = beta[b, h, c]
                g_c = g_i[b, h, c]

                g_cumsum = torch.cumsum(g_c, dim=0)  # AIV 全程 fp32
                gamma_c = torch.exp(g_cumsum)
                K_decayed_prepare = gamma_c * K_c
                K_decayed_c = _to_bf16_to_fp32(K_decayed_prepare)
                Q_decayed_c = _to_bf16_to_fp32(gamma_c * Q_c * scale_value)

                T = (
                    _reference_lower_matmul(
                        K_c,
                        K_c,
                        g_cumsum,
                        diagonal=-1,
                        cast_operand=_to_bf16_to_fp32,
                        safe_diagonal=safe_diagonal,
                    )
                    * beta_c
                )
                Mqk_c = _to_bf16_to_fp32(
                    _reference_lower_matmul(
                        Q_c * scale_value,
                        K_c,
                        g_cumsum,
                        diagonal=0,
                        cast_operand=_to_bf16_to_fp32,
                        safe_diagonal=safe_diagonal,
                    )
                )

                negative_t = _to_fp16_to_fp32(-T)
                resident_inv = _npu_neumann_inverse_from_negative_t(negative_t)

                K_decayed_beta_c = _to_bf16_to_fp32(beta_c * K_decayed_prepare)
                V_beta = _to_bf16_to_fp32(beta_c * V_c)
                # Keep the FP32 Cube accumulator for the persistent recurrence.
                U_pre_c = _to_bf16_to_fp32(resident_inv) @ V_beta
                workspace_w = _to_bf16_to_fp32(
                    _to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(-K_decayed_beta_c)
                )
                gamma_c_last = gamma_c[-1].reshape(K_c.shape[-1], 1)
                K_restored_c = _to_bf16_to_fp32(K_c * (g_cumsum[-1:] - g_cumsum).exp())

                gamma[b, h, c] = gamma_c
                K_decayed[b, h, c] = K_decayed_c
                K_decayed_beta[b, h, c] = K_decayed_beta_c
                Q_decayed[b, h, c] = Q_decayed_c
                inverse_out[b, h, c] = resident_inv
                U_pre[b, h, c] = U_pre_c
                W[b, h, c] = -workspace_w
                Mqk[b, h, c] = Mqk_c
                gamma_C[b, h, c] = gamma_c_last
                K_restored[b, h, c] = K_restored_c
    return _StageOneGolden(
        gamma,
        K_decayed_beta,
        K_decayed,
        Q_decayed,
        inverse_out,
        U_pre,
        W,
        Mqk,
        K_restored,
        gamma_C,
    )


def _stage2_cpu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 CPU 公式（全 fp32）：用 stage1 工作区 + 初始 state 递推。"""
    _, _, chunks, _, _ = stage1.U_pre.shape
    U = torch.zeros_like(stage1.U_pre)
    O_state = torch.zeros_like(stage1.U_pre)
    state = initial_state.to(torch.float32).clone()
    H = torch.empty(
        *stage1.U_pre.shape[:3],
        state.shape[-2],
        state.shape[-1],
        dtype=torch.float32,
        device=state.device,
    )
    U_pre = stage1.U_pre.to(torch.float32)
    W = stage1.W.to(torch.float32)
    Q_decayed = stage1.Q_decayed.to(torch.float32)
    K_restored = stage1.K_restored.to(torch.float32)
    gamma_C = stage1.gamma_C.to(torch.float32)
    for c in range(chunks):
        H[:, :, c] = state
        U_c = U_pre[:, :, c] - W[:, :, c] @ state
        O_c = Q_decayed[:, :, c] @ state
        state = gamma_C[:, :, c] * state + K_restored[:, :, c].transpose(-1, -2) @ U_c
        U[:, :, c] = U_c
        O_state[:, :, c] = O_c
    return _StageTwoGolden(U=U, O_state=O_state, S=state, H=H)


def _stage2_npu(
    stage1: _StageOneGolden, initial_state, *, direct_u=False
) -> _StageTwoGolden:
    """Stage2 NPU-oriented golden：显式 chunk 递推 + bf16 交接。"""
    batch, heads, chunks, _, _ = stage1.U_pre.shape
    U = torch.zeros_like(stage1.U_pre, dtype=torch.float32)
    O_state = torch.zeros_like(stage1.U_pre, dtype=torch.float32)
    U_pre = stage1.U_pre if direct_u else _to_bf16_to_fp32(stage1.U_pre)
    W_neg = _to_bf16_to_fp32(-stage1.W)  # workspace 存负 W
    Q_decayed = _to_bf16_to_fp32(stage1.Q_decayed)
    K_restored = _to_bf16_to_fp32(stage1.K_restored)
    gamma_C = stage1.gamma_C.to(torch.float32)
    state = initial_state.to(torch.float32).clone()
    H = torch.empty(
        batch,
        heads,
        chunks,
        state.shape[-2],
        state.shape[-1],
        dtype=torch.float32,
        device=state.device,
    )
    for b in range(batch):
        for h in range(heads):
            state_bh = state[b, h]
            for c in range(chunks):
                H[b, h, c] = state_bh
                w_s = _to_bf16_to_fp32(W_neg[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                U_c = _to_bf16_to_fp32(U_pre[b, h, c] + w_s)
                O_c = _to_bf16_to_fp32(Q_decayed[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                kr_u = _to_bf16_to_fp32(
                    K_restored[b, h, c].transpose(-1, -2)
                ) @ _to_bf16_to_fp32(U_c)
                # Current DSL retains the recurrent accumulator in FP32.
                scaled_state = gamma_C[b, h, c] * state_bh
                state_bh = scaled_state + kr_u
                U[b, h, c] = U_c
                O_state[b, h, c] = O_c
            state[b, h] = state_bh
    return _StageTwoGolden(U=U, O_state=O_state, S=state, H=H)


def _stage3_cpu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 CPU 公式：O = O_state + Mqk @ U。"""
    O_i = stage2.O_state.to(torch.float32) + stage1.Mqk.to(torch.float32) @ stage2.U.to(
        torch.float32
    )
    batch, heads, chunks, chunk, dim = O_i.shape
    return _StageThreeGolden(
        output=O_i.reshape(batch, heads, chunks * chunk, dim).contiguous()
    )


def _stage3_npu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 NPU-oriented golden：显式 chunk 合成 + bf16 交接。"""
    batch, heads, chunks, chunk, dim = stage2.O_state.shape
    O_i = torch.empty_like(stage2.O_state, dtype=torch.float32)
    # Q@state and Mqk@U accumulate into one FP32 L0C tile.
    O_state = stage2.O_state.float()
    Mqk = _to_bf16_to_fp32(stage1.Mqk)
    U = _to_bf16_to_fp32(stage2.U)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                mqk_u = _to_bf16_to_fp32(Mqk[b, h, c]) @ _to_bf16_to_fp32(U[b, h, c])
                O_i[b, h, c] = _to_bf16_to_fp32(O_state[b, h, c] + mqk_u)
    return _StageThreeGolden(
        output=O_i.reshape(batch, heads, chunks * chunk, dim).contiguous()
    )


def _chain_cpu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_cpu(
        inputs.K,
        inputs.V,
        inputs.Q,
        inputs.beta_brc,
        inputs.g,
        scale_value=inputs.scale_value,
    )
    s2 = _stage2_cpu(s1, inputs.initial_state)
    s3 = _stage3_cpu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def _chain_npu(
    inputs: KdaInputs, *, safe_diagonal=True, direct_u=False
) -> _ChainGolden:
    s1 = _stage1_npu(
        inputs.K,
        inputs.V,
        inputs.Q,
        inputs.beta_brc,
        inputs.g,
        scale_value=inputs.scale_value,
        safe_diagonal=safe_diagonal,
    )
    s2 = _stage2_npu(s1, inputs.initial_state, direct_u=direct_u)
    s3 = _stage3_npu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def empty():
    return torch.empty(0, dtype=torch.float32)


def reference(inputs, c, kind):
    f = effective(c)
    direct_u = False
    if kind == "npu_chunk":
        template_word = load_asset("inputs").HEADER_FIELDS.index("template")
        direct_u = inputs[5][template_word].item() == PERSISTENT_TEMPLATE_ID
    q, k, v, g, beta = [x.detach().cpu() for x in inputs[:5]]
    state, a, bias = [x.detach().cpu() for x in inputs[6:9]]
    # Raw input values/dtypes are preserved. Only npu_chunk models device casts.
    lists = [list(sequences(x, c)) for x in (q, k, v, g, beta)]
    result = [[] for _ in range(16)]
    for b, n in enumerate(c["valid_lengths"]):
        qi, ki, vi, gi, bi = [xs[b].float() for xs in lists]
        qr = (qi.square().sum(-1) + 1e-6).rsqrt()
        kr = (ki.square().sum(-1) + 1e-6).rsqrt()
        if f["use_qk_l2norm_in_kernel"]:
            qi = qi * qr.unsqueeze(-1)
            ki = ki * kr.unsqueeze(-1)
        if kind == "npu_chunk":
            gi = gi.bfloat16().float()  # Beta activation retains the actual FP32 input.
        if f["use_gate_in_kernel"]:
            gb = bias if c["dt_bias_mode"] != "none" else torch.zeros_like(bias)
            logits = gi + gb[None, :, None, :]
            alpha = a.exp()[None, :, None, None]
            bound = f["lower_bound"]
            gi = (
                -alpha * F.softplus(logits)
                if bound is None
                else bound * torch.sigmoid(alpha * logits)
            )
        if f["use_beta_sigmoid_in_kernel"]:
            bi = bi.sigmoid() * (2 if f["allow_neg_eigval"] else 1)
        if not c["inference"] and f["disable_recompute"]:
            for chunk in gi.split(CHUNK_SIZE, dim=2):
                result[10].append(chunk.cumsum(dim=2).flatten())
        beta_activation = bi
        h0 = (
            state[b : b + 1].transpose(-1, -2)
            if c["initial_state_mode"] == "random"
            else torch.zeros_like(state[b : b + 1])
        )
        pad = (-n) % 64
        args = KdaInputs(
            K=F.pad(ki, (0, 0, 0, pad)),
            V=F.pad(vi, (0, 0, 0, pad)),
            Q=F.pad(qi, (0, 0, 0, pad)),
            beta_brc=F.pad(bi.unsqueeze(-1), (0, 0, 0, pad)),
            g=F.pad(gi, (0, 0, 0, pad)),
            initial_state=h0,
            scale_value=qi.size(-1) ** -0.5 if f["scale"] is None else f["scale"],
            A_log=a,
            dt_bias=bias,
        )
        if kind == "cpu_chunk":
            chain = _chain_cpu(args)
        else:
            # The native bounded-gate reference uses a shared 16-row origin.
            # Unbounded/preprocessed gates need causal differences to avoid
            # overflow in the unused upper triangle before it is masked.
            bounded_gate = f["use_gate_in_kernel"] and f["lower_bound"] is not None
            chain = _chain_npu(args, safe_diagonal=not bounded_gate, direct_u=direct_u)
        result[0].append(chain.stage3.output[:, :, :n].flatten())
        if f["output_final_state"]:
            result[1].append(chain.stage2.S.transpose(-1, -2).flatten())
        if c["inference"] and f["return_intermediate_states"]:
            result[2].append(chain.stage2.H[0].permute(1, 0, 3, 2).flatten())
        if not c["inference"]:
            stage = chain.stage1
            for slot, matrix in ((3, stage.Mqk), (4, stage.inverse)):
                for ci in range(matrix.size(2)):
                    valid = min(CHUNK_SIZE, n - ci * CHUNK_SIZE)
                    result[slot].append(matrix[0, :, ci, :valid, :valid].flatten())
            if f["disable_recompute"]:
                q_heads = _chunk_major(args.Q).repeat_interleave(
                    vi.size(1) // qi.size(1), dim=1
                )
                # Match the existing _saved_golden reference: BF16 normalized Q
                # and BF16 public qg. Keep the cpu_chunk formula in FP32.
                if kind == "npu_chunk":
                    q_heads = _to_bf16_to_fp32(q_heads)
                qg = stage.Gamma * q_heads
                if kind == "npu_chunk":
                    qg = _to_bf16_to_fp32(qg)
                # The public U_pre cache is BF16 even on the persistent route.
                saved_u = (
                    _to_bf16_to_fp32(stage.U_pre)
                    if kind == "npu_chunk"
                    else stage.U_pre
                )
                for slot, cache in enumerate(
                    (stage.W, saved_u, qg, stage.K_restored, chain.stage2.U), 5
                ):
                    result[slot].append(cache.flatten(2, 3)[:, :, :n].flatten())
        if not c["inference"] and f["use_qk_l2norm_in_kernel"]:
            for j, x in zip(
                (11, 12, 13, 14), (qi.bfloat16().float(), qr, ki.bfloat16().float(), kr)
            ):
                result[j].append(x.flatten())
        if not c["inference"] and f["use_beta_sigmoid_in_kernel"]:
            result[15].append(beta_activation.transpose(1, 2).flatten())
    return [torch.cat(parts) if parts else empty() for parts in result]


def select_reference(outputs, c, *, forward=False):
    """Select by the requested contract, never by the DUT's observed arity."""
    slots = load_asset("compare").numerical_slots(c, forward=forward)
    return [outputs[slot] for slot in slots]


def golden(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    **kwargs,
):
    inputs = [
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
    ]
    c = load_asset("inputs").configuration(inputs, kwargs)
    if c["pending"]:
        raise RuntimeError(c["pending"])
    if c["suite"] in (
        "meta",
        "meta_negative",
        "negative",
        "metadata",
        "producer",
        "whitebox",
    ):
        return [torch.ones(1, dtype=torch.bool)]
    if c["suite"] == "performance":
        raise RuntimeError(
            "Performance CSV requires golden_api=disable; no long CPU golden"
        )
    # TTK compares forward results; the bool also requires unified-entry L2 and consistency.
    return select_reference(reference(inputs, c, "cpu_chunk"), c, forward=True) + [
        torch.ones(1, dtype=torch.bool)
    ]


class CpuSimulation:
    """CPU baseline for cross_check; not an independent GPU implementation."""

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        **kwargs,
    ):
        inputs = [
            q,
            k,
            v,
            g,
            beta,
            metadata,
            initial_state,
            A_log,
            dt_bias,
            cu_seqlens,
            seqused,
        ]
        c = load_asset("inputs").configuration(inputs, kwargs)
        if c["suite"] != "accuracy":
            return golden(*inputs, **kwargs)
        return select_reference(reference(inputs, c, "npu_chunk"), c, forward=True) + [
            torch.ones(1, dtype=torch.bool)
        ]


class FlashKdaTestSpec:
    """Native TestSpec, independent of TTK internal APIs."""

    golden = staticmethod(golden)
    third_party = {"torch": CpuSimulation}
    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bool": {"standard": "binary_equal"},
    }

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return load_asset("inputs").customize_inputs(*args, **kwargs)


__spec__ = {"kda_ttk_ops.flash_kda": "FlashKdaTestSpec"}
