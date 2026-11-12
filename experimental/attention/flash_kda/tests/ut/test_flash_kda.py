# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# Keep compact call argument groups consistent with the original FlashKDA samples.

"""Unified FlashKDA precision, return contracts, and forward compatibility."""

import dataclasses
import logging
import os

import pytest
import torch
import torch.nn.functional as F
from experimental.attention.flash_kda.op_dsl import flash_kda as _DYNAMIC_KDA
from experimental.attention.flash_kda.op_dsl import flash_kda_staged as _DYNAMIC_STAGED
from experimental.attention.flash_kda.op_dsl import flash_kda_persistent as _DYNAMIC_PERSISTENT
from experimental.attention.flash_kda.op_dsl import flash_kda_metadata as _DYNAMIC_METADATA

LOGGER = logging.getLogger(__name__)

CHUNK_SIZE = 64
REFERENCE_BLOCK_SIZE = 16
LOW_DTYPE = torch.bfloat16
SUPPORTED_HEAD_DIM = 128


@dataclasses.dataclass
class KdaInputs:
    k: torch.Tensor  # (B, Nk, S, D) bf16
    v: torch.Tensor  # (B, Nv, S, D) bf16
    q: torch.Tensor  # (B, Nk, S, D) bf16
    beta_brc: torch.Tensor  # (B, Nv, S, 1) raw BF16 logits
    g: torch.Tensor  # (B, Nv, S, D) raw BF16 logits
    initial_state: torch.Tensor  # (B, Nv, Dk, Dv) fp32
    scale_value: float
    a_log: torch.Tensor  # (Nv,) fp32
    dt_bias: torch.Tensor  # (Nv, D) fp32
    lower_bound: float


def _chunk_major(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    chunks = seq_len // CHUNK_SIZE
    return x.reshape(batch, heads, chunks, CHUNK_SIZE, dim).contiguous()


def make_inputs_generalized(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    *,
    nk: int | None = None,
    seed: int = 0,
    lower_bound: float = -1.0,
    v_scale: float = 1.0,
    b_spread: float = 1.0,
    init_scale: float = 0.1,
) -> KdaInputs:
    """Build BF16 Q/K/V and raw gate/beta logits with a seeded CPU generator."""
    if dim != SUPPORTED_HEAD_DIM:
        raise ValueError(f"flash_kda 当前只支持 D={SUPPORTED_HEAD_DIM}，收到 D={dim}")
    nk = heads if nk is None else nk
    if heads % nk != 0:
        raise ValueError(f"GQA requires Nv % Nk == 0，收到 Nv={heads}, Nk={nk}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = F.normalize(torch.randn(batch, nk, seq_len, dim, generator=gen), p=2, dim=-1)
    q = F.normalize(torch.randn(batch, nk, seq_len, dim, generator=gen), p=2, dim=-1)
    v = torch.randn(batch, heads, seq_len, dim, generator=gen) * float(v_scale)
    g = torch.randn(batch, heads, seq_len, dim, generator=gen).to(LOW_DTYPE)
    beta = (torch.randn(batch, heads, seq_len, 1, generator=gen) * float(b_spread)).to(LOW_DTYPE)
    initial_state = torch.rand(batch, heads, dim, dim, generator=gen) * float(init_scale)

    return KdaInputs(
        k=k.to(LOW_DTYPE), v=v.to(LOW_DTYPE), q=q.to(LOW_DTYPE), beta_brc=beta, g=g, initial_state=initial_state,
        scale_value=dim**-0.5, a_log=torch.linspace(-1.0, 0.5, heads, dtype=torch.float32),
        dt_bias=torch.randn(heads, dim, generator=gen, dtype=torch.float32), lower_bound=float(lower_bound),
    )


@dataclasses.dataclass
class _StageOneGolden:
    gamma: torch.Tensor
    k_decayed_beta: torch.Tensor
    k_decayed: torch.Tensor
    q_decayed: torch.Tensor
    t: torch.Tensor
    u_pre: torch.Tensor
    w: torch.Tensor
    mqk: torch.Tensor
    k_restored: torch.Tensor
    chunk_decay: torch.Tensor


@dataclasses.dataclass
class _StageTwoGolden:
    u: torch.Tensor
    o_state: torch.Tensor
    s: torch.Tensor
    state_history: torch.Tensor


@dataclasses.dataclass
class _StageThreeGolden:
    o: torch.Tensor


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
    safe_diagonal: bool = False,
    shared_score_reference: bool = False,
) -> torch.Tensor:
    """Compute causal QK/KKT tiles with the selected operand-rounding path."""
    result = torch.zeros(*left.shape[:-1], right.shape[-2], dtype=torch.float32, device=left.device)
    cast = (lambda x: x) if cast_operand is None else cast_operand
    for row_start in range(0, CHUNK_SIZE, REFERENCE_BLOCK_SIZE):
        row_end = row_start + REFERENCE_BLOCK_SIZE
        # Staged's shared score panels center each 32-row group at row 16/48.
        # Persistent and the staged fallback reference each 16-row block start.
        # The reference cancels algebraically, but not after BF16 operand casts.
        reference_index = (row_start // 32) * 32 + 16 if shared_score_reference else row_start
        reference_row = slice(reference_index, reference_index + 1)
        reference = cumulative_decay[..., reference_row, :]
        left_rows = cast(
            left[..., row_start:row_end, :] * (cumulative_decay[..., row_start:row_end, :] - reference).exp(),
        )
        for col_start in range(0, row_end, REFERENCE_BLOCK_SIZE):
            col_end = col_start + REFERENCE_BLOCK_SIZE
            if safe_diagonal and col_start == row_start:
                for row in range(row_start, row_end):
                    current_row = slice(row, row + 1)
                    causal_rows = slice(row_start, row + 1)
                    relative_decay = (
                        cumulative_decay[..., current_row, :]
                        - cumulative_decay[..., causal_rows, :]
                    ).exp()
                    right_cols = cast(right[..., causal_rows, :] * relative_decay)
                    result[..., current_row, causal_rows] = cast(left[..., current_row, :]) @ right_cols.transpose(
                        -1, -2,
                    )
                continue
            right_cols = cast(
                right[..., col_start:col_end, :] * (reference - cumulative_decay[..., col_start:col_end, :]).exp(),
            )
            result[..., row_start:row_end, col_start:col_end] = (
                left_rows @ right_cols.transpose(-1, -2)
            )

    return result * torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=left.device), diagonal=diagonal,
    )


def _npu_neumann_inverse_from_negative_t(negative_t: torch.Tensor) -> torch.Tensor:
    """从 NPU ``-T`` 交接重建驻留逆矩阵（packed64：16→32→64 块下三角组合）。"""
    inverse = torch.zeros_like(negative_t, dtype=torch.float32)
    for block_idx in range(0, CHUNK_SIZE, 16):
        block_slice = slice(block_idx, block_idx + 16)
        block = _to_fp16_to_fp32(negative_t[block_slice, block_slice])
        diag_inv = torch.eye(16, dtype=torch.float32, device=negative_t.device) + block
        diag_power2 = block @ block
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power2)
        diag_power4 = _to_fp16_to_fp32(diag_power2) @ _to_fp16_to_fp32(diag_power2)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power4)
        diag_power8 = _to_fp16_to_fp32(diag_power4) @ _to_fp16_to_fp32(diag_power4)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power8)
        inverse[block_slice, block_slice] = diag_inv

    for block_idx in range(0, CHUNK_SIZE, 32):
        even = slice(block_idx, block_idx + 16)
        odd = slice(block_idx + 16, block_idx + 32)
        odd_lower_even = _to_fp16_to_fp32(negative_t[odd, even])
        lower_tmp = odd_lower_even @ _to_fp16_to_fp32(inverse[even, even])
        inverse[odd, even] = _to_fp16_to_fp32(inverse[odd, odd]) @ _to_fp16_to_fp32(lower_tmp)

    even32 = slice(0, 32)
    odd32 = slice(32, 64)
    odd32_lower_even32 = _to_fp16_to_fp32(negative_t[odd32, even32])
    lower32_tmp = odd32_lower_even32 @ _to_fp16_to_fp32(inverse[even32, even32])
    inverse[odd32, even32] = _to_fp16_to_fp32(inverse[odd32, odd32]) @ _to_fp16_to_fp32(lower32_tmp)
    return _to_bf16_to_fp32(inverse)


def _stage1_cpu(k, v, q, beta_brc, g, *, scale_value) -> _StageOneGolden:
    """Stage1 CPU 公式（全 fp32）：生成不依赖状态的 chunk 工作区。"""
    if v.shape[1] != k.shape[1]:  # GQA repeat 到 Nv
        repeat = v.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        q = q.repeat_interleave(repeat, dim=1)
    k_i = _chunk_major(k).to(torch.float32)
    v_i = _chunk_major(v).to(torch.float32)
    q_i = _chunk_major(q).to(torch.float32)
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    cumulative_decay = torch.cumsum(g_i, dim=-2)
    gamma = torch.exp(cumulative_decay)
    k_decayed = gamma * k_i
    k_decayed_beta = beta * k_decayed
    q_decayed = gamma * q_i * scale_value

    t = _reference_lower_matmul(beta * k_i, k_i, cumulative_decay, diagonal=-1)

    identity = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=k.device)
    diag_beta = torch.diag_embed(beta.squeeze(-1))
    m = torch.linalg.solve_triangular(identity + t, diag_beta, upper=False, unitriangular=True)
    u_pre = m @ v_i
    w = m @ k_decayed
    mqk = _reference_lower_matmul(q_i * scale_value, k_i, cumulative_decay, diagonal=0)
    chunk_decay = gamma[..., -1, :, None].contiguous()
    k_restored = k_i * (cumulative_decay[..., -1:, :] - cumulative_decay).exp()
    return _StageOneGolden(gamma, k_decayed_beta, k_decayed, q_decayed, t, u_pre, w, mqk, k_restored, chunk_decay)


def _stage1_npu(k, v, q, beta_brc, g, *, scale_value, shared_score_reference=False) -> _StageOneGolden:
    """Stage1 NPU-oriented golden：显式 batch/head/chunk 循环 + bf16/fp16 交接。"""
    if v.shape[1] != k.shape[1]:
        rep = v.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        q = q.repeat_interleave(rep, dim=1)
    k_i = _to_bf16_to_fp32(_chunk_major(k))
    v_i = _chunk_major(v).to(torch.float32)
    q_i = _to_bf16_to_fp32(_chunk_major(q))
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    batch, heads, chunks, _, dim = k_i.shape
    gamma = torch.empty_like(k_i)
    k_decayed_beta = torch.empty_like(k_i)
    k_decayed = torch.empty_like(k_i)
    q_decayed = torch.empty_like(k_i)
    t_out = torch.empty(batch, heads, chunks, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=k.device)
    u_pre = torch.empty_like(k_i)
    w = torch.empty_like(k_i)
    mqk = torch.empty_like(t_out)
    k_restored = torch.empty_like(k_i)
    chunk_decay = torch.empty(batch, heads, chunks, dim, 1, dtype=torch.float32, device=k.device)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                k_c = k_i[b, h, c]
                v_c = v_i[b, h, c]
                q_c = q_i[b, h, c]
                beta_c = beta[b, h, c]
                g_c = g_i[b, h, c]

                g_cumsum = torch.cumsum(g_c, dim=0)  # AIV 全程 fp32
                gamma_c = torch.exp(g_cumsum)
                k_decayed_prepare = gamma_c * k_c
                k_decayed_c = _to_bf16_to_fp32(k_decayed_prepare)
                q_decayed_c = _to_bf16_to_fp32(gamma_c * q_c * scale_value)

                t = (
                    _reference_lower_matmul(
                        k_c, k_c, g_cumsum, diagonal=-1, cast_operand=_to_bf16_to_fp32,
                        shared_score_reference=shared_score_reference,
                    )
                    * beta_c
                )
                mqk_c = _to_bf16_to_fp32(
                    _reference_lower_matmul(
                        q_c * scale_value, k_c, g_cumsum, diagonal=0, cast_operand=_to_bf16_to_fp32,
                        shared_score_reference=shared_score_reference,
                    ),
                )

                negative_t = _to_fp16_to_fp32(-t)
                resident_inv = _npu_neumann_inverse_from_negative_t(negative_t)

                k_decayed_beta_c = _to_bf16_to_fp32(beta_c * k_decayed_prepare)
                v_beta = _to_bf16_to_fp32(beta_c * v_c)
                u_pre_c = _to_bf16_to_fp32(_to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(v_beta))
                workspace_w = _to_bf16_to_fp32(_to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(-k_decayed_beta_c))
                gamma_c_last = gamma_c[-1].reshape(k_c.shape[-1], 1)
                k_restored_c = _to_bf16_to_fp32(k_c * (g_cumsum[-1:] - g_cumsum).exp())

                gamma[b, h, c] = gamma_c
                k_decayed[b, h, c] = k_decayed_c
                k_decayed_beta[b, h, c] = k_decayed_beta_c
                q_decayed[b, h, c] = q_decayed_c
                t_out[b, h, c] = t
                u_pre[b, h, c] = u_pre_c
                w[b, h, c] = -workspace_w
                mqk[b, h, c] = mqk_c
                chunk_decay[b, h, c] = gamma_c_last
                k_restored[b, h, c] = k_restored_c
    return _StageOneGolden(gamma, k_decayed_beta, k_decayed, q_decayed, t_out, u_pre, w, mqk, k_restored, chunk_decay)


def _stage2_cpu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 CPU 公式（全 fp32）：用 stage1 工作区 + 初始 state 递推。"""
    _, _, chunks, _, _ = stage1.u_pre.shape
    u = torch.zeros_like(stage1.u_pre)
    o_state = torch.zeros_like(stage1.u_pre)
    state = initial_state.to(torch.float32).clone()
    state_history = torch.empty(
        *stage1.u_pre.shape[:3], state.shape[-2], state.shape[-1], dtype=torch.float32, device=state.device,
    )
    u_pre = stage1.u_pre.to(torch.float32)
    w = stage1.w.to(torch.float32)
    q_decayed = stage1.q_decayed.to(torch.float32)
    k_restored = stage1.k_restored.to(torch.float32)
    chunk_decay = stage1.chunk_decay.to(torch.float32)
    for c in range(chunks):
        state_history[:, :, c] = state
        u_c = u_pre[:, :, c] - w[:, :, c] @ state
        o_c = q_decayed[:, :, c] @ state
        state = (
            chunk_decay[:, :, c] * state + k_restored[:, :, c].transpose(-1, -2) @ u_c
        )
        u[:, :, c] = u_c
        o_state[:, :, c] = o_c
    return _StageTwoGolden(u=u, o_state=o_state, s=state, state_history=state_history)


def _stage2_npu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 NPU-oriented golden：显式 chunk 递推 + bf16 交接。"""
    batch, heads, chunks, _, _ = stage1.u_pre.shape
    u = torch.zeros_like(stage1.u_pre, dtype=torch.float32)
    o_state = torch.zeros_like(stage1.u_pre, dtype=torch.float32)
    u_pre = _to_bf16_to_fp32(stage1.u_pre)
    w_neg = _to_bf16_to_fp32(-stage1.w)  # workspace 存负 W
    q_decayed = _to_bf16_to_fp32(stage1.q_decayed)
    k_restored = _to_bf16_to_fp32(stage1.k_restored)
    chunk_decay = stage1.chunk_decay.to(torch.float32)
    state = initial_state.to(torch.float32).clone()
    state_history = torch.empty(
        batch, heads, chunks, state.shape[-2], state.shape[-1], dtype=torch.float32, device=state.device,
    )
    for b in range(batch):
        for h in range(heads):
            state_bh = state[b, h]
            for c in range(chunks):
                state_history[b, h, c] = state_bh
                w_s = _to_bf16_to_fp32(w_neg[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                u_c = _to_bf16_to_fp32(u_pre[b, h, c] + w_s)
                o_c = _to_bf16_to_fp32(q_decayed[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                kr_u = _to_bf16_to_fp32(k_restored[b, h, c].transpose(-1, -2)) @ _to_bf16_to_fp32(u_c)
                # Current DSL retains the recurrent accumulator in FP32.
                scaled_state = chunk_decay[b, h, c] * state_bh
                state_bh = scaled_state + kr_u
                u[b, h, c] = u_c
                o_state[b, h, c] = o_c
            state[b, h] = state_bh
    return _StageTwoGolden(u=u, o_state=o_state, s=state, state_history=state_history)


def _stage3_cpu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 CPU 公式：O = O_state + Mqk @ U。"""
    o_i = stage2.o_state.to(torch.float32) + stage1.mqk.to(torch.float32) @ stage2.u.to(torch.float32)
    batch, heads, chunks, chunk, dim = o_i.shape
    return _StageThreeGolden(o=o_i.reshape(batch, heads, chunks * chunk, dim).contiguous())


def _stage3_npu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 NPU-oriented golden：显式 chunk 合成 + bf16 交接。"""
    batch, heads, chunks, chunk, dim = stage2.o_state.shape
    o_i = torch.empty_like(stage2.o_state, dtype=torch.float32)
    # Q@state and Mqk@U accumulate into one FP32 L0C tile.
    o_state = stage2.o_state.float()
    mqk = _to_bf16_to_fp32(stage1.mqk)
    u = _to_bf16_to_fp32(stage2.u)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                mqk_u = _to_bf16_to_fp32(mqk[b, h, c]) @ _to_bf16_to_fp32(u[b, h, c])
                o_i[b, h, c] = _to_bf16_to_fp32(o_state[b, h, c] + mqk_u)
    return _StageThreeGolden(o=o_i.reshape(batch, heads, chunks * chunk, dim).contiguous())


def _chain_cpu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_cpu(inputs.k, inputs.v, inputs.q, inputs.beta_brc, inputs.g, scale_value=inputs.scale_value)
    s2 = _stage2_cpu(s1, inputs.initial_state)
    s3 = _stage3_cpu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def _chain_npu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_npu(inputs.k, inputs.v, inputs.q, inputs.beta_brc, inputs.g, scale_value=inputs.scale_value)
    s2 = _stage2_npu(s1, inputs.initial_state)
    s3 = _stage3_npu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def compute_golden(name: str, inputs: KdaInputs):
    """Apply activation/normalization, then pad with an identity recurrence tail."""
    tokens = inputs.q.shape[-2]
    alpha = torch.exp(inputs.a_log).view(1, -1, 1, 1)
    bias = inputs.dt_bias.view(1, -1, 1, inputs.dt_bias.shape[-1])
    activated = dataclasses.replace(
        inputs, q=inputs.q.float() / torch.sqrt(inputs.q.float().square().sum(-1, keepdim=True) + 1e-6),
        k=inputs.k.float() / torch.sqrt(inputs.k.float().square().sum(-1, keepdim=True) + 1e-6),
        g=inputs.lower_bound * torch.sigmoid(alpha * (inputs.g.float() + bias)),
        beta_brc=torch.sigmoid(inputs.beta_brc.float()),
    )
    # Zero activated decay means exp(g)=1; beta=0 means no state update.
    padding = (-tokens) % CHUNK_SIZE
    if padding:
        activated = dataclasses.replace(
            activated,
            **{
                name: F.pad(getattr(activated, name), (0, 0, 0, padding))
                for name in ("q", "k", "v", "g", "beta_brc")
            },
        )
    chain = _chain_cpu(activated) if name == "cpu_chunk" else _chain_npu(activated)
    return chain.stage3.o[:, :, :tokens].contiguous(), chain.stage2.s.transpose(-1, -2).contiguous()


pytestmark = pytest.mark.npu


_LOWER_BOUND = float(os.environ.get("KDA_LOWER_BOUND") or -5.0)


_TOL = float(os.environ.get("KDA_TOL", "5e-3"))


def require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


@dataclasses.dataclass(frozen=True)
class _DynamicPrecisionCase:
    layout: str
    has_cu: bool
    lengths: tuple[int, ...]
    nv: int
    nk: int
    seed: int
    lower_bound: float = _LOWER_BOUND
    qk_scale: float = 1.0

    @property
    def case_id(self):
        lengths_label = "-".join(str(length) for length in self.lengths)
        suffix = "-cu" if self.has_cu else ""
        scale_label = f"_qks{self.qk_scale:g}" if self.qk_scale != 1.0 else ""
        return f"{self.layout.lower()}{suffix}_n{self.nv}k{self.nk}_s{lengths_label}_seed{self.seed}{scale_label}"


_DYNAMIC_FAST_CASES = (
    _DynamicPrecisionCase("BNSD", False, (65,), 1, 1, 101),
    _DynamicPrecisionCase("BSND", False, (65,), 1, 1, 102),
    _DynamicPrecisionCase("TND", True, (65,), 1, 1, 105),
    _DynamicPrecisionCase("BNSD", False, (65,), 16, 1, 201),
    _DynamicPrecisionCase("BSND", False, (65,), 16, 1, 202),
    _DynamicPrecisionCase("TND", True, (65,), 16, 1, 205),
    _DynamicPrecisionCase("BNSD", False, (1, 1), 3, 1, 301, qk_scale=1e-4),
    _DynamicPrecisionCase("BSND", False, (16, 16), 6, 2, 302, qk_scale=64.0),
    _DynamicPrecisionCase("TND", True, (1, 63, 65), 3, 1, 303),
    _DynamicPrecisionCase("BNSD", False, (63,), 96, 96, 304),
    _DynamicPrecisionCase("BSND", False, (64,), 96, 96, 305),
    _DynamicPrecisionCase("TND", True, (65,), 96, 96, 306),
    _DynamicPrecisionCase("BNSD", False, (127,), 6, 3, 307),
    _DynamicPrecisionCase("BSND", False, (128,), 6, 3, 308),
    _DynamicPrecisionCase("TND", True, (129, 64, 1), 6, 2, 309),
    _DynamicPrecisionCase("TND", True, (511, 513), 3, 1, 310),
    _DynamicPrecisionCase("BNSD", False, (513,), 3, 1, 311),
    _DynamicPrecisionCase("BSND", False, (513,), 3, 1, 312),
)


def _make_dynamic_case(case):
    inputs = make_inputs_generalized(
        len(case.lengths), case.nv, max(case.lengths), 128, nk=case.nk, seed=case.seed, lower_bound=case.lower_bound,
    )
    inputs.q.mul_(case.qk_scale)
    inputs.k.mul_(case.qk_scale)
    outputs = {name: [] for name in ("cpu_chunk", "npu_chunk")}
    states = {name: [] for name in outputs}
    for batch_idx, length in enumerate(case.lengths):
        batch_slice = slice(batch_idx, batch_idx + 1)
        one = dataclasses.replace(
            inputs, k=inputs.k[batch_slice, :, :length], v=inputs.v[batch_slice, :, :length],
            q=inputs.q[batch_slice, :, :length], beta_brc=inputs.beta_brc[batch_slice, :, :length],
            g=inputs.g[batch_slice, :, :length], initial_state=inputs.initial_state[batch_slice],
        )
        for name in outputs:
            output, state = compute_golden(name, one)
            outputs[name].append(output)
            states[name].append(state)
    return (
        inputs,
        outputs,
        {name: torch.cat(values, dim=0) for name, values in states.items()},
    )


def _dynamic_layout_inputs(inputs, case):
    beta = inputs.beta_brc.squeeze(-1)
    if case.layout == "BNSD":
        return inputs.q, inputs.k, inputs.v, inputs.g, beta
    if case.layout == "BSND":
        return tuple(tensor.transpose(1, 2).contiguous() for tensor in (inputs.q, inputs.k, inputs.v, inputs.g, beta))

    def packed(tensor):
        return torch.cat(
            [tensor[b, :, :length].transpose(0, 1) for b, length in enumerate(case.lengths)], dim=0,
        ).contiguous()

    return tuple(packed(tensor) for tensor in (inputs.q, inputs.k, inputs.v, inputs.g, beta))


def _dynamic_valid_outputs(output, case):
    if case.layout == "TND":
        result = []
        start = 0
        for length in case.lengths:
            token_slice = slice(start, start + length)
            result.append(output[token_slice].transpose(0, 1).unsqueeze(0))
            start += length
        return result
    if case.layout == "BSND":
        output = output.transpose(1, 2)
    return [
        output[slice(b, b + 1), :, :length] for b, length in enumerate(case.lengths)
    ]


def _chunk_public_to_tokens(output, case):
    if case.layout == "TND":
        flat = output.transpose(0, 1).reshape(case.nv, -1, output.shape[-1])
        tokens = torch.empty((case.nv, sum(case.lengths), output.shape[-1]), dtype=output.dtype, device=output.device)
        token_start = 0
        for batch_idx, length in enumerate(case.lengths):
            chunk_base = token_start // CHUNK_SIZE + batch_idx
            token_slice = slice(token_start, token_start + length)
            chunk_token_start = chunk_base * CHUNK_SIZE
            chunk_token_slice = slice(chunk_token_start, chunk_token_start + length)
            tokens[:, token_slice] = flat[
                :,
                chunk_token_slice,
            ]
            token_start += length
        return tokens.transpose(0, 1)
    output = output.transpose(1, 2).reshape(len(case.lengths), case.nv, -1, output.shape[-1])
    max_length = max(case.lengths)
    output = output[:, :, :max_length]
    if case.layout == "BSND":
        return output.transpose(1, 2)
    return output


def _assert_schedule_covers_logical_chunks(metadata, case):
    schedule = _DYNAMIC_METADATA.decode_metadata(metadata.cpu().tolist())
    assert schedule["status"] == _DYNAMIC_METADATA.STATUS_OK
    assert schedule["batch"] == len(case.lengths)
    assert schedule["value_heads"] == case.nv
    chunk_cursor = [0] * len(case.lengths)
    starts = [sum(case.lengths[:batch]) for batch in range(len(case.lengths))]
    for index, record in enumerate(schedule["stage12_rounds"]):
        assert record["stage12_round_idx"] == index
        for active, batch in enumerate(record["batch_idx"]):
            assert record["valid_seq_len"][active] == case.lengths[batch]
            assert record["storage_batch_idx"][active] == (
                0 if case.layout == "TND" else batch
            )
            assert record["token_start"][active] == (
                starts[batch] if case.layout == "TND" else 0
            )
            assert record["chunk_start_per_group"][active] == chunk_cursor[batch]
            count = record["chunk_num_per_group"][active]
            assert 0 < count <= (case.lengths[batch] + 63) // 64 - chunk_cursor[batch]
            assert (
                record["stage1_task_prefix"][active + 1]
                - record["stage1_task_prefix"][active]
                == case.nv * count
            )
            chunk_cursor[batch] += count
        assert record["stage1_task_prefix"][-1] <= _DYNAMIC_METADATA.WORKSPACE_SLOTS
    assert chunk_cursor == [(length + 63) // 64 for length in case.lengths]
    return schedule


def _l0c_cache_reference(stage1, u_pre_fp32, initial_state):
    """Keep U_pre in FP32 through W*S, then round only the published U."""
    state = initial_state.float().clone()
    chunks = u_pre_fp32.shape[2]
    updated_values = torch.empty_like(u_pre_fp32)
    state_history = torch.empty(
        *u_pre_fp32.shape[:3], *state.shape[-2:], dtype=torch.float32,
    )
    for chunk in range(chunks):
        state_history[:, :, chunk] = state
        current = u_pre_fp32[:, :, chunk] - stage1.w[:, :, chunk] @ _to_bf16_to_fp32(state)
        updated_values[:, :, chunk] = current
        state = (
            stage1.chunk_decay[:, :, chunk] * state
            + stage1.k_restored[:, :, chunk].transpose(-1, -2) @ _to_bf16_to_fp32(current)
        )
    return updated_values, state_history


def _saved_matrix_golden(inputs, q_norm, k_norm, beta, g, *, shared_score_reference=False):
    length = q_norm.shape[-2]
    repeat = inputs.v.shape[1] // q_norm.shape[1]
    scale_value = inputs.scale_value

    def pad_to_chunk(tensor):
        return F.pad(tensor, (0, 0, 0, (-length) % CHUNK_SIZE))

    matrices = {}
    for name, cast in (("cpu_chunk", lambda x: x), ("npu_chunk", _to_bf16_to_fp32)):
        qc = _chunk_major(pad_to_chunk(cast(q_norm).repeat_interleave(repeat, 1)))
        kc = _chunk_major(pad_to_chunk(cast(k_norm).repeat_interleave(repeat, 1)))
        bc = _chunk_major(pad_to_chunk(beta))
        gc = _chunk_major(pad_to_chunk(g)).cumsum(-2)
        operand_cast = None if name == "cpu_chunk" else _to_bf16_to_fp32
        # Keep the independent FP32 formula unchanged; only the NPU reference
        # models the implementation's choice of BF16 operand scaling.
        shared_reference = name == "npu_chunk" and shared_score_reference
        mqk = _reference_lower_matmul(
            qc * scale_value, kc, gc, diagonal=0, cast_operand=operand_cast,
            shared_score_reference=shared_reference,
        )
        t = (
            _reference_lower_matmul(
                kc, kc, gc, diagonal=-1, cast_operand=operand_cast,
                shared_score_reference=shared_reference,
            )
            * bc
        )
        identity = torch.eye(CHUNK_SIZE).expand_as(t)
        if name == "cpu_chunk":
            inverse = torch.linalg.solve_triangular(identity + t, identity, upper=False, unitriangular=True)
        else:
            inverse_tiles = []
            for tile in t.reshape(-1, CHUNK_SIZE, CHUNK_SIZE):
                inverse_tiles.append(_npu_neumann_inverse_from_negative_t(_to_fp16_to_fp32(-tile)))
            inverse = torch.stack(inverse_tiles).reshape_as(t)
            mqk = _to_bf16_to_fp32(mqk)
        matrices[name] = [x.flatten(2, 3)[:, :, :length] for x in (mqk, inverse)]
    return matrices, inverse


def _saved_golden(inputs, batch_idx, length, *, direct_u=False, shared_score_reference=False):
    batch_slice = slice(batch_idx, batch_idx + 1)
    q = inputs.q[batch_slice, :, :length].float()
    k = inputs.k[batch_slice, :, :length].float()
    q_rstd = torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k_rstd = torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q_norm, k_norm = q * q_rstd, k * k_rstd
    beta = inputs.beta_brc[batch_slice, :, :length].float().sigmoid()
    g = inputs.g[batch_slice, :, :length].float()
    g = inputs.lower_bound * torch.sigmoid(
        inputs.a_log.exp().view(1, -1, 1, 1) * (g + inputs.dt_bias.view(1, -1, 1, 128)),
    )

    def pad_to_chunk(tensor):
        return F.pad(tensor, (0, 0, 0, (-length) % CHUNK_SIZE))

    repeat = inputs.v.shape[1] // q.shape[1]
    matrices, inverse = _saved_matrix_golden(
        inputs, q_norm, k_norm, beta, g, shared_score_reference=shared_score_reference,
    )
    padded_q = pad_to_chunk(q_norm.bfloat16())
    padded_k = pad_to_chunk(k_norm.bfloat16())
    padded_v = pad_to_chunk(inputs.v[batch_slice, :, :length])
    padded_beta = pad_to_chunk(beta)
    padded_g = pad_to_chunk(g)
    npu_stage1 = _stage1_npu(
        padded_k, padded_v, padded_q, padded_beta, padded_g, scale_value=inputs.scale_value,
        shared_score_reference=shared_score_reference,
    )
    npu_stage2 = _stage2_npu(npu_stage1, inputs.initial_state[batch_slice])
    u_reference = npu_stage2.u
    h_reference = npu_stage2.state_history
    if direct_u:
        # The final loop iteration above holds the NPU-rounded resident inverse.
        # This product stays FP32 in L0C; the public U_pre cache is still BF16.
        v_beta = _to_bf16_to_fp32(_chunk_major(padded_beta) * _chunk_major(padded_v).float())
        u_pre_fp32 = _to_bf16_to_fp32(inverse) @ v_beta
        u_reference, h_reference = _l0c_cache_reference(
            npu_stage1, u_pre_fp32, inputs.initial_state[batch_slice],
        )
    g_cumsum = _chunk_major(padded_g).cumsum(-2)
    q_value_heads = _chunk_major(padded_q).repeat_interleave(repeat, dim=1).float()
    qg = _to_bf16_to_fp32(q_value_heads * g_cumsum.exp())

    def flatten_tokens(value):
        return value.flatten(2, 3)[:, :, :length]

    extras = (
        flatten_tokens(npu_stage1.w).bfloat16(),
        flatten_tokens(npu_stage1.u_pre).bfloat16(),
        flatten_tokens(qg).bfloat16(),
        flatten_tokens(npu_stage1.k_restored).bfloat16(),
        flatten_tokens(u_reference) if direct_u else flatten_tokens(u_reference).bfloat16(),
        flatten_tokens(g_cumsum),
        h_reference.transpose(-1, -2),
    )
    return (q_norm.bfloat16(), k_norm.bfloat16(), q_rstd, k_rstd, beta), matrices, extras


def _assert_saved_shapes(saved, inputs, case, *, disable_recompute):
    assert len(saved) == 14
    q_norm, k_norm, q_rstd, k_rstd, beta, aqk, akk, w, u, qg, kg, v_new, g_cumsum, h = (
        saved
    )
    physical = _dynamic_layout_inputs(inputs, case)
    public_q = physical[0] if case.layout == "TND" else inputs.q
    public_v = physical[2]
    assert q_norm.shape == k_norm.shape == public_q.shape
    assert q_rstd.shape == k_rstd.shape == public_q.shape[:-1]
    public_beta = physical[4] if case.layout == "TND" else inputs.beta_brc.squeeze(-1).transpose(1, 2)
    assert beta.shape == public_beta.shape
    storage_length = (
        physical[0].shape[0]
        if case.layout == "TND"
        else physical[0].shape[2 if case.layout == "BNSD" else 1]
    )
    chunk_count = (
        _DYNAMIC_STAGED._tnd_state_chunk_capacity(storage_length, len(case.lengths))
        if case.layout == "TND"
        else (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    chunk_prefix = (chunk_count, case.nv) if case.layout == "TND" else (len(case.lengths), chunk_count, case.nv)
    assert aqk.shape == akk.shape == chunk_prefix + (CHUNK_SIZE, CHUNK_SIZE)
    if disable_recompute:
        assert all(isinstance(value, torch.Tensor) for value in (w, u, qg, kg, v_new, g_cumsum))
        assert g_cumsum.shape == chunk_prefix + (CHUNK_SIZE, SUPPORTED_HEAD_DIM)
        assert all(value.shape == public_v.shape for value in (w, u, qg, kg, v_new)), (
            [tuple(value.shape) for value in (w, u, qg, kg, v_new)],
            tuple(public_v.shape),
        )
    else:
        assert all(value is None for value in (w, u, qg, kg, v_new, g_cumsum))
    if case.layout == "TND":
        state_shape = (
            _DYNAMIC_STAGED._tnd_state_chunk_capacity(storage_length, len(case.lengths)),
            case.nv,
            128,
            128,
        )
    else:
        state_shape = chunk_prefix + (128, 128)
    assert h.shape == state_shape


def _saved_actual_tokens(saved, case, *, disable_recompute):
    actual = []
    for index, value in enumerate(saved[:-1]):
        if 7 <= index < 13 and not disable_recompute:
            assert value is None
            actual.append(None)
            continue
        assert value.dtype == (
            LOW_DTYPE if index in (0, 1, 5, 6, 7, 8, 9, 10, 11) else torch.float32
        )
        if index in (2, 3, 4):
            value = value.unsqueeze(-1)
        if index in (5, 6, 12):
            value = _chunk_public_to_tokens(value, case)
        elif index == 4 and case.layout == "BNSD":
            value = value.transpose(1, 2)
        elif index < 4 and case.layout == "BSND":
            value = value.transpose(1, 2)
        actual.append(_dynamic_valid_outputs(value.cpu(), case))
    assert saved[-1].dtype == torch.float32
    return actual


def _assert_saved_precision(saved, inputs, case, *, disable_recompute, direct_u=False, shared_score_reference=False):
    _assert_saved_shapes(saved, inputs, case, disable_recompute=disable_recompute)
    actual = _saved_actual_tokens(saved, case, disable_recompute=disable_recompute)
    h = saved[-1]
    for batch_idx, length in enumerate(case.lengths):
        token_golden, matrices, extras = _saved_golden(
            inputs, batch_idx, length, direct_u=direct_u, shared_score_reference=shared_score_reference,
        )
        for index, expected in enumerate(token_golden):
            tol = 1e-3 if index < 2 else 1e-6
            torch.testing.assert_close(actual[index][batch_idx].float(), expected.float(), atol=tol, rtol=tol)
        for name, pair in matrices.items():
            for index, expected in zip((5, 6), pair):
                torch.testing.assert_close(
                    actual[index][batch_idx].float(), expected.float(), atol=_TOL, rtol=_TOL,
                    msg=lambda msg: f"saved {index} vs {name}: {msg}",
                )
        for start in range(0, length, CHUNK_SIZE):
            rows = min(CHUNK_SIZE, length - start)
            for index in (5, 6):
                tile = actual[index][batch_idx][..., slice(start, start + rows), :]
                assert torch.count_nonzero(torch.triu(tile.float(), diagonal=1)) == 0
            tile = actual[6][batch_idx][..., slice(start, start + rows), :rows]
            assert torch.equal(tile.diagonal(dim1=-2, dim2=-1), torch.ones_like(tile.diagonal(dim1=-2, dim2=-1)))
        if disable_recompute:
            for index, expected in zip(range(7, 13), extras[:-1]):
                torch.testing.assert_close(
                    actual[index][batch_idx].float(), expected.float(), atol=_TOL, rtol=_TOL,
                    msg=lambda msg: f"saved {index} cache mismatch: {msg}",
                )
        chunk_count = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        if case.layout == "TND":
            token_start = sum(case.lengths[:batch_idx])
            chunk_base = _DYNAMIC_STAGED._tnd_state_chunk_base(token_start, batch_idx)
            actual_h = (
                h.cpu()[slice(chunk_base, chunk_base + chunk_count)] .transpose(0, 1) .unsqueeze(0)
            )
        else:
            batch_slice = slice(batch_idx, batch_idx + 1)
            actual_h = h.cpu()[batch_slice, :chunk_count].transpose(1, 2)
        torch.testing.assert_close(actual_h.float(), extras[-1].float(), atol=_TOL, rtol=_TOL)


@dataclasses.dataclass
class _DynamicForwardResult:
    output: torch.Tensor
    state: torch.Tensor
    saved: list[torch.Tensor | None]
    metadata: torch.Tensor
    initial_state: torch.Tensor
    initial_state_before: torch.Tensor


def _run_dynamic_forward(inputs, case, template, *, whitebox, disable_recompute=None):
    # None means omit the keyword to exercise the operator default (False).
    q, k, v, g, beta = (tensor.npu() for tensor in _dynamic_layout_inputs(inputs, case))
    cu_seqlens = None
    if case.has_cu:
        cu_seqlens = torch.tensor((0, *torch.tensor(case.lengths).cumsum(0).tolist()), dtype=torch.int32).npu()
    initial_state = inputs.initial_state.transpose(-1, -2).contiguous().npu()
    initial_state_before = initial_state.clone()
    attrs = dict(batch_size=len(case.lengths), nheads_qk=case.nk, nheads_v=case.nv,
                 seqlens=sum(case.lengths) if case.layout == "TND" else inputs.q.shape[2], layout=case.layout)

    def make_metadata(training):
        if template == "auto":
            return _DYNAMIC_METADATA.flash_kda_metadata(cu_seqlens, enable_grad=training, **attrs)
        return _DYNAMIC_METADATA._build_flash_kda_metadata(
            cu_seqlens=cu_seqlens, enable_grad=training, template=template, **attrs,
        )

    metadata = make_metadata(True)
    if whitebox:
        schedule = _assert_schedule_covers_logical_chunks(metadata, case)
        if (
            sum((length + 63) // 64 for length in case.lengths) * case.nv
            > _DYNAMIC_METADATA.WORKSPACE_SLOTS
        ):
            assert schedule["stage12_round_num"] > 1
    args = (
        q, k, v, g, beta, inputs.scale_value, initial_state, inputs.a_log.npu(), inputs.dt_bias.npu(),
        inputs.lower_bound, case.layout, metadata,
    )
    before = [value.cpu().clone() for value in args if isinstance(value, torch.Tensor)]
    kw = dict(initial_state=initial_state, A_log=args[7], dt_bias=args[8], scale=inputs.scale_value,
              lower_bound=inputs.lower_bound, layout=case.layout, cu_seqlens=cu_seqlens, output_final_state=True)
    if disable_recompute is not None:
        kw["disable_recompute"] = disable_recompute
    with torch.inference_mode(False):
        out = _DYNAMIC_KDA.flash_kda(q, k, v, g, beta, metadata=metadata, **kw)
        # The compatible full-forward entry lives in this same unified module.
        forward = _DYNAMIC_KDA.flash_kda_fwd(q, k, v, g, beta, metadata=metadata, **kw)
    assert len(out) == 3
    output, state, chunk_states = out
    assert chunk_states is None
    assert len(forward) == 16 and forward[2] is None
    for actual, expected in zip(_dynamic_valid_outputs(output, case), _dynamic_valid_outputs(forward[0], case)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, forward[1], rtol=0, atol=0)
    (_, _, _, aqk, akk, w, u, qg, kg, v_new, g_cumsum,
     q_norm, q_rstd, k_norm, k_rstd, beta_active) = forward
    with torch.inference_mode():
        inference = _DYNAMIC_KDA.flash_kda(
            q, k, v, g, beta, metadata=make_metadata(False), return_intermediate_states=True, **kw,
        )
    assert len(inference) == 3 and inference[2] is not None
    # Padding rows are unspecified. Check only defined token values.
    for trained, inferred in zip(_dynamic_valid_outputs(output, case), _dynamic_valid_outputs(inference[0], case)):
        torch.testing.assert_close(trained, inferred, rtol=0, atol=0)
    torch.testing.assert_close(state, inference[1], rtol=0, atol=0)
    saved = [q_norm, k_norm, q_rstd, k_rstd, beta_active, aqk, akk, w, u, qg, kg, v_new, g_cumsum, inference[2]]
    torch.npu.synchronize()

    for value, expected in zip((v for v in args if isinstance(v, torch.Tensor)), before):
        torch.testing.assert_close(value.cpu(), expected, atol=0, rtol=0, equal_nan=True)
    return _DynamicForwardResult(
        output=output, state=state, saved=saved, metadata=metadata,
        initial_state=initial_state, initial_state_before=initial_state_before,
    )


def _assert_dynamic_output_golden(result, case, golden_outputs, golden_state):
    output, state = result.output, result.state
    initial_state, initial_state_before = result.initial_state, result.initial_state_before
    actual_state = state.cpu().float()
    assert torch.isfinite(actual_state).all(), f"{case.case_id}: final_state contains inf/nan"
    torch.testing.assert_close(initial_state.cpu().float(), initial_state_before.cpu().float(), atol=0.0, rtol=0.0)
    actual_outputs = [actual.cpu().float() for actual in _dynamic_valid_outputs(output, case)]
    for name in ("cpu_chunk", "npu_chunk"):
        expected_state = golden_state[name].float()
        assert torch.isfinite(expected_state).all(), f"{case.case_id}: {name} state is not finite"
        torch.testing.assert_close(
            actual_state, expected_state, atol=_TOL, rtol=_TOL, msg=lambda msg: f"DSL vs {name} state: {msg}",
        )
        output_max = 0.0
        for actual, expected in zip(actual_outputs, golden_outputs[name]):
            assert torch.isfinite(actual).all(), f"{case.case_id}: DSL output is not finite"
            assert torch.isfinite(expected).all(), f"{case.case_id}: {name} output is not finite"
            torch.testing.assert_close(
                actual, expected.float(), atol=_TOL, rtol=_TOL, msg=lambda msg: f"DSL vs {name} output: {msg}",
            )
            output_max = max(output_max, (actual - expected).abs().amax().item())
        state_max = (actual_state - expected_state).abs().amax().item()
        LOGGER.info("%s vs %s: O max_abs=%.3e state max_abs=%.3e", case.case_id, name, output_max, state_max)
    torch.testing.assert_close(
        golden_state["npu_chunk"], golden_state["cpu_chunk"], atol=_TOL, rtol=_TOL,
        msg=lambda msg: f"npu_chunk vs cpu_chunk state: {msg}",
    )
    for simulated, fp32 in zip(golden_outputs["npu_chunk"], golden_outputs["cpu_chunk"]):
        torch.testing.assert_close(
            simulated, fp32, atol=_TOL, rtol=_TOL, msg=lambda msg: f"npu_chunk vs cpu_chunk output: {msg}",
        )


def _assert_dynamic_precision_case(
    case, *, whitebox=False, compare_output_golden=True, template="auto", disable_recompute=None,
):
    require_npu()
    inputs, golden_outputs, golden_state = _make_dynamic_case(case)
    if whitebox and case.has_cu and case.layout != "TND":
        # Invalid rows must never affect valid outputs or the final state.
        # Padding output is undefined, so only poison inputs, not expected output.
        for batch, length in enumerate(case.lengths):
            for tensor in (inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta_brc):
                tensor[batch, :, length:] = float("nan")
    result = _run_dynamic_forward(
        inputs, case, template, whitebox=whitebox, disable_recompute=disable_recompute,
    )
    saved, metadata = result.saved, result.metadata
    initial_state_before = result.initial_state_before
    selected_template = int(metadata[_DYNAMIC_METADATA.HEADER_TEMPLATE].item())
    # These cases enable bounded gates and sigmoid beta without negative
    # eigenvalues. Within this range staged uses centered, shared score panels.
    shared_score_reference = (
        selected_template == _DYNAMIC_METADATA.TEMPLATE_STAGED
        and -0.125 <= inputs.scale_value <= 0.125
        and inputs.lower_bound >= -5.0
    )
    _assert_saved_precision(
        saved, inputs, case, direct_u=selected_template == _DYNAMIC_METADATA.TEMPLATE_PERSISTENT,
        shared_score_reference=shared_score_reference, disable_recompute=disable_recompute is True,
    )
    if case.layout == "TND":
        state_h = saved[-1].cpu()
        for batch_idx, _ in enumerate(case.lengths):
            token_start = sum(case.lengths[:batch_idx])
            chunk_base = _DYNAMIC_STAGED._tnd_state_chunk_base(token_start, batch_idx)
            assert torch.equal(state_h[chunk_base], initial_state_before[batch_idx].cpu())
    else:
        assert torch.equal(saved[-1][:, 0].cpu(), initial_state_before.cpu())
    if not compare_output_golden:
        return

    _assert_dynamic_output_golden(result, case, golden_outputs, golden_state)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
def test_meta_training_outputs_and_forward_compatibility(layout):
    batch, length, heads = 2, 65, 3
    shape = {
        "BNSD": (batch, heads, length, SUPPORTED_HEAD_DIM),
        "BSND": (batch, length, heads, SUPPORTED_HEAD_DIM),
        "TND": (length, heads, SUPPORTED_HEAD_DIM),
    }[layout]
    v = torch.empty(shape, device="meta", dtype=LOW_DTYPE)
    beta = torch.empty(shape[:-1], device="meta", dtype=LOW_DTYPE)
    metadata = torch.empty(128, device="meta", dtype=torch.int32)
    cu = torch.empty(batch + 1, device="meta", dtype=torch.int32) if layout == "TND" else None
    kwargs = dict(layout=layout, cu_seqlens=cu, use_gate_in_kernel=False)
    for inference in (False, True):
        for final_state in (False, True):
            for intermediate_states in (False, True):
                with torch.inference_mode(inference):
                    result = _DYNAMIC_KDA.flash_kda(
                        v, v, v, v, beta, metadata, output_final_state=final_state,
                        return_intermediate_states=intermediate_states, **kwargs,
                    )
                assert len(result) == 3 and result[0].is_meta
                assert result[0].shape == shape and result[0].dtype == LOW_DTYPE
                assert (result[1] is not None) == final_state
                assert (result[2] is not None) == (inference and intermediate_states)
    def check_forward_result(result, *, inference, disable_recompute):
        assert len(result) == 16 and result[0].is_meta
        assert result[0].shape == v.shape and result[0].dtype == LOW_DTYPE
        assert result[1:3] == (None, None)
        assert all((value is not None) == (not inference) for value in result[3:5])
        assert all((value is not None) == (not inference) for value in result[11:16])
        if inference or not disable_recompute:
            assert all(value is None for value in result[5:11])
        else:
            for cache in result[5:10]:
                assert cache.shape == shape and cache.dtype == LOW_DTYPE
            chunks = (
                _DYNAMIC_STAGED._tnd_state_chunk_capacity(length, batch)
                if layout == "TND" else (length + CHUNK_SIZE - 1) // CHUNK_SIZE
            )
            prefix = (chunks, heads) if layout == "TND" else (batch, chunks, heads)
            assert result[10].shape == prefix + (CHUNK_SIZE, SUPPORTED_HEAD_DIM)
            assert result[10].dtype == torch.float32

    for op in (_DYNAMIC_KDA.flash_kda, _DYNAMIC_KDA.flash_kda_fwd):
        with pytest.raises(TypeError, match="metadata"):
            op(v, v, v, v, beta, **kwargs)
        with pytest.raises(ValueError, match="metadata is required"):
            op(v, v, v, v, beta, None, **kwargs)
    for inference in (False, True):
        with torch.inference_mode(inference):
            for op in (
                _DYNAMIC_KDA.flash_kda_fwd,
                _DYNAMIC_STAGED.staged_flash_kda_fwd,
                _DYNAMIC_PERSISTENT.persistent_flash_kda_fwd,
            ):
                default_result = op(v, v, v, v, beta, metadata, **kwargs)
                check_forward_result(default_result, inference=inference, disable_recompute=False)
                for disable_recompute in (False, True):
                    result = op(v, v, v, v, beta, metadata, disable_recompute=disable_recompute, **kwargs)
                    check_forward_result(result, inference=inference, disable_recompute=disable_recompute)


def test_public_gate_chunks_preserves_fp32_at_packed_boundaries():
    lengths = (3, 65)
    values = torch.arange(sum(lengths), dtype=torch.float32).view(1, 1, -1, 1) + 1.003
    values = values.expand(-1, -1, -1, SUPPORTED_HEAD_DIM)
    chunks = _DYNAMIC_KDA._public_gate_chunks(values, (2, 1, 1, 68, "TND"), lengths)
    assert chunks.dtype == torch.float32
    assert torch.equal(chunks[0, 0, :3], values[0, 0, :3])
    assert torch.equal(chunks[1, 0], values[0, 0, 3:67])
    assert torch.equal(chunks[2, 0, :1], values[0, 0, 67:])
    assert torch.count_nonzero(chunks[0, 0, 3:]) == 0


@pytest.mark.npu
@pytest.mark.parametrize("template", ["persistent", "staged"])
def test_zero_query_and_key_forward_compatibility(template):
    require_npu()
    q = torch.zeros((1, 1, 2, SUPPORTED_HEAD_DIM), device="npu", dtype=LOW_DTYPE)
    metadata = _DYNAMIC_METADATA._build_flash_kda_metadata(
        batch_size=1, nheads_qk=1, nheads_v=1, seqlens=2, enable_grad=True, layout="BNSD", template=template,
    )
    result = _DYNAMIC_KDA.flash_kda_fwd(
        q, q, q, q, q[..., 0].contiguous(), A_log=q.new_zeros((1,), dtype=torch.float32),
        metadata=metadata, layout="BNSD",
    )
    unified = _DYNAMIC_KDA.flash_kda(
        q, q, q, q, q[..., 0].contiguous(), A_log=q.new_zeros((1,), dtype=torch.float32),
        metadata=metadata, layout="BNSD",
    )
    assert len(unified) == 3 and unified[1:] == (None, None)
    torch.testing.assert_close(unified[0], result[0], atol=0, rtol=0)
    for inverse_norm in (result[12], result[14]):
        assert torch.equal(inverse_norm, torch.full_like(inverse_norm, 1000.0))


def test_forward_meta_validates_tensor_inputs():
    q = torch.empty((1, 1, 2, 128), dtype=torch.bfloat16, device="meta")
    state = torch.empty((1, 1, 128, 128), device="meta")
    args = dict(q=q, k=q, v=q, g=q, beta=q[..., 0].contiguous(), initial_state=state,
                A_log=torch.empty(1, device="meta"), layout="BNSD",
                metadata=torch.empty(164, dtype=torch.int32, device="meta"))
    for overrides, reason in [
        ({"q": q[0]}, "q rank"), ({"v": q[..., :64]}, "v head dimension expected 128"),
        ({"v": q[:, :, :1]}, "v shape"), ({"q": q.half()}, "q dtype"),
        ({"initial_state": state[0]}, "initial_state rank"),
        ({"initial_state": state[..., :64]}, "initial_state shape"),
        ({"initial_state": state.bfloat16()}, "initial_state dtype"),
        ({"q": q.transpose(-1, -2).contiguous().transpose(-1, -2)}, "q must be contiguous"),
    ]:
        with pytest.raises(ValueError, match=reason):
            _DYNAMIC_KDA.flash_kda(**(args | overrides))


@pytest.mark.parametrize("template", ("auto", "staged", "persistent"))
@pytest.mark.parametrize("case", _DYNAMIC_FAST_CASES, ids=lambda case: case.case_id)
def test_flash_kda_dynamic_precision(case, template):
    _assert_dynamic_precision_case(case, template=template)


@pytest.mark.npu
@pytest.mark.parametrize("template", ("staged", "persistent"))
@pytest.mark.parametrize("disable_recompute", (False, True), ids=("recompute", "save_caches"))
def test_flash_kda_recompute_modes(disable_recompute, template):
    # Keep the full shape pool on the default; explicit modes use a small case.
    _assert_dynamic_precision_case(
        _DYNAMIC_FAST_CASES[0], template=template, disable_recompute=disable_recompute,
    )


def teardown_module():
    _DYNAMIC_KDA.clear_caches()
    _DYNAMIC_METADATA.clear_metadata_caches()
