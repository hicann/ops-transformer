#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UndGenQkvRmsNormRopeCache 的 golden 参考实现，同时是 TTK 的输入/golden 插件。

本文件是本算子**唯一的 golden 实现**，自包含、不依赖同目录以外的任何文件，
供下列使用方共用：

  - ``examples/test_torch_und_gen_qkv_rms_norm_rope_cache.py``：torch 接口上板精度比对
  - ``tests/st/arch35/ttk_kernel_*.csv``：TTK kernel 用例，经文末的
    ``__input__`` / ``__golden__`` 注册表的 ``kernel`` 层接入

    python3 -m ttk kernel -i <csv> --plugin <此文件> --compare close

    判据用 `close`（TTK 的默认值，显式带上是为了不受调用方改默认值影响）：
    `|a-g| <= atol + rtol*|g|`，rtol/atol/ptol 由 CSV 的 `precision_tolerances` /
    `absolute_precision` 逐输出给出：rtol=2^-7（2 个 bf16 ULP）、atol=2^-13、
    ptol=0（不容许任何一个元素超标）。
    不要换成相对误差类判据：本算子的输出里存在 golden 恰好为 0、NPU 给出 2^-23
    量级的元素（RoPE 低半与高半两项相消），相对误差会被算成接近 1，而张量本身的
    典型量级是 0.4，这点绝对偏差没有意义。

  - ``tests/st/arch35/ttk_aclnn_*.csv``：TTK aclnn 用例，经同一注册表的 ``aclnn`` 层接入
    （适配函数见文末 ``aclnn_input`` / ``aclnn_golden``，与 kernel 层的契约差异见那里的注释）

    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
    python3 -m ttk aclnn -i <csv> --plugin <此文件> --compare close

    aclnn 侧同样带 `--compare close`，原因与 kernel 侧相同；容差也一样由该 CSV 的
    `precision_tolerances` / `absolute_precision` 逐输出给出（输出为 q / k_cache / v_cache 三个）。

张量约定
  und_qkv       bf16 [und_len, N, D]，N = Hq+Hk+Hv，D = head_dim = 128
  gen_qkv       bf16 [gen_len, N, D]，可为 None
  weights       bf16 [D] x4（und_q / und_k / gen_q / gen_k）
  cos_sin_cache f32  [max_pos, D]，前半 cos 后半 sin
  positions     i64  [3, total]（或 [total]）
  cat_indices   i64  [total]，out_t -> src_t，可为 None
  slot_mapping  i64  [total]，slot = block_idx * block_size + row_idx（索引类张量统一 int64）
  k_cache       bf16 [num_blocks, block_size, Hk, D]（逻辑 shape）
  v_cache       bf16 [num_blocks, block_size, Hv, D]（逻辑 shape）
  q             bf16 [total, Hq, D]

纯 CPU torch 实现，不依赖 NPU。
"""

import torch

__all__ = [
    "HEAD_DIM",
    "BLOCK_SIZE",
    "HEAD_COMBOS",
    "make_cos_sin_cache",
    "mrope_axis_map",
    "golden_und_gen_qkv_rms_norm_rope_cache",
    "golden_dense",
    "gather_cache_rows",
    "TYPICAL_CASES",
    "GENERALIZED_CASES",
    "build_case",
    "ttk_input",
    "ttk_golden",
]

# 本期支持范围
HEAD_DIM = 128
HEAD_COMBOS = ((8, 1, 1), (16, 2, 2))
# 算子对 block_size 无约束，这只是 case 未指定 block_size 时的默认值
BLOCK_SIZE = 128


# 工具
def make_cos_sin_cache(max_pos, head_dim, device="cpu", base=10000.0):
    """构造 cos_sin_cache [max_pos, head_dim]，前半 cos 后半 sin（与竞品一致）。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def mrope_axis_map(head_dim, mrope_section):
    """预计算 half 个索引各自取哪一轴的 cos/sin（Host Tiling 下发的 axisLut）。

    规则（与竞品 _mrope 完全一致）：
        axis = 0
        if i % 3 == 1 and i < 3 * sec[1]: axis = 1
        if i % 3 == 2 and i < 3 * sec[2]: axis = 2

    这三个数**不是**对 half 的划分：只有 sec[1]/sec[2] 被读，sec[0] 从不参与计算，
    T 是"其余全归它"的兜底轴。所以 [16,16,16] 实际得到 T/H/W = 32/16/16 而不是
    16/16/16，[64,16,16] 与 [0,16,16] 的轴映射逐位相同。
    """
    half = head_dim // 2
    if mrope_section is None or len(mrope_section) == 0:
        mrope_section = [half, 0, 0]
    assert len(mrope_section) == 3, "mrope_section 必须是长度 3 的列表"
    # 只是挡手误的粗筛，不是语义要求：参考实现没有这条约束，它自己的用例就是 sum=48。
    # 要放宽得连 op_host/..._base_tiling.cpp 的 CheckAttrsValid 一起改。
    assert sum(mrope_section) <= half, "mrope_section 三轴之和不能超过 head_dim/2"

    idx = torch.arange(half, dtype=torch.int64)
    axis = torch.zeros(half, dtype=torch.int64)
    axis[(idx % 3 == 1) & (idx < 3 * int(mrope_section[1]))] = 1
    axis[(idx % 3 == 2) & (idx < 3 * int(mrope_section[2]))] = 2
    return axis


def _normalize_qkv(x, num_heads_total, name):
    """QKV 输入约定为 3D [T, N, D]；为兼容旧脚本也接受 2D [T, N*D]。"""
    if x is None:
        return None
    if x.dim() == 2:
        assert x.shape[1] % num_heads_total == 0, f"{name} 的 hidden 无法被 N={num_heads_total} 整除"
        x = x.reshape(x.shape[0], num_heads_total, x.shape[1] // num_heads_total)
    assert x.dim() == 3, f"{name} 期望 3D [T, N, D]，实得 {tuple(x.shape)}"
    assert x.shape[1] == num_heads_total, (
        f"{name} 的 N={x.shape[1]} 与 Hq+Hk+Hv={num_heads_total} 不一致"
    )
    return x


def _normalize_positions(positions, total):
    """positions 支持 [3, total] 与 [total]（单序列，三轴广播）。"""
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    assert positions.shape == (3, total), f"positions 期望 [3, {total}]，实得 {tuple(positions.shape)}"
    return positions.to(torch.int64)


def _rmsnorm(x, weight, eps):
    """x: [T, H, D] float32；weight: [T, D] float32（每 token 已按 und/gen 选好）。"""
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv * weight.unsqueeze(1)


def _rope(x, cos, sin):
    """x: [T, H, D] float32；cos/sin: [T, half] float32（已按 mask 合并三轴）。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# 稠密段：split + index + rmsnorm + mrope（不含 cache 写入）
def golden_dense(
    und_qkv,
    gen_qkv,
    und_weights_q,
    und_weights_k,
    gen_weights_q,
    gen_weights_k,
    cos_sin_cache,
    positions,
    cat_indices,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    norm_eps=1e-6,
    mrope_section=None,
):
    """返回 float32 的 (q, k, v)，shape 分别 [total, Hq/Hk/Hv, D]。

    这是 KV Cache scatter 之前的中间结果，供精度定界与单元比对使用。
    """
    device = und_qkv.device
    num_heads_total = num_heads_q + num_heads_k + num_heads_v
    und_qkv = _normalize_qkv(und_qkv, num_heads_total, "und_qkv")
    gen_qkv = _normalize_qkv(gen_qkv, num_heads_total, "gen_qkv")

    und_len, _, head_dim = und_qkv.shape
    gen_len = 0 if gen_qkv is None else gen_qkv.shape[0]
    if gen_qkv is not None:
        assert gen_qkv.shape[2] == head_dim, "gen_qkv 的 head_dim 必须与 und_qkv 一致"

    total = (und_len + gen_len) if cat_indices is None else cat_indices.numel()
    if cat_indices is None:
        src = torch.arange(total, dtype=torch.int64, device=device)  # 恒等映射
    else:
        assert cat_indices.dtype in (torch.int64, torch.int32), "cat_indices 应为 int64"
        src = cat_indices.to(torch.int64)
    assert int(src.min()) >= 0 and int(src.max()) < und_len + gen_len, "cat_indices 越界"

    is_und = src < und_len

    # ---- index：按 src_t 从 und/gen 两段各自 gather，不做 concat ----
    rows = torch.empty(total, num_heads_total, head_dim, dtype=torch.float32, device=device)
    if bool(is_und.any()):
        rows[is_und] = und_qkv[src[is_und]].float()
    if bool((~is_und).any()):
        assert gen_qkv is not None, "src_t >= und_len 但 gen_qkv 为 None"
        rows[~is_und] = gen_qkv[src[~is_und] - und_len].float()

    # ---- split（N 维切 Q/K/V）----
    q = rows[:, :num_heads_q, :]
    k = rows[:, num_heads_q:num_heads_q + num_heads_k, :]
    v = rows[:, num_heads_q + num_heads_k:, :]

    # ---- rmsnorm（按 token 选 und/gen 权重）----
    und_wq = und_weights_q.float()
    und_wk = und_weights_k.float()
    gen_wq = und_wq if gen_weights_q is None else gen_weights_q.float()
    gen_wk = und_wk if gen_weights_k is None else gen_weights_k.float()
    sel = is_und.unsqueeze(-1)
    w_q = torch.where(sel, und_wq.unsqueeze(0), gen_wq.unsqueeze(0))  # [total, D]
    w_k = torch.where(sel, und_wk.unsqueeze(0), gen_wk.unsqueeze(0))
    q = _rmsnorm(q, w_q, norm_eps)
    k = _rmsnorm(k, w_k, norm_eps)

    # ---- MRoPE：三轴 cos/sin 按 axisLut 合并成一份，再做标准 RoPE ----
    half = head_dim // 2
    positions = _normalize_positions(positions, total)
    axis = mrope_axis_map(head_dim, mrope_section).to(device)          # [half]
    pos_sel = positions[axis]                                          # [half, total]
    pos_sel = pos_sel.transpose(0, 1).contiguous()                     # [total, half]
    assert int(pos_sel.min()) >= 0 and int(pos_sel.max()) < cos_sin_cache.shape[0], (
        "positions 超出 cos_sin_cache 的 max_pos 范围"
    )
    col = torch.arange(half, dtype=torch.int64, device=device).unsqueeze(0)  # [1, half]
    cos_sin_f32 = cos_sin_cache.float()
    cos = cos_sin_f32[pos_sel, col]                                    # [total, half]
    sin = cos_sin_f32[pos_sel, col + half]
    q = _rope(q, cos, sin)
    k = _rope(k, cos, sin)

    # V 分支：既不 rmsnorm 也不 rope，仅 index 后直通
    return q, k, v.contiguous()


# 分页 KV Cache 写入 / 读回
def _check_cache(cache, name):
    """cache 固定为连续 BBND：[num_blocks, block_size, N, D]。"""
    assert cache.dim() == 4, f"{name} 期望 4D [num_blocks, block_size, N, D]，实得 {tuple(cache.shape)}"
    assert cache.is_contiguous(), f"{name} 必须内存连续（BBND），本算子不支持非连续布局"
    return cache


def _scatter_cache(cache, slot_mapping, data):
    """按 slot_mapping 原地写入；data: [total, N, D]（已是 cache 的 dtype）。"""
    num_blocks, block_size = cache.shape[0], cache.shape[1]
    assert slot_mapping.dtype in (torch.int64, torch.int32), "slot_mapping 应为 int64（兼容 int32）"
    slot = slot_mapping.to(torch.int64)
    assert int(slot.min()) >= 0 and int(slot.max()) < num_blocks * block_size, "slot_mapping 越界"
    cache[slot // block_size, slot % block_size] = data
    return cache


def gather_cache_rows(cache, slot_mapping):
    """按 slot_mapping 从 cache 读回 [total, N, D]，用于精度比对。"""
    block_size = cache.shape[1]
    slot = slot_mapping.to(torch.int64)
    return cache[slot // block_size, slot % block_size]


# 算子 golden 主入口
def golden_und_gen_qkv_rms_norm_rope_cache(
    und_qkv,
    und_weights_q,
    und_weights_k,
    cos_sin_cache,
    k_cache,
    v_cache,
    slot_mapping,
    positions,
    gen_qkv=None,
    gen_weights_q=None,
    gen_weights_k=None,
    cat_indices=None,
    num_heads_q=8,
    num_heads_k=1,
    num_heads_v=1,
    norm_eps=1e-6,
    mrope_section=None,
    inplace=False,
):
    """返回 (q, k_cache, v_cache)，dtype 全为 bf16。

    k_cache / v_cache 语义与算子一致：调用方预分配，算子原地写入；
    未被 slot_mapping 命中的位置保持传入的原值。默认返回副本（inplace=False），
    传 inplace=True 则直接改写入参。
    """
    q_f32, k_f32, v_f32 = golden_dense(
        und_qkv,
        gen_qkv,
        und_weights_q,
        und_weights_k,
        gen_weights_q,
        gen_weights_k,
        cos_sin_cache,
        positions,
        cat_indices,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        norm_eps,
        mrope_section,
    )

    total = q_f32.shape[0]
    assert slot_mapping.numel() == total, (
        f"slot_mapping 长度 {slot_mapping.numel()} 与 total {total} 不一致"
    )
    assert torch.unique(slot_mapping.to(torch.int64)).numel() == total, (
        "slot_mapping 存在重复 slot，多核写入结果不确定"
    )

    _check_cache(k_cache, "k_cache")
    _check_cache(v_cache, "v_cache")
    k_out = k_cache if inplace else k_cache.clone()
    v_out = v_cache if inplace else v_cache.clone()

    # 写入前 Cast float32 -> bf16（与 Kernel 的 Cast + DataCopy 对齐）
    _scatter_cache(k_out, slot_mapping, k_f32.to(k_out.dtype))
    _scatter_cache(v_out, slot_mapping, v_f32.to(v_out.dtype))

    q_out = q_f32.to(torch.bfloat16)
    return q_out, k_out, v_out

# 典型 case 集
# --------------------------------------------------------------------------- #
# headDim 固定 128；(Hq,Hk,Hv) 取 (8,1,1) / (16,2,2)。
# T = und_len + gen_len 不设上限（算子侧只要求为正，真实上限是 KV Cache 容量）；
# 下面的 case 集覆盖到 64K，再大只受跑用例的机器内存限制。
# block_size 同样不设限，case 里用 block_size 键指定，缺省取 BLOCK_SIZE。
# slot_mode: "shuffled"（分页乱序，默认）/ "contiguous"（顺序占位）
# NOTE: 当前算子实现只支持 gen_qkv / gen_weights_q / gen_weights_k / cat_indices
#       全部提供的场景，因此所有 case 均 gen_len > 0 且 cat_indices 非 None。
#       golden 本身仍支持缺省退化路径（见 golden_dense），待算子放开后可直接加用例。
TYPICAL_CASES = (
    dict(name="t2_min",            heads=(8, 1, 1),  und_len=1,     gen_len=1,
         cat="shuffled", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t8_shuffle",        heads=(8, 1, 1),  und_len=5,     gen_len=3,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t56_cores",         heads=(16, 2, 2), und_len=40,    gen_len=16,
         cat="identity", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t96_mixed",         heads=(16, 2, 2), und_len=80,    gen_len=16,
         cat="identity", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t128_block_edge",   heads=(8, 1, 1),  und_len=127,   gen_len=1,
         cat="shuffled", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t1024_decode",      heads=(8, 1, 1),  und_len=1,     gen_len=1023,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t4096_und_heavy",   heads=(16, 2, 2), und_len=4095,  gen_len=1,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t4096_plain_rope",  heads=(8, 1, 1),  und_len=4095,  gen_len=1,
         cat="identity", slot_mode="contiguous", mrope_section=[],           big=False),
    dict(name="t10752_typical_h8", heads=(8, 1, 1),  und_len=6652,  gen_len=4100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t10752_typical_h16", heads=(16, 2, 2), und_len=6652, gen_len=4100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    # block_size 覆盖：算子对 Bs 无假设，这几条把 2 的幂与非 2 的幂、小块与大块都过一遍
    dict(name="t64_bs16",          heads=(8, 1, 1),  und_len=40,    gen_len=24,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=16),
    dict(name="t200_bs64",         heads=(16, 2, 2), und_len=150,   gen_len=50,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=64),
    dict(name="t300_bs100",        heads=(8, 1, 1),  und_len=200,   gen_len=100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=100),
    dict(name="t512_bs256",        heads=(8, 1, 1),  und_len=256,   gen_len=256,
         cat="identity", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False, block_size=256),
    dict(name="t1024_bs512",       heads=(16, 2, 2), und_len=512,   gen_len=512,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=512),
    dict(name="t65536_h8",         heads=(8, 1, 1),  und_len=32768, gen_len=32768,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=True),
    dict(name="t65536_h16",        heads=(16, 2, 2), und_len=32768, gen_len=32768,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=True),
)


# --------------------------------------------------------------------------- #
# 泛化 case 集
# --------------------------------------------------------------------------- #
# TYPICAL_CASES 是稳定的回归基线，刻意把随机种子与标量参数钉死，方便逐次比对；
# 代价是若干维度只覆盖了单点：mrope_section 17 例里 16 例是同一个 [16,16,16]，
# seed / norm_eps / max_pos 更是全集一个值。下面这组专门补这些维度，
# T 都取小值（CPU golden 是逐 token 向量化实现，小 T 才跑得快），只求覆盖不求规模。
#
# 轴映射 α(l) 的规则是 `l%3==1 且 l<3*sec[1] -> H`、`l%3==2 且 l<3*sec[2] -> W`、其余 -> T，
# 只读 sec[1]/sec[2]，sec[0] 不参与计算。所以下面按「截断点落在哪」来选值，
# 而不是按三段和是否等于 D/2 来选。
#
# 新增维度都做过区分力自检（扰动 golden 看用例是否会 FAIL）：忽略 mrope_section -> 4.6~64.8%
# 元素不达标；忽略 cat_indices -> 96.6~99.8%；忽略 norm_eps 需要把输入幅度压到 x_scale=1e-2
# 才测得出（默认幅度下 eps 对 rms 的影响低于 bf16 判据，g_eps_tiny 更是原理上无区分力，
# 只验极小 eps + 小幅度输入下 rsqrt 不出 inf/nan）。
GENERALIZED_CASES = (
    # ---- mrope_section：轴映射的各种截断形态 ----
    dict(name="g_mrope_zeros",     heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 0, 0],   big=False),
    dict(name="g_mrope_t_only",    heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[64, 0, 0],  big=False),
    dict(name="g_mrope_no_t",      heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 21, 21], big=False),
    dict(name="g_mrope_w_only",    heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 0, 21],  big=False),
    dict(name="g_mrope_sum_full",  heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[32, 16, 16], big=False),
    dict(name="g_mrope_tiny_sec",  heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[10, 1, 2],  big=False),
    # sec[1] 大到 3*sec[1] 越过 D/2：所有 l%3==1 都归 H，覆盖「截断点在区间外」
    dict(name="g_mrope_h_all",     heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 44, 20], big=False),

    # ---- cat_indices 的分布形态：决定 undMask 与 gamma 选择 ----
    dict(name="g_cat_all_und",     heads=(8, 1, 1),  und_len=50,  gen_len=14,
         cat="all_und", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_cat_all_gen",     heads=(8, 1, 1),  und_len=14,  gen_len=50,
         cat="all_gen", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_cat_reverse",     heads=(16, 2, 2), und_len=40,  gen_len=24,
         cat="reverse", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_cat_dup",         heads=(8, 1, 1),  und_len=30,  gen_len=34,
         cat="dup",     slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),

    # ---- slot_mapping 的分布形态 ----
    dict(name="g_slot_reverse",    heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="reverse", mrope_section=[16, 16, 16], big=False),
    dict(name="g_slot_high",       heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="high",    mrope_section=[16, 16, 16], big=False),

    # ---- tile / 分核边界：ubFactor 上限是 64，尾块与核数边界都过一遍 ----
    dict(name="g_t55_under_cores", heads=(8, 1, 1),  und_len=30,  gen_len=25,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_t57_over_cores",  heads=(8, 1, 1),  und_len=30,  gen_len=27,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_t65_tail1",       heads=(16, 2, 2), und_len=32,  gen_len=33,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_t113_tail1",      heads=(8, 1, 1),  und_len=56,  gen_len=57,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),

    # ---- 标量参数：eps 与 cos_sin_cache 行数的边界 ----
    # x_scale 压到 1e-2 让 mean(x^2) ~ 1e-4，eps 才成为 rms 的主导项；
    # 用默认幅度这两条对 eps 没有任何区分力（实测扰动 eps->0 时不达标元素为 0）
    dict(name="g_eps_large",       heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         norm_eps=1e-3, x_scale=1e-2),
    dict(name="g_eps_mid",         heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         norm_eps=1e-4, x_scale=1e-2),
    # NOTE: 这条对「实现是否用了 eps」没有区分力，也不可能有——eps=1e-8 相对 mean(x^2)~1e-4
    #       本就可忽略，"忽略 eps" 在数值上就是正确答案。它验的是另一件事：
    #       极小 eps + 小幅度输入下 rsqrt 不出 inf/nan。
    dict(name="g_eps_tiny",        heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         norm_eps=1e-8, x_scale=1e-2),
    # max_pos=1 时所有 position 只能取 0，cos/sin 退化成同一行；覆盖 cache 只有一行的边界
    dict(name="g_maxpos_1",        heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         max_pos=1),
    dict(name="g_maxpos_2",        heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         max_pos=2),

    # ---- 换随机输入：同一 shape 换 4 个种子，覆盖「不同数值分布」而非不同 shape ----
    dict(name="g_seed_101",        heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=101),
    dict(name="g_seed_202",        heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=202),
    dict(name="g_seed_303",        heads=(8, 1, 1),  und_len=200, gen_len=112,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=303),
    dict(name="g_seed_404",        heads=(16, 2, 2), und_len=200, gen_len=112,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=404),

    # ---- 组合：多个非默认维度同时偏离，防止「单维各自过、组合起来挂」 ----
    dict(name="g_combo_a",         heads=(16, 2, 2), und_len=61,  gen_len=52,
         cat="all_gen", slot_mode="high",    mrope_section=[0, 21, 21], big=False,
         block_size=37, seed=505, norm_eps=1e-4),
    dict(name="g_combo_b",         heads=(8, 1, 1),  und_len=99,  gen_len=1,
         cat="dup",     slot_mode="reverse", mrope_section=[10, 1, 2], big=False,
         block_size=7,  seed=606, max_pos=8),
)


def build_case(spec, device="cpu", seed=7, max_pos=4096, init_cache="randn"):
    """按 case spec 造一套完整输入，返回可直接 ``**kwargs`` 调 golden 的 dict。

    额外键（下划线开头）是 case 元信息，调用前用 ``pop`` 去掉即可，
    ``run_case`` 已代为处理。
    """
    # case 可以逐条覆盖随机种子与标量参数：默认集把它们钉死是为了让回归基线可复现，
    # 泛化集则靠改这几项把「同一 shape 换一批随机输入 / 换一组标量」也纳入覆盖
    seed = int(spec.get("seed", seed))
    max_pos = int(spec.get("max_pos", max_pos))
    norm_eps = float(spec.get("norm_eps", 1e-6))
    # qkv 的整体幅度。默认 1.0 时 mean(x^2) ~ 1，eps 在 [1e-8, 1e-3] 内只改变 rms 约 5e-4 相对量，
    # 落在 bf16 判据（rtol 4e-3）之下 —— 也就是说默认幅度下**测不出 eps 用没用**。
    # 把幅度压到 1e-2 后 mean(x^2) ~ 1e-4，eps 才成为 rms 的主导项，用例才有区分力。
    x_scale = float(spec.get("x_scale", 1.0))
    torch.manual_seed(seed)
    hq, hk, hv = spec["heads"]
    assert (hq, hk, hv) in HEAD_COMBOS, f"{spec['name']}: 不在支持的 head 组合内"
    n = hq + hk + hv
    d = HEAD_DIM
    und_len, gen_len = spec["und_len"], spec["gen_len"]
    total = und_len + gen_len
    assert total >= 1, "T 必须为正"

    # KV Cache 预留：按 block_size 向上取整再多给 2 个 block，制造未命中位置
    # block_size 由 case 指定（默认 BLOCK_SIZE）：算子不限制 Bs，cache 展平后 slot 直接
    # 当行号用，Bs 只影响 [Bn, Bs, ...] 怎么切分同一片连续内存
    bs = int(spec.get("block_size", BLOCK_SIZE))
    assert bs >= 1, "block_size 必须为正"
    # extra_blocks 控制冗余块数：默认 2 留出未命中位置，取 0 则 Bn*Bs 贴着 ceil(T/Bs)*Bs，
    # 覆盖 CheckKvCacheValid 的 blockNum_*blockSize_ >= totalTokens_ 下界（slot 铺满整片 cache）
    extra_blocks = int(spec.get("extra_blocks", 2))
    assert extra_blocks >= 0, "extra_blocks 不能为负"
    num_blocks = (total + bs - 1) // bs + extra_blocks
    k_shape, v_shape = (num_blocks, bs, hk, d), (num_blocks, bs, hv, d)  # 连续 BBND
    init = torch.randn if init_cache == "randn" else torch.zeros

    # cat_indices 取值只要求落在 [0, T-1]，不要求是排列：算子按 src_t < und_len 逐 token
    # 选段与选 gamma，所以 und/gen 的**分布形态**才是被覆盖的维度
    cat_mode = spec["cat"]
    if cat_mode == "none":
        cat_indices = None
    elif cat_mode == "identity":
        cat_indices = torch.arange(total, dtype=torch.int64, device=device)
    elif cat_mode == "reverse":
        # 逆序：und/gen 的边界翻到另一头，und 段落在 tile 尾部
        cat_indices = torch.arange(total - 1, -1, -1, dtype=torch.int64, device=device)
    elif cat_mode == "all_und":
        # 全部指向 und 段：undMask 恒为全 1，只走 und 的 gamma
        cat_indices = torch.randint(0, und_len, (total,), dtype=torch.int64, device=device)
    elif cat_mode == "all_gen":
        # 全部指向 gen 段：undMask 恒为 0，只走 gen 的 gamma
        assert gen_len > 0, "all_gen 需要 gen_len > 0"
        cat_indices = torch.randint(und_len, total, (total,), dtype=torch.int64, device=device)
    elif cat_mode == "dup":
        # 允许重复源：同一个源 token 被多个输出位置引用（接口只约束值域，不约束唯一性）
        cat_indices = torch.randint(0, total, (total,), dtype=torch.int64, device=device)
    else:  # shuffled：und/gen 交错，模拟真实 cat_indices
        cat_indices = torch.randperm(total, device=device).to(torch.int64)
    assert cat_mode == "none" or gen_len > 0 or und_len == total

    # slot_mapping 取值必须唯一（重复 slot 多核写冲突，结果不确定），下面每种模式都保证这点
    slot_mode = spec["slot_mode"]
    capacity = num_blocks * bs
    if slot_mode == "contiguous":
        slot_mapping = torch.arange(total, dtype=torch.int64, device=device)
    elif slot_mode == "reverse":
        slot_mapping = torch.arange(total - 1, -1, -1, dtype=torch.int64, device=device)
    elif slot_mode == "high":
        # 贴着容量上界写，覆盖最大 slot = Bn*Bs-1 这个边界行
        slot_mapping = torch.arange(capacity - total, capacity, dtype=torch.int64, device=device)
    else:
        slot_mapping = torch.randperm(capacity, device=device)[:total].to(torch.int64)

    case = dict(
        und_qkv=(torch.randn(und_len, n, d, dtype=torch.bfloat16, device=device) * x_scale),
        und_weights_q=torch.randn(d, dtype=torch.bfloat16, device=device),
        und_weights_k=torch.randn(d, dtype=torch.bfloat16, device=device),
        cos_sin_cache=make_cos_sin_cache(max_pos, d, device=device),
        k_cache=init(k_shape, dtype=torch.bfloat16, device=device),
        v_cache=init(v_shape, dtype=torch.bfloat16, device=device),
        slot_mapping=slot_mapping,
        positions=torch.randint(0, max_pos, (3, total), dtype=torch.int64, device=device),
        gen_qkv=((torch.randn(gen_len, n, d, dtype=torch.bfloat16, device=device) * x_scale)
                 if gen_len > 0 else None),
        gen_weights_q=(torch.randn(d, dtype=torch.bfloat16, device=device) if gen_len > 0 else None),
        gen_weights_k=(torch.randn(d, dtype=torch.bfloat16, device=device) if gen_len > 0 else None),
        cat_indices=cat_indices,
        num_heads_q=hq,
        num_heads_k=hk,
        num_heads_v=hv,
        norm_eps=norm_eps,
        mrope_section=spec["mrope_section"],
    )
    case["_meta"] = dict(name=spec["name"], total=total, heads=(hq, hk, hv),
                         num_blocks=num_blocks, block_size=bs, seed=seed,
                         max_pos=max_pos, norm_eps=norm_eps, x_scale=x_scale,
                         cat=cat_mode, slot_mode=slot_mode,
                         mrope_section=spec["mrope_section"])
    return case


# --------------------------------------------------------------------------- #
# 自检：仅验证 case 集构造与 golden 可跑通（纯 CPU，不涉及 NPU）
# --------------------------------------------------------------------------- #
def _self_check(include_big=False):
    print(f"golden case set (headDim={HEAD_DIM}, block_size={BLOCK_SIZE}, heads in "
          f"{{{HEAD_COMBOS[0]}, {HEAD_COMBOS[1]}}})")
    for spec in TYPICAL_CASES:
        total = spec["und_len"] + spec["gen_len"]
        if spec["big"] and not include_big:
            print(f"  [SKIP] {spec['name']:<22} T={total}")
            continue
        case = build_case(spec)
        meta = case.pop("_meta")
        q, k_cache, v_cache = golden_und_gen_qkv_rms_norm_rope_cache(**case)
        assert q.shape == (total, meta["heads"][0], HEAD_DIM)
        assert q.dtype == torch.bfloat16
        print(f"  [OK]   {meta['name']:<22} T={total:<6} heads={meta['heads']} "
              f"q={tuple(q.shape)} k_cache={tuple(k_cache.shape)}")
    print("golden self-check PASS")


if __name__ == "__main__":
    import sys
    _self_check(include_big="--full" in sys.argv)


# TTK 插件：输入构造与 golden 注册
# k_cache / v_cache 是原地更新：IR 里输出名与输入名相同，TTK 会据此自动填
# output_inplace_indexes，kernel CSV 不必显式给出。

# IR 输入序（与 op_host/und_gen_qkv_rms_norm_rope_cache_def.cpp 一致）
IDX_UND_QKV = 0
IDX_UND_WEIGHTS_Q = 1
IDX_UND_WEIGHTS_K = 2
IDX_COS_SIN_CACHE = 3
IDX_K_CACHE = 4
IDX_V_CACHE = 5
IDX_SLOT_MAPPING = 6
IDX_POSITIONS = 7
IDX_GEN_QKV = 8
IDX_GEN_WEIGHTS_Q = 9
IDX_GEN_WEIGHTS_K = 10
IDX_CAT_INDICES = 11

TTK_INPUT_NUM = 12


def _np_to_torch(array):
    """numpy -> torch；bf16 经 float32 中转（numpy 侧由 ml_dtypes.bfloat16 承载）。"""
    import numpy as np

    if array is None:
        return None
    if str(array.dtype) == "bfloat16":
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(array))


def _torch_to_np(tensor, like_dtype):
    """torch -> numpy，dtype 对齐到 TTK 侧对应输出的 numpy dtype。"""
    if tensor.dtype == torch.bfloat16:
        out = tensor.float().numpy()
    else:
        out = tensor.numpy()
    return out.astype(like_dtype) if like_dtype is not None else out


def _seed_of(kwargs):
    """按用例名派生种子：同一条用例每次跑拿到同一组索引，便于复现。"""
    name = str(kwargs.get("testcase_name", "und_gen_qkv_rms_norm_rope_cache"))
    return abs(hash(name)) % (2 ** 31)


def _gen_index_values(arrays, kwargs):
    """按用例名后缀生成三个索引类输入的合法取值（numpy），kernel / aclnn 两条通路共用。

    返回 (slots, cat, positions)，未提供对应输入时该项为 None。
    rng 抽取顺序固定为 slot -> cat -> positions，改动会让既有用例的输入整体漂移。
    """
    import numpy as np

    slot_mapping = arrays[IDX_SLOT_MAPPING]
    k_cache = arrays[IDX_K_CACHE]
    total = int(slot_mapping.shape[0])
    block_num, block_size = int(k_cache.shape[0]), int(k_cache.shape[1])
    max_slot = block_num * block_size
    if max_slot < total:
        raise RuntimeError(
            "KV Cache 容量不足：Bn*Bs=%d < T=%d，请调整用例的 k_cache/v_cache shape"
            % (max_slot, total)
        )

    name = str(kwargs.get("testcase_name", ""))
    rng = np.random.default_rng(_seed_of(kwargs))

    if "_slotseq" in name:
        slots = np.arange(total)
    else:
        slots = rng.choice(max_slot, size=total, replace=False)

    cat = None
    if arrays[IDX_CAT_INDICES] is not None:
        und_len = int(arrays[IDX_UND_QKV].shape[0])
        if "_catid" in name:
            cat = np.arange(total)
        elif "_catrev" in name:
            cat = np.arange(total)[::-1].copy()
        elif "_catund" in name:
            cat = rng.integers(0, und_len, size=total)
        elif "_catgen" in name:
            cat = rng.integers(und_len, total, size=total)
        else:
            cat = rng.permutation(total)

    positions = None
    if arrays[IDX_POSITIONS] is not None:
        max_pos = int(arrays[IDX_COS_SIN_CACHE].shape[0])
        positions = rng.integers(0, max_pos, size=tuple(arrays[IDX_POSITIONS].shape))

    return slots, cat, positions


def ttk_input(*inputs, **kwargs):
    """把三个索引类输入改写成合法取值，随机 range 做不到。

    其中 slot_mapping 必须**互不重复**：重复槽位会让多个核写同一 cache 行，写入顺序
    与结果都不确定，随机整数几乎必然撞号，不改写就会得到一个每次跑都不一样的比对。

    形态由用例名后缀决定，与 CSV 的 slot_form / cat_form 列一一对应：
      _slotseq                              -> slot_mapping 连续
      _catid / _catrev / _catund / _catgen  -> cat_indices 形态
    """
    arrays = list(inputs)
    if len(arrays) != TTK_INPUT_NUM:
        raise RuntimeError(
            "und_gen_qkv_rms_norm_rope_cache 期望 %d 个输入，实际收到 %d 个"
            % (TTK_INPUT_NUM, len(arrays))
        )

    slots, cat, positions = _gen_index_values(arrays, kwargs)
    arrays[IDX_SLOT_MAPPING] = slots.astype(arrays[IDX_SLOT_MAPPING].dtype)
    if cat is not None:
        arrays[IDX_CAT_INDICES] = cat.astype(arrays[IDX_CAT_INDICES].dtype)
    if positions is not None:
        arrays[IDX_POSITIONS] = positions.astype(arrays[IDX_POSITIONS].dtype)

    return arrays


def ttk_golden(und_qkv,
               und_weights_q,
               und_weights_k,
               cos_sin_cache,
               k_cache,
               v_cache,
               slot_mapping,
               positions,
               gen_qkv=None,
               gen_weights_q=None,
               gen_weights_k=None,
               cat_indices=None,
               **kwargs):
    '''Kernel golden for und_gen_qkv_rms_norm_rope_cache.

    参数序与 @und_gen_qkv_rms_norm_rope_cache_def.cpp 的输入一致（不含输出），
    张量为 numpy.ndarray，bf16 由 ml_dtypes.bfloat16 承载，kwargs 带算子属性。
    返回写入 slot 后的整块 (q, k_cache, v_cache)。
    '''
    mrope_section = kwargs.get("mrope_section", None)
    if mrope_section is not None:
        mrope_section = list(mrope_section)

    q_t, k_t, v_t = golden_und_gen_qkv_rms_norm_rope_cache(
        _np_to_torch(und_qkv),
        _np_to_torch(und_weights_q),
        _np_to_torch(und_weights_k),
        _np_to_torch(cos_sin_cache),
        _np_to_torch(k_cache),
        _np_to_torch(v_cache),
        _np_to_torch(slot_mapping),
        _np_to_torch(positions),
        gen_qkv=_np_to_torch(gen_qkv),
        gen_weights_q=_np_to_torch(gen_weights_q),
        gen_weights_k=_np_to_torch(gen_weights_k),
        cat_indices=_np_to_torch(cat_indices),
        num_heads_q=int(kwargs.get("num_heads_q", 8)),
        num_heads_k=int(kwargs.get("num_heads_k", 1)),
        num_heads_v=int(kwargs.get("num_heads_v", 1)),
        norm_eps=float(kwargs.get("norm_eps", 1e-6)),
        mrope_section=mrope_section,
        inplace=False,
    )

    kv_dtype = k_cache.dtype
    return (
        _torch_to_np(q_t, kv_dtype),
        _torch_to_np(k_t, kv_dtype),
        _torch_to_np(v_t, kv_dtype),
    )


# --------------------------------------------------------------------------- #
# aclnn 层插件
# --------------------------------------------------------------------------- #
# 与 kernel 层的三处差异，改动前先看清楚：
#   1. 入参序是 aclnn 头文件的形参序（12 个张量 + 5 个属性 + qOut），
#      不是 _def.cpp 的输入序，属性走位置参数而不是 kwargs；
#   2. input 插件的返回值被 TTK 丢弃（op_api/input_generation.py 只调不收），
#      必须原地改写张量；
#   3. golden 的返回序要与 CSV 的 output_tensor_indexes 声明序一致，
#      本算子声明为 (12, 4, 5) 即 (q, k_cache, v_cache)。
ACLNN_ATTR_NUM = 5


def _aclnn_split_args(args):
    """拆 aclnn 形参：前 12 个张量 + 5 个属性（qOut 及其后的形参 golden 用不到）。"""
    if len(args) < TTK_INPUT_NUM + ACLNN_ATTR_NUM:
        raise RuntimeError(
            "aclnnUndGenQkvRmsNormRopeCache 期望至少 %d 个形参，实际收到 %d 个"
            % (TTK_INPUT_NUM + ACLNN_ATTR_NUM, len(args))
        )
    tensors = list(args[:TTK_INPUT_NUM])
    hq, hk, hv, eps, sec = args[TTK_INPUT_NUM:TTK_INPUT_NUM + ACLNN_ATTR_NUM]
    attrs = dict(num_heads_q=int(hq), num_heads_k=int(hk), num_heads_v=int(hv),
                 norm_eps=float(eps),
                 mrope_section=(list(sec) if sec is not None else None))
    return tensors, attrs


def _as_torch(x):
    """numpy（bf16 由 ml_dtypes 承载）或 torch 张量 -> CPU torch 张量。"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return _np_to_torch(x)


def _write_back(dst, values):
    """把 numpy 取值原地写回 dst（numpy 或 torch 均可）。"""
    if isinstance(dst, torch.Tensor):
        dst.copy_(torch.from_numpy(values.astype("int64")).to(dst.dtype))
    else:
        dst[...] = values.astype(dst.dtype)


def aclnn_input(*args, **kwargs):
    """aclnn 层的索引改写，取值规则与 ttk_input 完全一致（共用 _gen_index_values）。"""
    tensors, _ = _aclnn_split_args(args)
    slots, cat, positions = _gen_index_values(tensors, kwargs)
    _write_back(tensors[IDX_SLOT_MAPPING], slots)
    if cat is not None:
        _write_back(tensors[IDX_CAT_INDICES], cat)
    if positions is not None:
        _write_back(tensors[IDX_POSITIONS], positions)


def aclnn_golden(*args, **kwargs):
    """aclnn 层 golden，返回 (q, k_cache, v_cache)，与 output_tensor_indexes=(12,4,5) 对齐。"""
    tensors, attrs = _aclnn_split_args(args)
    q_t, k_t, v_t = golden_und_gen_qkv_rms_norm_rope_cache(
        *[_as_torch(t) for t in tensors[:8]],
        gen_qkv=_as_torch(tensors[IDX_GEN_QKV]),
        gen_weights_q=_as_torch(tensors[IDX_GEN_WEIGHTS_Q]),
        gen_weights_k=_as_torch(tensors[IDX_GEN_WEIGHTS_K]),
        cat_indices=_as_torch(tensors[IDX_CAT_INDICES]),
        inplace=False,
        **attrs,
    )
    ref = tensors[IDX_K_CACHE]
    if isinstance(ref, torch.Tensor):
        return q_t.to(ref.dtype), k_t.to(ref.dtype), v_t.to(ref.dtype)
    return (_torch_to_np(q_t, ref.dtype),
            _torch_to_np(k_t, ref.dtype),
            _torch_to_np(v_t, ref.dtype))


__input__ = {
    "kernel": {
        "und_gen_qkv_rms_norm_rope_cache": "ttk_input"
    },
    "aclnn": {
        "aclnnUndGenQkvRmsNormRopeCache": "aclnn_input"
    }
}

__golden__ = {
    "kernel": {
        "und_gen_qkv_rms_norm_rope_cache": "ttk_golden"
    },
    "aclnn": {
        "aclnnUndGenQkvRmsNormRopeCache": "aclnn_golden"
    }
}
