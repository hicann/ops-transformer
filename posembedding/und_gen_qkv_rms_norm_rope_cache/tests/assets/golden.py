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
"""UndGenQkvRmsNormRopeCache 的 golden 参考实现，同时是本算子的 TTK TestSpec。

本文件是本算子**唯一的 golden 实现**，自包含、不依赖同目录以外的任何文件，
供下列使用方共用：

  - ``examples/test_torch_und_gen_qkv_rms_norm_rope_cache.py``：torch 接口上板精度比对
  - ``tests/st/arch35/ttk_kernel_*.csv``：TTK kernel 用例，经 ``__spec__`` 注册的
    ``UndGenQkvRmsNormRopeCacheTestSpec`` 接入

    python3 -m ttk kernel -i <csv> --plugin <此文件>

  - ``tests/st/arch35/ttk_aclnn_*.csv``：TTK aclnn 用例，经
    ``AclnnUndGenQkvRmsNormRopeCacheTestSpec`` 接入

    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
    python3 -m ttk aclnn -i <csv> --plugin <此文件>

  - E2E（torch）用例，经 ``TorchUndGenQkvRmsNormRopeCacheTestSpec`` 接入，
    CSV 由同目录 ``make_e2e_csv.py`` 现场生成

    python3 -m ttk e2e -i <csv> --plugin <此文件>

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

import logging
import zlib

import torch

# TestSpec 注册：kernel 用 CSV 的 op_name，aclnn / e2e 用 CSV 的 api_name
__spec__ = {
    "und_gen_qkv_rms_norm_rope_cache": "UndGenQkvRmsNormRopeCacheTestSpec",
    "aclnnUndGenQkvRmsNormRopeCache": "AclnnUndGenQkvRmsNormRopeCacheTestSpec",
    "torch.ops.cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache": "TorchUndGenQkvRmsNormRopeCacheTestSpec",
}

__all__ = [
    "HEAD_DIM",
    "BLOCK_SIZE",
    "HEAD_COMBOS",
    "make_cos_sin_cache",
    "mrope_axis_map",
    "golden_und_gen_qkv_rms_norm_rope_cache",
    "golden_dense",
    "gather_cache_rows",
    "UndGenQkvRmsNormRopeCacheTestSpec",
    "AclnnUndGenQkvRmsNormRopeCacheTestSpec",
    "TorchUndGenQkvRmsNormRopeCacheTestSpec",
]

# 本期支持范围
HEAD_DIM = 128
HEAD_COMBOS = ((8, 1, 1), (16, 2, 2))
# 算子对 block_size 无约束，这只是 case 未指定 block_size 时的默认值
BLOCK_SIZE = 128


# 工具
def make_cos_sin_cache(max_pos, head_dim, device="cpu", base=10000.0):
    """构造 cos_sin_cache [max_pos, head_dim]，前半 cos 后半 sin（与竞品一致）。"""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
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
        assert x.shape[1] % num_heads_total == 0, (
            f"{name} 的 hidden 无法被 N={num_heads_total} 整除"
        )
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
    assert positions.shape == (3, total), (
        f"positions 期望 [3, {total}]，实得 {tuple(positions.shape)}"
    )
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
    assert int(src.min()) >= 0 and int(src.max()) < und_len + gen_len, (
        "cat_indices 越界"
    )

    is_und = src < und_len

    # ---- index：按 src_t 从 und/gen 两段各自 gather，不做 concat ----
    rows = torch.empty(
        total, num_heads_total, head_dim, dtype=torch.float32, device=device
    )
    if bool(is_und.any()):
        rows[is_und] = und_qkv[src[is_und]].float()
    if bool((~is_und).any()):
        assert gen_qkv is not None, "src_t >= und_len 但 gen_qkv 为 None"
        rows[~is_und] = gen_qkv[src[~is_und] - und_len].float()

    # ---- split（N 维切 Q/K/V）----
    q = rows[:, :num_heads_q, :]
    k = rows[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = rows[:, num_heads_q + num_heads_k :, :]

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
    axis = mrope_axis_map(head_dim, mrope_section).to(device)  # [half]
    pos_sel = positions[axis]  # [half, total]
    pos_sel = pos_sel.transpose(0, 1).contiguous()  # [total, half]
    assert int(pos_sel.min()) >= 0 and int(pos_sel.max()) < cos_sin_cache.shape[0], (
        "positions 超出 cos_sin_cache 的 max_pos 范围"
    )
    col = torch.arange(half, dtype=torch.int64, device=device).unsqueeze(0)  # [1, half]
    cos_sin_f32 = cos_sin_cache.float()
    cos = cos_sin_f32[pos_sel, col]  # [total, half]
    sin = cos_sin_f32[pos_sel, col + half]
    q = _rope(q, cos, sin)
    k = _rope(k, cos, sin)

    # V 分支：既不 rmsnorm 也不 rope，仅 index 后直通
    return q, k, v.contiguous()


# 分页 KV Cache 写入 / 读回
def _check_cache(cache, name):
    """cache 固定为连续 BBND：[num_blocks, block_size, N, D]。"""
    assert cache.dim() == 4, (
        f"{name} 期望 4D [num_blocks, block_size, N, D]，实得 {tuple(cache.shape)}"
    )
    assert cache.is_contiguous(), f"{name} 必须内存连续（BBND），本算子不支持非连续布局"
    return cache


def _scatter_cache(cache, slot_mapping, data):
    """按 slot_mapping 原地写入；data: [total, N, D]（已是 cache 的 dtype）。"""
    num_blocks, block_size = cache.shape[0], cache.shape[1]
    assert slot_mapping.dtype in (torch.int64, torch.int32), (
        "slot_mapping 应为 int64（兼容 int32）"
    )
    slot = slot_mapping.to(torch.int64)
    assert int(slot.min()) >= 0 and int(slot.max()) < num_blocks * block_size, (
        "slot_mapping 越界"
    )
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


# --------------------------------------------------------------------------- #
# TTK TestSpec：输入构造与 golden
# --------------------------------------------------------------------------- #
# 注册见文件顶部的 __spec__；三条通路各一个 spec 类，共用下面的私有 helper。
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
    """按用例名派生种子。用 crc32 不用内置 hash()——后者随 PYTHONHASHSEED 变。"""
    name = str(kwargs.get("testcase_name", "und_gen_qkv_rms_norm_rope_cache"))
    return zlib.crc32(name.encode("utf-8"))


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


# --------------------------------------------------------------------------- #
# 精度判据：cross_check / L1，标杆用 golden 自身
# --------------------------------------------------------------------------- #
try:
    from ttk.core_modules.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.comparison.resolve import resolve_tolerance

    _TTK_CROSS_CHECK_AVAILABLE = True
except ImportError:  # 不在 ops-test-kit 环境里（如直接跑本文件自检）
    _TTK_CROSS_CHECK_AVAILABLE = False

TOLERANCE = {"bfloat16": {"standard": "cross_check", "level": "L1"}}


class _SpecBase:
    """三条通路共用的判据。"""

    tolerance = TOLERANCE

    def compare(*outputs, **kwargs):
        return _cross_check(*outputs)


def _dtype_name(array):
    """numpy（bf16 由 ml_dtypes 承载）或 torch 张量 -> resolve_tolerance 认的 dtype 名。"""
    import numpy as np

    if isinstance(array, torch.Tensor):
        return str(array.dtype).rsplit(".", 1)[-1]
    return np.dtype(array.dtype).name


def _cross_check(*outputs):
    """outputs = (NPU_0..n-1, golden_0..n-1)，逐输出返回 compare 契约的 dict。

    dtype 取 NPU 输出：golden 在 cross_check 下会被 Promote 抬成 fp32。
    """
    if not _TTK_CROSS_CHECK_AVAILABLE:
        raise RuntimeError(
            "ttk.core_modules.comparison 不可用：请在 ops-test-kit checkout 下运行"
        )
    half = len(outputs) // 2
    results = []
    for idx, (npu, gold) in enumerate(zip(outputs[:half], outputs[half:])):
        dtype_str = _dtype_name(npu)
        params = resolve_tolerance(TOLERANCE, None, None, [dtype_str], None)[0].params
        precision, log, is_pass, metrics = CrossCheckComparison(
            npu, gold, idx, dtype_str, params, third_party=gold
        ).compare()
        logging.info(
            "[cross_check] output %d %s %s", idx, precision, metrics.get("result")
        )
        results.append(
            {
                "pass": is_pass,
                "precision": precision,
                "metrics": metrics,
                "error_info": None if is_pass else log,
            }
        )
    return results


class UndGenQkvRmsNormRopeCacheTestSpec(_SpecBase):
    """UndGenQkvRmsNormRopeCache 的 kernel / GEIR 通路测试规范。

    参数序与 op_host/und_gen_qkv_rms_norm_rope_cache_def.cpp 的输入一致（不含输出），
    张量为 numpy.ndarray，bf16 由 ml_dtypes.bfloat16 承载，属性走 kwargs。

    判据见上方 TOLERANCE：cross_check / L1，标杆即 golden 自身。
    """

    def customize_inputs(
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
        **kwargs,
    ):
        """把三个索引类输入改写成合法取值，随机 range 做不到。

        其中 slot_mapping 必须**互不重复**：重复槽位会让多个核写同一 cache 行，写入顺序
        与结果都不确定，随机整数几乎必然撞号，不改写就会得到一个每次跑都不一样的比对。

        形态由用例名后缀决定，与 CSV 的 slot_form / cat_form 列一一对应：
          _slotseq                              -> slot_mapping 连续
          _catid / _catrev / _catund / _catgen  -> cat_indices 形态
        """
        arrays = [
            und_qkv,
            und_weights_q,
            und_weights_k,
            cos_sin_cache,
            k_cache,
            v_cache,
            slot_mapping,
            positions,
            gen_qkv,
            gen_weights_q,
            gen_weights_k,
            cat_indices,
        ]

        slots, cat, new_positions = _gen_index_values(arrays, kwargs)
        arrays[IDX_SLOT_MAPPING] = slots.astype(slot_mapping.dtype)
        if cat is not None:
            arrays[IDX_CAT_INDICES] = cat.astype(cat_indices.dtype)
        if new_positions is not None:
            arrays[IDX_POSITIONS] = new_positions.astype(positions.dtype)

        return arrays

    def golden(
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
        **kwargs,
    ):
        """返回写入 slot 后的整块 (q, k_cache, v_cache)。"""
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
# aclnn / E2E 通路
# --------------------------------------------------------------------------- #
# 与 kernel 通路的三处差异，改动前先看清楚：
#   1. 入参序是 aclnn 头文件的形参序（12 个张量 + 5 个属性 + qOut），
#      不是 _def.cpp 的输入序，属性走位置参数而不是 kwargs；
#   2. customize_inputs 的返回值被 TTK 丢弃（op_api/input_generation.py 只调不收），
#      必须原地改写张量；
#   3. golden 的返回序要与 CSV 的 output_tensor_indexes 声明序一致，
#      本算子声明为 (12, 4, 5) 即 (q, k_cache, v_cache)。
# E2E 与 aclnn 只差两点：没有尾部的 qOut，且 5 个属性因 torch schema 里在 `*` 之后
# 是 keyword-only 而走关键字下发，故两个 spec 各自声明签名、共用下面两个实现。


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


def _rewrite_indices(tensors, kwargs):
    """索引改写，取值规则与 kernel 侧完全一致（共用 _gen_index_values）。"""
    slots, cat, positions = _gen_index_values(tensors, kwargs)
    _write_back(tensors[IDX_SLOT_MAPPING], slots)
    if cat is not None:
        _write_back(tensors[IDX_CAT_INDICES], cat)
    if positions is not None:
        _write_back(tensors[IDX_POSITIONS], positions)
    return tensors


def _golden_from_tensors(
    tensors, num_heads_q, num_heads_k, num_heads_v, norm_eps, mrope_section
):
    """返回 (q, k_cache, v_cache)，dtype 随 k_cache（numpy 或 torch 都认）。"""
    q_t, k_t, v_t = golden_und_gen_qkv_rms_norm_rope_cache(
        *[_as_torch(t) for t in tensors[:8]],
        gen_qkv=_as_torch(tensors[IDX_GEN_QKV]),
        gen_weights_q=_as_torch(tensors[IDX_GEN_WEIGHTS_Q]),
        gen_weights_k=_as_torch(tensors[IDX_GEN_WEIGHTS_K]),
        cat_indices=_as_torch(tensors[IDX_CAT_INDICES]),
        num_heads_q=int(num_heads_q),
        num_heads_k=int(num_heads_k),
        num_heads_v=int(num_heads_v),
        norm_eps=float(norm_eps),
        mrope_section=(list(mrope_section) if mrope_section is not None else None),
        inplace=False,
    )
    ref = tensors[IDX_K_CACHE]
    if isinstance(ref, torch.Tensor):
        return q_t.to(ref.dtype), k_t.to(ref.dtype), v_t.to(ref.dtype)
    return (
        _torch_to_np(q_t, ref.dtype),
        _torch_to_np(k_t, ref.dtype),
        _torch_to_np(v_t, ref.dtype),
    )


class AclnnUndGenQkvRmsNormRopeCacheTestSpec(_SpecBase):
    """aclnnUndGenQkvRmsNormRopeCache 的 aclnn 通路测试规范。

    形参与 aclnn 头文件一一对应共 18 个，全部按位置下发。判据同 kernel 侧。
    """

    def customize_inputs(
        und_qkv,
        und_weights_q,
        und_weights_k,
        cos_sin_cache,
        k_cache,
        v_cache,
        slot_mapping,
        positions,
        gen_qkv,
        gen_weights_q,
        gen_weights_k,
        cat_indices,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        norm_eps,
        mrope_section,
        q_out,
        **kwargs,
    ):
        """返回值仅为符合 customize_inputs 契约，aclnn 通路实际取的是原地改写的结果。"""
        return _rewrite_indices(
            [
                und_qkv,
                und_weights_q,
                und_weights_k,
                cos_sin_cache,
                k_cache,
                v_cache,
                slot_mapping,
                positions,
                gen_qkv,
                gen_weights_q,
                gen_weights_k,
                cat_indices,
            ],
            kwargs,
        )

    def golden(
        und_qkv,
        und_weights_q,
        und_weights_k,
        cos_sin_cache,
        k_cache,
        v_cache,
        slot_mapping,
        positions,
        gen_qkv,
        gen_weights_q,
        gen_weights_k,
        cat_indices,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        norm_eps,
        mrope_section,
        q_out,
        **kwargs,
    ):
        """返回 (q, k_cache, v_cache)，与 output_tensor_indexes=(12, 4, 5) 对齐。"""
        return _golden_from_tensors(
            [
                und_qkv,
                und_weights_q,
                und_weights_k,
                cos_sin_cache,
                k_cache,
                v_cache,
                slot_mapping,
                positions,
                gen_qkv,
                gen_weights_q,
                gen_weights_k,
                cat_indices,
            ],
            num_heads_q,
            num_heads_k,
            num_heads_v,
            norm_eps,
            mrope_section,
        )


class TorchUndGenQkvRmsNormRopeCacheTestSpec(_SpecBase):
    """E2E（torch）通路测试规范，入参是设备上的 torch.Tensor。

    签名照抄 torch schema：12 个张量按位置，5 个属性在 `*` 之后是 keyword-only。
    输出序 (q, k_cache, v_cache) 由 CSV 的 inplace_input_indexes=(4,5) 保证。
    """

    def customize_inputs(
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
        *,
        num_heads_q=8,
        num_heads_k=1,
        num_heads_v=1,
        norm_eps=1e-6,
        mrope_section=(),
        **kwargs,
    ):
        """返回值仅为符合契约，实际取的是原地改写的结果，同 aclnn 侧。"""
        return _rewrite_indices(
            [
                und_qkv,
                und_weights_q,
                und_weights_k,
                cos_sin_cache,
                k_cache,
                v_cache,
                slot_mapping,
                positions,
                gen_qkv,
                gen_weights_q,
                gen_weights_k,
                cat_indices,
            ],
            kwargs,
        )

    def golden(
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
        *,
        num_heads_q=8,
        num_heads_k=1,
        num_heads_v=1,
        norm_eps=1e-6,
        mrope_section=(),
        **kwargs,
    ):
        """返回 (q, k_cache, v_cache)。"""
        return _golden_from_tensors(
            [
                und_qkv,
                und_weights_q,
                und_weights_k,
                cos_sin_cache,
                k_cache,
                v_cache,
                slot_mapping,
                positions,
                gen_qkv,
                gen_weights_q,
                gen_weights_k,
                cat_indices,
            ],
            num_heads_q,
            num_heads_k,
            num_heads_v,
            norm_eps,
            mrope_section,
        )
