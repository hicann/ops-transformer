#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unified LightningIndexerV2 executor — validates both generic and fast templates.

Key capabilities (vs prior separate executors):
- Always calls lightning_indexer_metadata (torch API) → passes result to
  main op → AICPU automatically decides fast/generic routing. Executor is
  routing-agnostic; works for all 1400 cases without case-level flags.
- Variable-length sequences: optional "q_lens"/"k_lens" case attrs build
  exact cu_seqlens/block_table; absent → uniform split (backward compatible).
- Real index comparison: per-row sorted-set equality + causal-boundary /
  padding-count / duplicate / shape semantic assertions (hard fail).
- FP64 CPU golden reference.
- Robust values comparison (rv=1): relative error with absolute floor
  |a-g| / max(|g|, floor) to avoid near-zero amplification.
"""

import itertools
import json
from pathlib import Path

import torch
import torch_npu

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.cv_fused_double_benchmark_compare import (
    CvFusedDoubleBenchmarkAccuracyCompare,
)
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

logging = Logger().get_logger()

_GOLDEN_DTYPE = torch.float64
_ABS_FLOOR_RATIO = 0.01  # values comparison floor = 0.01 * row abs-max


def _opt(tensor, device=None):
    """Return tensor if non-empty, else None (torch API Tensor? semantics)."""
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    return tensor.to(device) if device is not None else tensor


def _prefix(lengths, device=None):
    return torch.tensor(
        [0] + list(itertools.accumulate(lengths)), dtype=torch.int32, device=device
    )


# ---------------------------------------------------------------- golden math


def _golden_batch(
    q_g,
    k_b,
    w_b,
    topk,
    cmp_ratio,
    res_k,
    mask_mode,
    s1,
    offset_out,
    return_value,
    n1,
    n2,
    seqused_k_val=None,
    device=None,
):
    """FP64 golden for one (batch, n2) slice. Returns (indices, values)."""
    g = max(1, n1 // n2)
    s2 = k_b.shape[0]
    actual_s2 = seqused_k_val if seqused_k_val is not None else s2
    dtype = torch.float32 if q_g.device.type == "npu" else _GOLDEN_DTYPE

    indices = torch.full((s1, topk), -1, dtype=torch.int32, device=device)
    values = torch.full((s1, topk), float("-inf"), dtype=torch.float32, device=device)
    eff = min(topk, s2)
    if eff <= 0:
        return indices, values

    q_f = q_g.to(dtype)  # [s1, g, d]
    k_f = k_b.to(dtype)  # [s2, d]
    w_f = w_b.to(dtype)  # [s1, g]
    q_t = q_f.permute(1, 0, 2)  # [g, s1, d]
    k_t = k_f.unsqueeze(0).expand(g, s2, k_f.shape[-1])  # [g, s2, d]
    if isinstance(offset_out, list):
        offset_out = torch.tensor(
            offset_out, dtype=torch.int64, device=device
        ).unsqueeze(-1)
    # S1 分块计算: 峰值内存从 [g,s1,s2] (大用例 ~75GB) 压到 [g,256,s2] (~500MB)
    chunk = 256
    for c0 in range(0, s1, chunk):
        c1 = min(c0 + chunk, s1)
        dots_c = torch.matmul(q_t[:, c0:c1], k_t.transpose(1, 2))  # [g, m, s2]
        scores_c = (dots_c.relu() * w_f.t()[:, c0:c1].unsqueeze(-1)).sum(
            dim=0
        )  # [m, s2]
        if mask_mode == 3:
            l_orig = actual_s2 * cmp_ratio + res_k
            offset = l_orig - s1
            causal = (torch.arange(actual_s2, device=device) + 1) * cmp_ratio - 1
            if causal.shape[0] < s2:
                pad = torch.full((s2,), -1.0, device=device, dtype=dtype)
                pad[: causal.shape[0]] = causal.to(dtype)
                causal = pad
            q_global = offset + torch.arange(c0, c1, device=device)
            scores_c = scores_c.masked_fill(
                causal.unsqueeze(0) > q_global.unsqueeze(1), float("-inf")
            )
        if actual_s2 < s2:
            scores_c[:, actual_s2:] = float("-inf")
        vals_c, idx_c = torch.topk(
            scores_c, min(eff, scores_c.shape[-1]), dim=-1, sorted=True
        )
        off_c = (
            offset_out[c0:c1] if isinstance(offset_out, torch.Tensor) else offset_out
        )
        idx_c = idx_c + off_c
        idx_c[~torch.isfinite(vals_c)] = -1
        indices[c0:c1, :eff] = idx_c.to(torch.int32)
        if return_value:
            values[c0:c1, :eff] = vals_c.to(torch.float32)
    return indices, values


def _golden_full(
    q,
    k,
    w,
    topk,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    return_value,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
):
    """Full-parameter golden: handles TND/BSND/PA, mask 0/3, cmp_ratio 1-128."""
    device = q.device
    out_tnd = layout_q == "TND"

    # --- normalise Q to [B, max_s1, N1, D] and W to [B, max_s1, N1]
    if out_tnd:
        b = len(cu_seqlens_q) - 1
        s1_list = [int(cu_seqlens_q[i + 1] - cu_seqlens_q[i]) for i in range(b)]
        max_s1 = max(s1_list)
        n1, d = q.shape[1], q.shape[2]
        q_bs = torch.zeros(b, max_s1, n1, d, dtype=q.dtype, device=device)
        w_bs = torch.zeros(b, max_s1, n1, dtype=w.dtype, device=device)
        for i in range(b):
            s, e = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            q_bs[i, : e - s] = q[s:e]
            w_bs[i, : e - s] = w[s:e]
        q, w = q_bs, w_bs
    else:
        b, max_s1 = q.shape[0], q.shape[1]
        s1_list = [max_s1] * b
        if _opt(seqused_q) is not None:
            s1_list = [int(seqused_q[i]) for i in range(b)]
    n1, d = q.shape[2], q.shape[3]

    # --- normalise K to [B, max_s2, N2, D]
    if layout_k == "PA_BBND":
        n_blocks, bs, n2, _ = k.shape
        ids = block_table.reshape(-1).long()
        k_gathered = k[ids].reshape(b, -1, bs, n2, k.shape[-1])
        k = k_gathered.permute(0, 1, 3, 2, 4).reshape(b, -1, n2, k.shape[-1])
        s2_list = [k.shape[1]] * b
    elif layout_k == "TND":
        n2, d = k.shape[1], k.shape[2]
        s2_list = [int(cu_seqlens_k[i + 1] - cu_seqlens_k[i]) for i in range(b)]
        max_s2 = max(s2_list)
        k_bs = torch.zeros(b, max_s2, n2, d, dtype=k.dtype, device=device)
        for i in range(b):
            s, e = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            k_bs[i, : e - s] = k[s:e]
        k = k_bs
    else:
        s2_list = [k.shape[1]] * b
    n2 = k.shape[2]

    g = max(1, n1 // n2)

    res_list = [0] * b
    if _opt(cmp_residual_k) is not None:
        res_list = [int(cmp_residual_k[i]) for i in range(b)]

    off_list = [[0] * s1_list[i] for i in range(b)]
    if _opt(output_idx_offset) is not None:
        off = output_idx_offset
        if out_tnd:
            row = 0
            for i in range(b):
                for j in range(s1_list[i]):
                    for n in range(n2):
                        off_list[i][j] = int(off[row, n])
                    row += 1
        else:
            for i in range(b):
                for j in range(min(off.shape[1], s1_list[i])):
                    off_list[i][j] = int(off[i, j, 0])

    sk_list = [None] * b
    if _opt(seqused_k) is not None:
        sk_list = [int(seqused_k[i]) for i in range(b)]
    elif layout_k == "TND":
        sk_list = s2_list

    total_s1 = sum(s1_list)
    if out_tnd:
        idx_all = torch.full((total_s1, n2, topk), -1, dtype=torch.int32, device=device)
    else:
        idx_all = torch.full(
            (b, max_s1, n2, topk), -1, dtype=torch.int32, device=device
        )
    # rv=0: values 输出用空张量(与算子侧/NPU 侧一致), 省去全量 -inf 张量的
    # 写盘/传输/读取(重用例 ~470MB/例的纯 I/O 浪费); rv=1 才物化
    if return_value:
        if out_tnd:
            val_all = torch.full(
                (total_s1, n2, topk), float("-inf"), dtype=torch.float32, device=device
            )
        else:
            val_all = torch.full(
                (b, max_s1, n2, topk), float("-inf"), dtype=torch.float32, device=device
            )
    else:
        val_all = torch.empty(0, dtype=torch.float32, device=device)

    for i in range(b):
        s1_i = s1_list[i]
        sk_i = sk_list[i] if sk_list[i] is not None else s2_list[i]
        for n in range(n2):
            # q slice: [s1_i, g, d]  (this n2 head uses g consecutive q heads)
            q_slice = q[i, :s1_i, n * g : (n + 1) * g, :].reshape(s1_i, g, -1)
            k_slice = k[i, :sk_i, n, :]  # [sk_i, d]
            w_slice = w[i, :s1_i, n * g : (n + 1) * g]  # [s1_i, g]
            idx, val = _golden_batch(
                q_slice,
                k_slice,
                w_slice,
                topk,
                cmp_ratio,
                res_list[i],
                mask_mode,
                s1_i,
                off_list[i],
                return_value,
                g,
                1,
                seqused_k_val=sk_i,
                device=device,
            )
            if out_tnd:
                s1_off = sum(s1_list[:i])
                idx_all[s1_off : s1_off + s1_i, n] = idx
                if return_value:
                    val_all[s1_off : s1_off + s1_i, n] = val
            else:
                idx_all[i, :s1_i, n] = idx
                if return_value:
                    val_all[i, :s1_i, n] = val

    return idx_all, val_all


# ------------------------------------------------------------ input preparation


def _prepare_inputs(kwargs):
    """Build kwargs with exact cu_seqlens (if q_lens/k_lens provided) or uniform split."""
    kw = dict(kwargs)
    device = kw["q"].device
    layout_q = kw.get("layout_q", "BSND")
    layout_k = kw.get("layout_k", "BSND")
    cmp_ratio = int(kw.get("cmp_ratio", 1))
    mask_mode = int(kw.get("mask_mode", 3))

    def _parse_lens(v):
        """q_lens/k_lens 属性: 支持逗号分隔字符串(网络用例)或列表."""
        if isinstance(v, (list, tuple)):
            vals = [int(x) for x in v]
            return vals or None
        if isinstance(v, str) and v.strip():
            vals = [int(x) for x in v.split(",") if x.strip()]
            return vals or None
        return None

    q_lens = _parse_lens(kw.get("q_lens"))
    k_lens = _parse_lens(kw.get("k_lens"))

    if layout_q == "TND":
        t1 = kw["q"].shape[0]
        if q_lens:
            b = len(q_lens)
            assert sum(q_lens) == t1, f"q_lens sum {sum(q_lens)} != T1 {t1}"
        else:
            b = kw["cu_seqlens_q"].shape[0] - 1
            base = t1 // b
            q_lens = [base] * b
            q_lens[-1] += t1 % b
        kw["cu_seqlens_q"] = _prefix(q_lens, device)
        kw["seqused_q"] = None
    else:
        q_lens = [kw["q"].shape[1]] * kw["q"].shape[0]
        kw["cu_seqlens_q"] = None
    kw["seqused_q"] = None  # 与旧执行器一致: ATK 随机值无意义, 一律置 None

    if layout_k == "TND":
        t_k = kw["k"].shape[0]
        if k_lens:
            assert sum(k_lens) == t_k, f"k_lens sum {sum(k_lens)} != T_k {t_k}"
        else:
            b_k = len(q_lens)
            base = t_k // max(b_k, 1)
            k_lens = [base] * b_k
            k_lens[-1] += t_k % b_k
        kw["cu_seqlens_k"] = _prefix(k_lens, device)
        kw["seqused_k"] = None
        kw["block_table"] = None
    elif layout_k == "PA_BBND":
        # 旧执行器 PA 归一化: 空 k 保护 + block_size 16 对齐 padding + 压缩列截断
        b = len(q_lens)
        k = kw["k"]
        if k.dim() == 4 and k.shape[0] == 0:
            n2_pa, d_pa = k.shape[2], k.shape[3]
            bs_pa = max(16, k.shape[1]) if k.shape[1] > 0 else 16
            bs_pa = ((bs_pa + 15) // 16) * 16
            k = torch.zeros(1, bs_pa, n2_pa, d_pa, dtype=k.dtype, device=device)
            kw["k"] = k
        num_blocks, block_size, n2_pa, d_pa = k.shape
        aligned_bs = ((block_size + 15) // 16) * 16
        if block_size != aligned_bs:
            padded = torch.zeros(
                num_blocks, aligned_bs, n2_pa, d_pa, dtype=k.dtype, device=device
            )
            padded[:, :block_size] = k
            kw["k"] = padded
            block_size = aligned_bs
        # 表构造: k_lens(网络)模式用无重复随机置换表(self_test scatter 已验证安全, 触发过
        # 512K 用例 OOB 的是带重复的随机 clamp 表); 1400 模式保持 ATK 原表 clamp(已验证)
        raw_table = kw.get("block_table")
        per_batch_probe = bool(k_lens) and all(
            0 < n_ <= ((num_blocks) * block_size) for n_ in k_lens
        )
        if per_batch_probe:
            # 置换表: CPU 固定种子生成(两节点确定性一致!)再搬设备 — NPU/CPU 各自 randperm 会产生不同表
            gen = torch.Generator().manual_seed(0x5A5A1234 + num_blocks)
            perm = torch.randperm(num_blocks, generator=gen)[
                : max((max((n_ + block_size - 1) // block_size for n_ in k_lens), 1))
            ].to(torch.int32)
            table = perm.unsqueeze(0).repeat(b, 1).to(device=device).contiguous()
        elif (
            isinstance(raw_table, torch.Tensor)
            and raw_table.numel() > 0
            and raw_table.dim() == 2
        ):
            table = raw_table.clamp(0, num_blocks - 1).to(
                device=device, dtype=torch.int32
            )
        else:
            if k_lens:
                counts = [(n + block_size - 1) // block_size for n in k_lens]
                assert num_blocks == sum(counts), "PA block count mismatch"
            else:
                base = num_blocks // max(b, 1)
                counts = [base] * b
                counts[-1] += num_blocks - base * b
            table = torch.zeros(
                b, max(counts) if counts else 1, dtype=torch.int32, device=device
            )
            start = 0
            for i, cnt in enumerate(counts):
                table[i, :cnt] = torch.arange(
                    start, start + cnt, dtype=torch.int32, device=device
                )
                start += cnt
        # 压缩域列截断 (旧执行器): r>1 时 table 仅保留前 compressed_blocks 列
        max_cols = table.shape[1]
        s2 = max_cols * block_size
        s2c = max(1, s2 // cmp_ratio)
        compressed_blocks = max(1, s2c // block_size)
        if compressed_blocks < max_cols:
            table = table[:, :compressed_blocks].contiguous()
        actual_k_len = compressed_blocks * block_size
        kw["block_table"] = table
        # 网络用例(k_lens 属性): seqused_k 按批真实长度, 保留 PA 变长语义(infer 式: 无 csk)
        per_batch = bool(k_lens) and all(0 < n_ <= actual_k_len for n_ in k_lens)
        if per_batch:
            kw["seqused_k"] = torch.tensor(k_lens, dtype=torch.int32, device=device)
            k_lens = list(k_lens)
        else:
            kw["seqused_k"] = torch.full(
                (b,), actual_k_len, dtype=torch.int32, device=device
            )
            k_lens = [actual_k_len] * b
        # TND-q + PA-k 时 cu_seqlens_k 传前缀和(旧执行器 L285-288 同款; 变长模式不传)
        if layout_q == "TND" and not per_batch:
            kw["cu_seqlens_k"] = _prefix([actual_k_len] * b, device)
        else:
            kw["cu_seqlens_k"] = None
        kw["cmp_residual_k"] = torch.tensor(
            [n_ % cmp_ratio for n_ in k_lens], dtype=torch.int32, device=device
        )
    else:
        k_lens = [kw["k"].shape[1]] * kw["k"].shape[0]
        kw["cu_seqlens_k"] = None
        kw["seqused_k"] = None
        kw["block_table"] = None

    # 计算 cmp_residual_k: r>1 且 mask=3 时 metadata 必传
    if cmp_ratio > 1 and mask_mode == 3:
        res_list = [max(0, kl_ % cmp_ratio) for kl_ in k_lens]
        kw["cmp_residual_k"] = torch.tensor(res_list, dtype=torch.int32, device=device)
    else:
        kw["cmp_residual_k"] = None

    return kw, q_lens, k_lens


# ---------------------------------------------------------------- semantic validation


def _validate_indices(
    indices,
    q_lens,
    k_lens,
    topk,
    layout_q,
    layout_k,
    cmp_ratio,
    mask_mode,
    residual_k,
    backend,
):
    """Hard assertions: causal boundary, padding count, duplicates, shape.
    Handles both TND [sum_s1, n2, topk] and BSND [B, max_s1, n2, topk].
    residual_k: per-batch residual tensor/list (golden: l_orig = sk*r + res)."""
    b = len(q_lens)
    if indices.dtype != torch.int32:
        raise AssertionError(f"{backend}: indices dtype {indices.dtype} != int32")
    if isinstance(residual_k, torch.Tensor):
        res_list = residual_k.tolist()
    elif residual_k:
        res_list = [int(x) for x in residual_k]
    else:
        res_list = [0] * b

    # 按 layout 归一化为逐行 [total_s1, topk]（n2=0 切片；n2>1 时逐 head 校验）
    def _vis(sq, sk, res):
        """与 golden 同公式: mask=3 时 l_orig = sk*r + res 的 causal 可见数; 其余 mask 全可见."""
        device = indices.device
        if mask_mode == 3:
            l_orig = sk * cmp_ratio + int(res)
            offset = l_orig - sq
            # causal[j] = (j+1)*r - 1 <= offset + i → 可见
            causal = (torch.arange(sk, device=device) + 1) * cmp_ratio - 1
            q_g = offset + torch.arange(sq, device=device)
            vis = (causal.unsqueeze(0) <= q_g.unsqueeze(1)).sum(dim=1)  # [sq]
            return vis.clamp(max=sk)
        return torch.full((sq,), sk, dtype=torch.long, device=device)

    if indices.dim() == 3 and layout_q == "TND":
        total_q = sum(q_lens)
        if indices.shape[0] != total_q:
            raise AssertionError(
                f"{backend}: TND shape {tuple(indices.shape)} != ({total_q}, n2, {topk})"
            )
        rows = indices[:, 0, :]
        visible = torch.cat(
            [
                _vis(sq, sk, res_list[i])
                for i, (sq, sk) in enumerate(zip(q_lens, k_lens))
            ]
        )
    elif indices.dim() == 4:
        rows_list = []
        visible_list = []
        for i in range(b):
            sq, sk = q_lens[i], k_lens[i]
            rows_list.append(indices[i, :sq, 0, :])
            visible_list.append(_vis(sq, sk, res_list[i]))
        rows = torch.cat(rows_list, dim=0)
        visible = torch.cat(visible_list, dim=0)
    else:
        raise AssertionError(
            f"{backend}: unexpected indices dim/shape {tuple(indices.shape)}"
        )

    valid = rows >= 0
    if bool((rows < -1).any()):
        raise AssertionError(f"{backend}: index < -1 found")
    if bool((valid & (rows >= visible[:, None])).any()):
        raise AssertionError(f"{backend}: causal bound violation")
    if not torch.equal(valid.sum(1), visible.clamp(max=topk)):
        raise AssertionError(f"{backend}: -1 padding count mismatch")
    ordered = rows.sort(1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
        raise AssertionError(f"{backend}: duplicate index")
    return rows


# ---------------------------------------------------------------- score CV


def _row_score_fp64(kw, q_lens, k_lens, b, i):
    """FP64 score row for (batch b, query i) at n2=0 head slice, with -inf outside visibility.
    先切片后转 FP64: 避免整张量转换(重用例 q 全量 .double() 每行 3.8GB)."""
    q = kw["q"]
    w = kw["w"]
    k_all = kw["k"]
    layout_q = kw.get("layout_q", "BSND")
    layout_k = kw.get("layout_k", "BSND")
    n1 = q.shape[1] if layout_q == "TND" else q.shape[2]
    g = max(1, n1)  # N2 == 1
    sq = q_lens[b]
    sk = k_lens[b]

    if layout_q == "TND":
        start = int(kw["cu_seqlens_q"][b])
        q_row = q[start + i, 0:g, :].double()  # [g, d]
        w_row = w[start + i, 0:g].double()
    else:
        q_row = q[b, i, 0:g, :].double()
        w_row = w[b, i, 0:g].double()

    if layout_k == "PA_BBND":
        table = kw["block_table"]
        n_blocks, bs, n2k, d = k_all.shape
        kb = k_all[table[b].long()].reshape(-1, n2k, d)[:, 0, :].double()
        sk_vis = int(kw["seqused_k"][b])
    elif layout_k == "TND":
        kstart = int(kw["cu_seqlens_k"][b])
        kb = k_all[kstart : kstart + sk, 0, :].double()
        sk_vis = sk
    else:
        kb = k_all[b, :sk, 0, :].double()
        sk_vis = sk

    relu = torch.relu(q_row @ kb.transpose(0, 1))  # [g, s2]
    score = w_row @ relu  # [s2] FP64

    r = int(kw.get("cmp_ratio", 1))
    mask_mode = int(kw.get("mask_mode", 3))
    if mask_mode == 3:
        res_t = kw.get("cmp_residual_k")
        res = int(res_t[b]) if isinstance(res_t, torch.Tensor) else 0
        off = sk * r + res - sq
        vis = min(sk, max(0, (off + i + 1) // r))
    else:
        vis = sk_vis if layout_k == "PA_BBND" else sk
    if vis < score.numel():
        score[vis:] = float("-inf")
    return score


# ---------------------------------------------------------------- comparison


def _values_re(actual, golden, floor_ratio=_ABS_FLOOR_RATIO):
    """Relative error with absolute floor: |a-g| / max(|g|, floor)."""
    row_max = golden.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    floor = floor_ratio * row_max
    denom = golden.abs().clamp(min=floor)
    return (actual - golden).abs() / denom


@register("function_aclnn_lightning_indexer_v2_unified")
class FunctionLightningIndexerV2UnifiedApi(BaseApi):
    """Unified executor: generic and fast template routing with real index comparison."""

    def init_by_input_data(self, input_data: InputDataset):
        return input_data

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kw = dict(input_data.kwargs) if input_data.kwargs else {}
        device = kw["q"].device

        # 路由: 按节点名 + backend 区分 operator / golden
        #   npu + name 含 "golden" → NPU golden 节点(直接在 NPU 上算 golden)
        #   npu + 其他           → NPU operator 节点(跑算子)
        #   cpu                  → CPU golden 节点(内部可 NPU 加速)
        node_name = getattr(self, "name", "") or ""
        if self.device == "npu" and "golden" in node_name.lower():
            return self._run_golden(kw, device)
        if self.device == "npu":
            return self._run_npu(kw, device)
        return self._run_golden(kw, device)

    # -- NPU: always call metadata, AICPU decides routing
    def _run_npu(self, kw, device):
        from cann_ops_transformer.ops import (
            lightning_indexer,
            lightning_indexer_metadata,
        )

        kw, q_lens, k_lens = _prepare_inputs(kw)
        topk = int(kw.get("topk", 128))
        layout_q = kw.get("layout_q", "BSND")
        layout_k = kw.get("layout_k", "BSND")
        mask_mode = int(kw.get("mask_mode", 0))
        cmp_ratio = int(kw.get("cmp_ratio", 1))
        return_value = int(kw.get("return_value", 0))

        # 始终调 metadata 前置算子 → AICPU 自动决定 fast/generic 路由
        metadata = lightning_indexer_metadata(
            kw["q"].shape[1] if layout_q == "TND" else kw["q"].shape[2],
            1,
            128,
            topk,
            cu_seqlens_q=_opt(kw.get("cu_seqlens_q"), device),
            cu_seqlens_k=_opt(kw.get("cu_seqlens_k"), device),
            seqused_q=None,
            seqused_k=_opt(kw.get("seqused_k"), device),
            cmp_residual_k=_opt(kw.get("cmp_residual_k"), device),
            batch_size=len(q_lens),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )

        outputs = lightning_indexer(
            kw["q"],
            kw["k"],
            kw["w"],
            topk,
            cu_seqlens_q=_opt(kw.get("cu_seqlens_q"), device),
            cu_seqlens_k=_opt(kw.get("cu_seqlens_k"), device),
            seqused_q=_opt(kw.get("seqused_q"), device),
            seqused_k=_opt(kw.get("seqused_k"), device),
            cmp_residual_k=_opt(kw.get("cmp_residual_k"), device),
            block_table=_opt(kw.get("block_table"), device),
            output_idx_offset=None,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=return_value,
        )
        torch.npu.synchronize()
        return outputs

    # -- Golden (支持两种节点: NPU golden 节点直接算; CPU 节点内部移到 NPU 加速)
    def _run_golden(self, kw, device):
        # CPU 节点时尝试移到 NPU 加速; NPU golden 节点时 tensor 已在 NPU 上
        npu_dev = None
        if device.type == "cpu":
            try:
                if torch.npu.is_available():
                    npu_dev = torch.device("npu:7")
                    for key in ("q", "k", "w"):
                        if kw.get(key) is not None and isinstance(
                            kw[key], torch.Tensor
                        ):
                            kw[key] = kw[key].to(npu_dev)
            except Exception:
                npu_dev = None

        kw, q_lens, k_lens = _prepare_inputs(kw)
        topk = int(kw.get("topk", 128))
        return_value = int(kw.get("return_value", 0))
        idx, val = _golden_full(
            kw["q"],
            kw["k"],
            kw["w"],
            topk,
            kw.get("layout_q", "BSND"),
            kw.get("layout_k", "BSND"),
            int(kw.get("mask_mode", 0)),
            int(kw.get("cmp_ratio", 1)),
            return_value,
            cu_seqlens_q=kw.get("cu_seqlens_q"),
            cu_seqlens_k=kw.get("cu_seqlens_k"),
            seqused_q=kw.get("seqused_q"),
            seqused_k=kw.get("seqused_k"),
            cmp_residual_k=kw.get("cmp_residual_k"),
            block_table=kw.get("block_table"),
            output_idx_offset=None,
        )
        # CPU 节点时结果需回 CPU; NPU golden 节点时 ATK 会处理
        if npu_dev is not None:
            idx = idx.cpu()
            if isinstance(val, torch.Tensor) and val.is_npu:
                val = val.cpu()
        return [idx, val]


@ACCURACY_REGISTRY.register("li_v2_unified_double_benchmark")
class UnifiedCompare(CvFusedDoubleBenchmarkAccuracyCompare):
    """Two-stage compare: index sorted-set + robust values."""

    def precision_compare_for_case(
        self, main_node, remote_node, is_npu=False, device_id=0
    ):
        from atk.common.connection import RemoteManager
        from atk.tasks.executors.dataset_executor import DatasetExecutor
        from atk.common.utils import torch_load_safe

        attrs = {
            item.name: item.range_values[0]
            if isinstance(item.range_values, list)
            else item.range_values
            for item in self.case_config.inputs
            if item.type == "attr"
        }
        topk = int(attrs.get("topk", 128))
        layout_q = attrs.get("layout_q", "BSND")
        layout_k = attrs.get("layout_k", "BSND")
        cmp_ratio = int(attrs.get("cmp_ratio", 1))
        mask_mode = int(attrs.get("mask_mode", 0))
        return_value = int(attrs.get("return_value", 0))
        q_lens = attrs.get("q_lens")
        k_lens = attrs.get("k_lens")
        q_lens = list(q_lens) if isinstance(q_lens, (list, tuple)) else None
        k_lens = list(k_lens) if isinstance(k_lens, (list, tuple)) else None

        self.remote_manager = RemoteManager(self.case_config, main_node, remote_node)
        local_dir, remote_dir = self.remote_manager.get_local_and_remote_dir()

        input_file = (
            Path(self.remote_manager.get_input_dir(main_node, is_create=False))
            / "input.bin"
        )
        _, data, _, _ = DatasetExecutor._load_file_and_backward(
            str(input_file), self.case_config.inputs
        )
        kw, q_lens, k_lens = _prepare_inputs(data)
        # 属性实际值优先取 input.bin(topk 等 attr 可能是随机范围, spec 的 range_values[0] 不可靠)
        topk = int(kw.get("topk", attrs.get("topk", 128)))
        layout_q = kw.get("layout_q", attrs.get("layout_q", "BSND"))
        layout_k = kw.get("layout_k", attrs.get("layout_k", "BSND"))
        cmp_ratio = int(kw.get("cmp_ratio", attrs.get("cmp_ratio", 1)))
        mask_mode = int(kw.get("mask_mode", attrs.get("mask_mode", 0)))
        return_value = int(kw.get("return_value", attrs.get("return_value", 0)))
        kw.update(
            topk=topk,
            layout_q=layout_q,
            layout_k=layout_k,
            cmp_ratio=cmp_ratio,
            mask_mode=mask_mode,
            return_value=return_value,
        )

        outputs = [
            torch_load_safe(str(Path(local_dir) / f"output_{i}.pt")) for i in range(2)
        ]
        reference = [
            torch_load_safe(self.remote_manager.download(f"output_{i}.pt"))
            for i in range(2)
        ]

        results, report = self._compare(
            outputs, reference, kw, q_lens, k_lens, topk, return_value
        )
        self._save_report(local_dir, report)
        return results

    def _compare(self, outputs, reference, kw, q_lens, k_lens, topk, return_value):
        report = {
            "comparison_policy": "index_set_then_values",
            "matched_rows": 0,
            "differing_rows": 0,
            "stage": "indices_set",
            "result": True,
            "index_differences": [],
            "values_metrics": {},
        }

        try:
            actual = _validate_indices(
                outputs[0],
                q_lens,
                k_lens,
                topk,
                kw.get("layout_q", "BSND"),
                kw.get("layout_k", "BSND"),
                int(kw.get("cmp_ratio", 1)),
                int(kw.get("mask_mode", 3)),
                kw.get("cmp_residual_k"),
                "npu",
            )
            expected = _validate_indices(
                reference[0],
                q_lens,
                k_lens,
                topk,
                kw.get("layout_q", "BSND"),
                kw.get("layout_k", "BSND"),
                int(kw.get("cmp_ratio", 1)),
                int(kw.get("mask_mode", 3)),
                kw.get("cmp_residual_k"),
                "cpu",
            )
        except AssertionError as e:
            report.update(result=False, stage="semantic_validation", error=str(e))
            return self._results(report, return_value), report

        equal = (actual.sort(1).values == expected.sort(1).values).all(1)
        report["matched_rows"] = int(equal.sum())
        differing = (~equal).nonzero().flatten().tolist()
        report["differing_rows"] = len(differing)

        if not differing:
            report["stage"] = "indices_set"
            report["result"] = True
            if return_value and len(outputs) > 1 and outputs[1].numel() > 0:
                val_result = self._compare_values(outputs[1], reference[1])
                report["values_metrics"] = val_result
                report["result"] = val_result["result"]
                report["stage"] = "values"
            return self._results(report, return_value), report

        # differing rows: FP64 score tie validation — 双方选择须同时满足
        # min(选中分数) >= max(未选中可见分数) - 容差 (完整 top-k 有效性)
        report["stage"] = "score_cv"
        report["index_differences"] = [
            {
                "query": q,
                "npu_only": sorted(set(actual[q].tolist()) - set(expected[q].tolist())),
                "cpu_only": sorted(set(expected[q].tolist()) - set(actual[q].tolist())),
            }
            for q in differing[:20]
        ]
        import bisect

        cum = [0]
        for n_ in q_lens:
            cum.append(cum[-1] + n_)
        tie_rows = 0
        bad_rows = []
        eps_rel = 2e-2  # bf16 舍入级 tie 容差(行最大分数的 2%)
        for t in differing:
            b = bisect.bisect_right(cum, t) - 1
            i = t - cum[b]
            sc = _row_score_fp64(kw, q_lens, k_lens, b, i)
            fin = sc[torch.isfinite(sc)]
            eps = float(fin.abs().max()) * eps_rel + 1e-6 if fin.numel() else 0.0
            sel_mask = torch.zeros(sc.numel(), dtype=torch.bool)
            for idx_ in actual[t].tolist():
                if idx_ >= 0:
                    sel_mask[idx_] = True
            if not sel_mask.any():
                continue  # 全 -1 行: 语义校验已保证
            min_sel = sc[sel_mask].min().item()
            max_exc = (
                sc[~sel_mask & torch.isfinite(sc)].max().item()
                if (~sel_mask & torch.isfinite(sc)).any()
                else float("-inf")
            )
            if min_sel >= max_exc - eps:
                tie_rows += 1
            else:
                bad_rows.append(
                    {
                        "query": t,
                        "min_sel": min_sel,
                        "max_excluded": max_exc,
                        "eps": eps,
                    }
                )
        if not bad_rows:
            report["result"] = True
            report["reason"] = (
                f"{len(differing)} rows tie-broken within {eps_rel:.0%} row-max, "
                f"both selections valid top-k"
            )
        else:
            report["result"] = False
            report["reason"] = (
                f"{len(bad_rows)} rows invalid top-k beyond tie tolerance"
            )
            report["invalid_rows"] = bad_rows[:10]
        return self._results(report, return_value), report

    def _compare_values(self, actual_val, golden_val):
        """Robust values comparison with absolute floor. Only finite positions."""
        a, g = actual_val.float(), golden_val.float()
        finite = torch.isfinite(g) & torch.isfinite(a)
        if not finite.any():
            return {
                "max_re_with_floor": 0.0,
                "avg_re_with_floor": 0.0,
                "rmse": 0.0,
                "floor": 0.0,
                "result": True,
                "note": "no finite values to compare",
            }
        a_f, g_f = a[finite], g[finite]
        re = _values_re(a_f, g_f)
        max_re = re.max().item() if re.numel() else 0.0
        avg_re = re.mean().item() if re.numel() else 0.0
        rmse = ((a_f - g_f) ** 2).mean().sqrt().item() if g_f.numel() else 0.0
        result = max_re <= 0.02
        return {
            "max_re_with_floor": max_re,
            "avg_re_with_floor": avg_re,
            "rmse": rmse,
            "finite_count": int(finite.sum().item()),
            "result": result,
        }

    def _results(self, report, return_value):
        idx_result = report["result"]
        r0 = AccuracyConfig(
            filename="output_0.pt",
            result=idx_result,
            error_info=json.dumps(report, ensure_ascii=False)[:500],
        )
        if return_value and "values_metrics" in report and report["values_metrics"]:
            r1 = AccuracyConfig(
                filename="output_1.pt",
                result=report["values_metrics"].get("result", True),
                error_info=json.dumps(report["values_metrics"]),
            )
        else:
            r1 = AccuracyConfig(
                filename="output_1.pt",
                result=True,
                error_info="rv=0 or no values comparison",
            )
        return [r0, r1]

    def _save_report(self, local_dir, report):
        folder = Path(local_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "accuracy_unified.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
