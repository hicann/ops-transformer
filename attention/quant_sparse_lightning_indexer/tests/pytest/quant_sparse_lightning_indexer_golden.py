#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU reference for QSLI INT8 and MXFP4, shared by pytest and TTK."""

import math
from collections import namedtuple

import numpy as np
import torch

NEG_INF = float("-inf")


def topk_golden(score_row, topk):
    """
    参与池 = 可达位置 (score != -inf); 候选外位置在调用方已置 -inf, 与不可达位置同样排除。
    有效数 < topk 时输出 -1 补齐 (对齐模型 masked_fill(~candidates, -inf) + where(..., -1))。
    排序键 (-score, index) 稳定排序: 按真实分降序, tie 取小索引。
    """
    n = score_row.shape[-1]
    indices = torch.arange(n)
    valid = score_row != NEG_INF
    v_idx = indices[valid]
    if v_idx.numel() == 0:
        return torch.full((topk,), -1, dtype=torch.int32)
    stacked = torch.stack([-score_row[v_idx], v_idx.to(torch.float32)], dim=1)
    _, sorted_idx = torch.sort(stacked, dim=0, stable=True)
    order = sorted_idx[:, 0]  # 有效位置降序 (tie 取小索引)
    out = torch.full((topk,), -1, dtype=torch.int32)
    actual = min(order.numel(), topk)
    out[:actual] = v_idx[order[:actual]].to(torch.int32)
    return out


FP4_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def to_cpu_tensor(value):
    """Preserve the encoded FP4/E8M0 bytes supplied by Torch or TTK."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    array = np.array(value, copy=True, order="C")
    name = str(array.dtype)
    if name == "float4_e2m1":
        if array.ndim == 0 or array.shape[-1] % 2:
            raise ValueError("QSLI FP4 input requires an even last dimension")
        codes = array.view(np.uint8) & 15
        return torch.from_numpy(codes[..., ::2] | (codes[..., 1::2] << 4))
    if name in ("float8_e8m0", "float8_e8m0fnu"):
        return torch.from_numpy(array.view(np.uint8))
    if name == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def copy_tensor(dst, src, name):
    if dst is None:
        return
    src = to_cpu_tensor(src)
    if torch.is_tensor(dst):
        if tuple(dst.shape) != tuple(src.shape):
            raise ValueError(f"{name} shape mismatch: {dst.shape}, {src.shape}")
        if str(dst.dtype) in ("torch.float4_e2m1fn_x2", "torch.float8_e8m0fnu"):
            dst.view(torch.uint8).copy_(src.view(torch.uint8).to(dst.device))
        else:
            dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
        return
    dst = np.asarray(dst)
    if str(dst.dtype) == "float4_e2m1":
        packed = src.view(torch.uint8).numpy()
        if dst.shape[:-1] != packed.shape[:-1] or dst.shape[-1] != 2 * packed.shape[-1]:
            raise ValueError(f"{name} FP4 storage shape mismatch")
        dst.view(np.uint8)[..., ::2] = packed & 15
        dst.view(np.uint8)[..., 1::2] = packed >> 4
    elif str(dst.dtype) in ("float8_e8m0", "float8_e8m0fnu"):
        np.copyto(dst.view(np.uint8), src.view(torch.uint8).numpy())
    elif str(dst.dtype) == "bfloat16":
        np.copyto(
            dst.view(np.uint16), src.to(torch.bfloat16).view(torch.uint16).numpy()
        )
    else:
        np.copyto(dst, src.numpy().astype(dst.dtype, copy=False))


def list_value(attributes, name):
    value = attributes.get(name)
    if value is None:
        value = attributes.get(f"{name}_values")
    if value is None:
        return None
    return [int(item) for item in to_cpu_tensor(value).reshape(-1).tolist()]


def decode_fp4(value):
    raw = to_cpu_tensor(value).view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    return FP4_VALUES[codes.long()]


def decode_scale(value):
    raw = to_cpu_tensor(value).view(torch.uint8)
    decoded = torch.pow(2.0, raw.float() - 127)
    return (
        torch.where(raw == 255, float("nan"), decoded)
        .flatten(-2)
        .repeat_interleave(32, dim=-1)
    )


def round_fp64_to_bf16(value):
    """Match QLI v2's single RNE rounding, including BF16 subnormals."""
    mantissa, exponent = torch.frexp(value)
    normal = torch.ldexp(torch.round(mantissa * 256), exponent - 8)
    subnormal = torch.round(value / (2.0**-133)) * (2.0**-133)
    rounded = torch.where(value.abs() < 2.0**-126, subnormal, normal)
    return torch.where(torch.isfinite(value), rounded, value).float().bfloat16()


def round_fp32_to_bf16_cast_round(value):
    value = value.to(torch.float32).contiguous()
    # Match CAST_ROUND: BF16 midpoint ties round away from zero.
    rounded_bits = (value.view(torch.int32) + 0x8000) & -0x10000
    rounded = rounded_bits.view(torch.float32)
    return torch.where(torch.isfinite(value), rounded, value).to(torch.bfloat16)


def mxfp4_scores(query, key, weights, query_scale, key_scale):
    query = decode_fp4(query) * decode_scale(query_scale)
    key = decode_fp4(key) * decode_scale(key_scale)
    qk = (
        torch.bmm(
            query.permute(1, 0, 2),
            key[:, 0].t().unsqueeze(0).expand(query.shape[1], -1, -1),
        )
        .clamp_min(0)
        .bfloat16()
        .permute(1, 0, 2)
    )
    weight = weights.bfloat16()
    score = torch.zeros((query.shape[0], key.shape[0]), dtype=torch.bfloat16)
    for head in range(query.shape[1]):
        score = round_fp64_to_bf16(
            score.double() + weight[:, head : head + 1].double() * qk[:, head].double()
        )
    return score.float()


def unpack_paged(key, key_scale, quant_mode):
    if quant_mode == 5:
        pages, groups, width = key.shape
        if width != 544:
            raise ValueError("QSLI packed PA key requires width 544")
        key_scale = key[..., 512:].contiguous().reshape(pages, groups * 8, 1, 2, 2)
        key = key[..., :512].contiguous().reshape(pages, groups * 8, 1, 64)
    return key, key_scale


SequenceGeometry = namedtuple(
    "SequenceGeometry", "batch q_prefix q_capacity q_used k_prefix k_used"
)
CompareParams = namedtuple(
    "CompareParams",
    "batch_size q_seq k_seq q_t_size k_t_size q_head_num k_head_num head_dim "
    "block_size block_num qk_dtype weight_dtype dequant_dtype actual_seq_dtype "
    "cu_seqlens_q cu_seqlens_k seqused_q seqused_k cmp_residual_k max_seqlen_q "
    "quant_mode layout_query layout_key sparse_count sparse_mode query_datarange "
    "key_datarange weights_datarange q_scale_datarange k_scale_datarange "
    "cmp_ratio return_value output_idx_offset_range",
)


def geometry(query, key, attributes):
    layout_q = attributes.get("layout_q", "BSND")
    layout_k = attributes.get("layout_k", "BSND")
    if layout_q == "BSND":
        batch, q_size = query.shape[:2]
        q_prefix = [b * q_size for b in range(batch + 1)]
    else:
        q_prefix = list_value(attributes, "cu_seqlens_q")
        if q_prefix is None:
            raise ValueError("QSLI TND query requires cu_seqlens_q")
        batch = len(q_prefix) - 1
    q_capacity = [right - left for left, right in zip(q_prefix, q_prefix[1:])]
    q_used = list_value(attributes, "seqused_q")
    if q_used is None:
        q_used = q_capacity
    k_prefix = list_value(attributes, "cu_seqlens_k")
    k_used = list_value(attributes, "seqused_k")
    if k_used is None:
        if layout_k == "BSND":
            k_used = [key.shape[1]] * batch
        elif layout_k == "TND" and k_prefix is not None:
            k_used = [right - left for left, right in zip(k_prefix, k_prefix[1:])]
        else:
            raise ValueError("QSLI PA requires seqused_k; TND requires cu_seqlens_k")
    if layout_k == "TND" and k_prefix is None:
        raise ValueError("QSLI TND key requires cu_seqlens_k")
    return SequenceGeometry(batch, q_prefix, q_capacity, q_used, k_prefix, k_used)


def compare_params(query, key, weights, query_scale, topk, quant_mode, attributes):
    batch, q_prefix, q_capacity, q_used, k_prefix, k_used = geometry(
        query, key, attributes
    )
    layout_q = attributes.get("layout_q", "BSND")
    layout_k = attributes.get("layout_k", "BSND")
    ranges = attributes.get("qli_input_ranges") or attributes.get("input_ranges") or ()
    ranges = tuple(repr(list(value)) if value is not None else None for value in ranges)
    ranges = (ranges + (None,) * 5)[:5]
    return CompareParams(
        batch,
        max(q_capacity, default=0),
        max(k_used, default=0),
        query.shape[0] if layout_q == "TND" else 0,
        key.shape[0] if layout_k == "TND" else 0,
        query.shape[-2],
        1,
        query.shape[-1] * (2 if quant_mode == 5 else 1),
        key.shape[1] if layout_k == "PA_BBND" else 0,
        key.shape[0] if layout_k == "PA_BBND" else 0,
        "FLOAT4_E2M1FN_X2" if quant_mode == 5 else "INT8",
        "FP32" if weights.dtype == torch.float32 else "FP16",
        "FLOAT8_E8M0FNU" if quant_mode == 5 else "FP16",
        "INT32",
        q_prefix if layout_q == "TND" else None,
        k_prefix,
        q_used,
        k_used,
        list_value(attributes, "cmp_residual_k"),
        attributes.get("max_seqlen_q", -1),
        quant_mode,
        layout_q,
        layout_k,
        topk,
        attributes.get("mask_mode", 0),
        *ranges,
        attributes.get("cmp_ratio", 1),
        bool(attributes.get("return_value", 0)),
        attributes.get("output_idx_offset_range"),
    )


def build_data(
    query,
    key,
    weights,
    query_scale,
    key_scale,
    candidates,
    lengths,
    topk,
    quant_mode,
    attributes,
):
    """Evaluate actual inputs, without generating replacement data for replay."""
    if quant_mode not in (2, 5):
        raise ValueError("QSLI golden supports quant_mode 2 and 5")
    query, key, weights, query_scale, key_scale, candidates, lengths = (
        to_cpu_tensor(value)
        for value in (query, key, weights, query_scale, key_scale, candidates, lengths)
    )
    layout_q = attributes.get("layout_q", "BSND")
    layout_k = attributes.get("layout_k", "BSND")
    if layout_k == "PA_BBND":
        key, key_scale = unpack_paged(key, key_scale, quant_mode)
    batch, prefix, capacity, used_q, k_prefix, used_k = geometry(query, key, attributes)
    params = compare_params(
        query, key, weights, query_scale, topk, quant_mode, attributes
    )
    inputs = {name: [] for name in ("q", "w", "q_scale", "k_dense", "k_scale_dense")}
    offset = to_cpu_tensor(attributes.get("output_idx_offset"))
    inputs["output_idx_offsets"] = (
        None if offset is None else offset.reshape(-1).tolist()
    )
    inputs["candidate_block_length"] = lengths
    table = to_cpu_tensor(attributes.get("block_table"))
    for b in range(batch):
        start, end = prefix[b : b + 2]
        for name, tensor in (("q", query), ("w", weights), ("q_scale", query_scale)):
            if quant_mode == 2 and name in ("w", "q_scale"):
                tensor = tensor.half()
            inputs[name].append(tensor[b] if layout_q == "BSND" else tensor[start:end])
        if layout_k == "PA_BBND":
            if table is None:
                raise ValueError("QSLI PA golden requires block_table")
            pages = table[b, : math.ceil(used_k[b] / key.shape[1])].long()
            k = key[pages].flatten(0, 1)[: used_k[b]]
            ks = key_scale[pages].flatten(0, 1)[: used_k[b]]
        elif layout_k == "BSND":
            k, ks = key[b, : used_k[b]], key_scale[b, : used_k[b]]
        else:
            k_start = k_prefix[b]
            k = key[k_start : k_start + used_k[b]]
            ks = key_scale[k_start : k_start + used_k[b]]
        inputs["k_dense"].append(k[:, 0])
        inputs["k_scale_dense"].append(ks[:, 0])
    case = {
        "batch": batch,
        "g": query.shape[-2],
        "q_seq": max(capacity, default=0),
        "seqs_q": capacity,
        "seqused_q": used_q,
        "seqused_k": used_k,
        "cmp_ratio": int(attributes.get("cmp_ratio", 1)),
        "cmp_res": list_value(attributes, "cmp_residual_k"),
        "mask_mode": int(attributes.get("mask_mode", 0)),
        "cand_mode": 2,
        "topk": topk,
        "quant_mode": quant_mode,
        "round_score_bf16": quant_mode == 2,
        "score_chunk_size": 64,
    }
    indices, _, score_rows, _ = golden_run_ext(inputs, case, candidates)
    scores = torch.full(
        (batch, 1, max(capacity, default=0), max(used_k, default=0)), NEG_INF
    )
    values = torch.full(indices.shape, NEG_INF, dtype=torch.bfloat16)
    for b in range(batch):
        for i in range(capacity[b]):
            row = prefix[b] + i
            score = score_rows[row]
            scores[b, 0, i] = score
            selected = indices[row, 0].long()
            valid = selected != -1
            if offset is not None:
                selected = selected - int(offset.reshape(-1)[row])
            values[row, 0, valid] = score[selected[valid]].bfloat16()
    shape = (*query.shape[:-2], 1, topk)
    return {
        "params": params,
        "scores": scores,
        "topk_value": scores,
        "cpu_result": indices.reshape(shape),
        "cpu_topk_value": values.reshape(shape),
        "output_idx_offset": offset,
        "score_layout": layout_q,
        "cu_seqlens_q": to_cpu_tensor(prefix) if layout_q == "TND" else None,
    }


CAND_BLOCKS = 2048
CAND_BS = 8


def select_candidate_blocks_ref(score_row, compress_len, topk_blocks, bs):
    """参考实现: pad -> 块内 amax -> pin 尾块 -> topk -> -1 槽 (numpy/torch 交叉验证过)"""
    width = score_row.shape[-1]
    pad = (-width) % bs
    padded = torch.nn.functional.pad(score_row, (0, pad), value=NEG_INF)
    nb = padded.shape[-1] // bs
    blocks = padded.unflatten(-1, (-1, bs)).amax(dim=-1)
    last = (compress_len - 1) // bs
    blocks[last] = float("inf")
    k = min(topk_blocks, nb)
    topv, topi = blocks.topk(k)
    out = torch.full((topk_blocks,), -1, dtype=torch.int32)
    keep = topv > NEG_INF
    out[:k][keep] = topi[keep].to(torch.int32)
    return out, blocks


def golden_run(inputs, case, cand_blocks_in=None):
    return golden_run_ext(inputs, case, cand_blocks_in)[:2]


def _golden_row_score(q_b, i, w_b, qs_b, k_b, ks_b):
    """单行 golden score: q 行 [G,D] x k [S2,D], 与全量 einsum 同数学 (逐行, 大 shape 抽样用)"""
    q_row = q_b[i].to(torch.float32)  # [G, D]
    qk = torch.matmul(q_row, k_b.to(torch.float32).t())  # [G, S2] (int8 值域内精确整数)
    qk_relu = (qk / 1024.0).clamp_min(0.0).to(torch.float16)
    cur_w = (w_b[i] * qs_b[i]).to(torch.float16)  # [G]
    score = torch.matmul(
        cur_w.to(torch.float32).unsqueeze(0), qk_relu.to(torch.float32)
    ).reshape(-1)
    return score * ks_b.to(torch.float32)


def golden_run_ext(inputs, case, cand_blocks_in=None, need_rows=None):
    """
    torch 参考: score (对齐 golden cal_atten_per_batch_int8) + candidate/masked topk;
    额外返回 score_rows (mask 后位置分) 与 blk_rows (amax+pin 后块分) 供官方边界容忍比较。
    TND 输入重整为 per-batch 列表后统一处理; output_idx_offset 逐元素加到 sparse_indices (对齐 arch35 全加语义)。
    need_rows: 行号集合 (全行拼接序) — 大 shape 抽样模式, 仅计算/返回该批行 (逐行 matmul, 避免整 batch 大矩阵)
    """
    b = case["batch"]
    g = case["g"]
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res") or [0] * b
    mask_mode = case["mask_mode"]
    cand_mode = case["cand_mode"]
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * b
    quant_mode = case.get("quant_mode", 2)
    seqused_k = case["seqused_k"]
    seqused_q = case.get("seqused_q") or seqs_q
    if isinstance(inputs["q"], list):
        q_l, w_l, qs_l = inputs["q"], inputs["w"], inputs["q_scale"]
        k_l, ks_l = inputs["k_dense"], inputs["k_scale_dense"]
        if quant_mode == 2:
            q_l = [q.to(torch.int32) for q in q_l]
            k_l = [k.to(torch.int32) for k in k_l]
    else:
        q_all = inputs["q"].cpu()
        w_all = inputs["w"].cpu()
        qs_all = inputs["q_scale"].cpu()
        k_all = inputs["k_dense"].cpu()
        ks_all = inputs["k_scale_dense"].cpu()
        if quant_mode == 2:
            q_all = q_all.to(torch.int32)
            k_all = k_all.to(torch.int32)
        if case.get("layout") == "TND":
            # TND: 仅 Q 侧拼接 (cu_q 前缀切分); K 为 PA 的 [B,1,S2max,D] 稠密表示, 与 BSND 同构索引
            cu_q = inputs["cu_q"]
            q_l = [q_all[cu_q[bi] : cu_q[bi + 1]] for bi in range(b)]
            w_l = [w_all[cu_q[bi] : cu_q[bi + 1]] for bi in range(b)]
            qs_l = [qs_all[cu_q[bi] : cu_q[bi + 1]] for bi in range(b)]
            k_l = [k_all[bi, 0, : seqused_k[bi], :] for bi in range(b)]
            ks_l = [ks_all[bi, 0, : seqused_k[bi]] for bi in range(b)]
        else:
            q_l = [q_all[bi] for bi in range(b)]
            w_l = [w_all[bi] for bi in range(b)]
            qs_l = [qs_all[bi] for bi in range(b)]
            k_l = [k_all[bi, 0, : seqused_k[bi], :] for bi in range(b)]
            ks_l = [ks_all[bi, 0, : seqused_k[bi]] for bi in range(b)]
    offsets = inputs.get("output_idx_offsets") or [0] * (
        sum(seqs_q) if case.get("layout") == "TND" else b * case["q_seq"]
    )
    if cand_blocks_in is not None:
        cand_blocks_in = cand_blocks_in.reshape(
            -1, cand_blocks_in.shape[-1]
        )  # [B,S1,1,cb] -> [总行, cb]

    candidate_lengths = inputs.get("candidate_block_length")
    if candidate_lengths is not None and candidate_lengths.numel():
        candidate_lengths = candidate_lengths.reshape(-1)
    else:
        candidate_lengths = None

    idx_out_rows = []
    cand_out_rows = []
    score_rows = []  # [总行, s2max] mask 后位置分 (mode=2 为候选掩蔽后)
    blk_rows = []  # [总行, nbMax] amax+pin 后块分 (仅 cand_mode==1)
    s2max = max(seqused_k)
    nb_max = (s2max + CAND_BS - 1) // CAND_BS
    row_global = 0
    for bi in range(b):
        s1 = seqs_q[bi]
        s2 = seqused_k[bi]
        act_k = seqused_k[bi] * ratio + res[bi]  # 未压缩长度
        k_b = k_l[bi]  # [S2, D]
        # 抽样模式: 本 batch 需要的行 (全行拼接序)
        batch_need = None
        if need_rows is not None:
            batch_need = [i for i in range(s1) if (row_global + i) in need_rows]
        rows_to_calc = batch_need if batch_need is not None else list(range(s1))
        if not rows_to_calc:
            row_global += s1
            continue
        # 行级有效压缩长度 (vl, valid length: mask_mode=3 下该行可见的压缩后 key 长度)
        valid_lens = []
        for i in range(s1):
            if i >= seqused_q[bi]:
                valid_lens.append(-1)
            elif mask_mode == 3:
                base = act_k - seqused_q[bi] + i + 1
                if base < 0:
                    valid_lens.append(-1)
                else:
                    vl = int(base // ratio)
                    valid_lens.append(vl)
            else:
                valid_lens.append(s2)
        if quant_mode == 5:
            score = torch.full((s1, s2), NEG_INF)
            for begin in range(0, seqused_q[bi], 64):
                end = min(begin + 64, seqused_q[bi])
                score[begin:end] = mxfp4_scores(
                    q_l[bi][begin:end],
                    k_b.unsqueeze(1),
                    w_l[bi][begin:end],
                    qs_l[bi][begin:end],
                    ks_l[bi].unsqueeze(1),
                )
        elif need_rows is None:
            # 全量模式: 整 batch matmul (小 shape)
            score = torch.full((s1, s2), NEG_INF)
            chunk_size = case.get("score_chunk_size", max(s1, 1))
            for begin in range(0, seqused_q[bi], chunk_size):
                end = min(begin + chunk_size, seqused_q[bi])
                q_b = q_l[bi][begin:end].permute(1, 0, 2)  # [G, S1, D]
                qk = torch.matmul(q_b, k_b.t())  # [G, S1, S2] int32
                qk_relu = (
                    (qk.to(torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
                )
                cur_w = (w_l[bi][begin:end] * qs_l[bi][begin:end]).to(torch.float16)
                score[begin:end] = torch.einsum(
                    "in,nij->ij", cur_w.to(torch.float32), qk_relu.to(torch.float32)
                ) * ks_l[bi].to(torch.float32)
        for i in rows_to_calc:
            if quant_mode == 5 or need_rows is None:
                row = score[i]
            else:
                # 抽样模式: 逐行 matmul, 避免大 batch 矩阵
                row = _golden_row_score(q_l[bi], i, w_l[bi], qs_l[bi], k_b, ks_l[bi])
            if case.get("round_score_bf16", False):
                row = round_fp32_to_bf16_cast_round(row).float()
            if valid_lens[i] < 0:
                row = torch.full_like(row, NEG_INF)
            elif valid_lens[i] < s2:
                row[valid_lens[i] :] = NEG_INF
            row_off = offsets[row_global + i] if (row_global + i) < len(offsets) else 0
            # sparse_indices golden (output_idx_offset: R13 语义 — 仅有效槽加偏移, -1 槽保持 -1,
            # 对齐模型 where(idxs < compress_lens, idxs + offset, -1) 与官方比对器 offset 处理)
            idx_row = topk_golden(row, case["topk"])
            if row_off:
                idx_row = torch.where(idx_row >= 0, idx_row + row_off, idx_row).to(
                    torch.int32
                )
            idx_out_rows.append(idx_row)
            row_eff = row
            # candidate golden (相对块号, 不加 offset — §11.6 契约)
            if cand_mode == 1:
                cand_row, blk_score_row = select_candidate_blocks_ref(
                    row, valid_lens[i], case.get("cand_blocks", CAND_BLOCKS), CAND_BS
                )
                cand_out_rows.append(cand_row)
                blk_pad = torch.full((nb_max,), NEG_INF)
                blk_pad[: blk_score_row.numel()] = blk_score_row
                blk_rows.append(blk_pad)
            elif cand_mode == 2:
                assert cand_blocks_in is not None
                blk_list = cand_blocks_in[row_global + i]
                if candidate_lengths is not None:
                    blk_list = blk_list[: int(candidate_lengths[row_global + i])]
                nb = math.ceil(s2 / CAND_BS)
                blk_mask = torch.zeros(nb, dtype=torch.bool)
                valid_blk = blk_list[(blk_list >= 0) & (blk_list < nb)]
                blk_mask[valid_blk.to(torch.long)] = True
                pos_mask = blk_mask.repeat_interleave(CAND_BS)[:s2]
                # 候选外（含不可达位置）→ -inf: 对齐模型 masked_fill(~candidates, -inf);
                # 有效数 < topk 时不足槽位输出 -1（不再有候选外填充泄漏）
                masked = torch.where(pos_mask, row, torch.full_like(row, NEG_INF))
                m_idx = topk_golden(masked, case["topk"])
                if row_off:
                    # R13: 仅有效槽加偏移, -1 槽保持 -1 (对齐模型 where 语义)
                    m_idx = torch.where(m_idx >= 0, m_idx + row_off, m_idx).to(
                        torch.int32
                    )
                idx_out_rows[-1] = m_idx
                row_eff = masked
            # 有效位置分 (mode=2 为候选掩蔽后) 填充到 s2max, 供官方边界容忍比较
            row_pad = torch.full((s2max,), NEG_INF)
            if valid_lens[i] >= 0:
                row_pad[:s2] = row_eff
            score_rows.append(row_pad)
        row_global += s1
    idx_golden = torch.stack(idx_out_rows).reshape(-1, 1, case["topk"])
    if need_rows is not None:
        # 抽样模式: 额外返回行号列表 (调用方按行对位)
        return idx_golden, cand_out_rows, score_rows, blk_rows, sorted(need_rows)
    return idx_golden, cand_out_rows, score_rows, blk_rows
