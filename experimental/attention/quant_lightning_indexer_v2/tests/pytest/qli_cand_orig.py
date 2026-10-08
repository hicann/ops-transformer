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
# QuantLightningIndexerV2 两级TopK (candidate) 设备测试
#
# 来源与本分支改造 (2026-09-30):
#   本文件由 /opt/tjj/qli_cand_test/test_qli_cand.py 拷贝后，按本分支现状改造:
#   - 属性 candidate_mode 已删除：开关改由 candidate_topk_blocks 承载（-1=关闭/现网，2048=开启第一级/source）
#   - 用例按 cand_mode 路由到两个入口（见 npu_run）：
#       cand_mode=3 -> quant_lightning_indexer            现网基础入口（候选关闭，2 输出）
#       cand_mode=1 -> quant_lightning_indexer_candidate  第一级 source（3 输出，末元为块级候选索引）
#     返回统一补成 3 元组，下游比对代码（pin/透传检查）无需改动
#   - cand_mode=2（第二级 consumer 独立算子 quant_sparse_lightning_indexer）的 33 个用例已迁至
#     该算子的测试目录：quant_sparse_lightning_indexer/tests/pytest/qli_cand_orig_consumer.py
#   - 删除「新接口规范适配层」相关用例：该 py 适配层（candidate_source/candidate_consumer 封装）已在本分支删除
#   - 官方比对规则 result_compare_method 改为取自本目录（仓库内），去掉对 /opt 绝对路径的依赖
# 自包含 harness: 输入生成(910b int8/PA) + torch 参考实现 + NPU 调用 + 集合比较
# 用例矩阵: 设计文档 §6.5 (cmp_ratio 1/2 x mode 1/3, g64/32, decode/prefill, numBlocks 边界)
#      注: 1M 组与 cand_blocks=64 (超出本分支 2048-only 契约) 的用例已删除
#
# 运行（需先 source CANN 的 set_env.sh，以取到仓库侧的 cann_ops_transformer 包）:
#   source <CANN>/set_env.sh
#   ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest qli_cand_orig.py [-k <用例 id>]
#   注: 文件名不是 test_*.py，pytest 默认收集不到，需显式指定本文件路径（或改名）
import math
import os
import sys
import time

import numpy as np
import torch
import pytest
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cann_ops_transformer.ops.attention.quant_lightning_indexer_v2 import (  # noqa: E402
    quant_lightning_indexer,
    quant_lightning_indexer_candidate,
    quant_lightning_indexer_metadata,
)
from result_compare_method import compare_topk_valid  # noqa: E402

torch.npu.set_device(0)

NEG_INF = float("-inf")
CAND_BLOCKS = 2048
CAND_BS = 8
HEAD_DIM = 128
NEG_HUGE = -1.0e30


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


def topk_golden(score_row, topk):
    """R11 leak 语义 (对齐模型 where(idxs < compress_lens, idxs + offset, -1)):
    参与池 = 可达位置 (score != -inf, 不可达排除); 候选外可达位置以 NEG_HUGE 参与排序
    (降级但保留入选资格, 仍高于 -inf) — 有效数 < topk 时作为填充泄漏成有效索引, 尾部补 -1。
    排序键 (-score, index) 稳定排序: 候选内按真实分降序, 候选外 tie 取小索引 (regime 2 确定性规则)。"""
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


def gen_case_inputs(case):
    """生成 NPU 输入 (int8/PA) 与 golden 所需的稠密表示; 支持 layout=TND 与 pa_gap (key 0 轴非连续)"""
    b = case["batch"]
    s1 = case["q_seq"]
    g = case["g"]
    seqused_k = case["seqused_k"]
    seqs_q = case.get("seqs_q") or [s1] * b  # TND: 每 batch query 行数
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res")
    topk = case["topk"]
    block_size = case.get("pa_block_size", 128)
    layout = case.get("layout", "BSND")
    pa_gap = case.get("pa_gap", False)
    gen = torch.Generator().manual_seed(case.get("seed", 7))
    np.random.seed(case.get("seed", 7) + 500)  # block_table 排列也固定 (保证可复现)

    if layout == "TND":
        # TND: 仅 Q 侧变长拼接 ([T,G,D] + cu_seqlens_q); K 仍为 PA_BBND (block_table 分页, seqused_k)
        s2max = max(seqused_k)
        q_flat = torch.cat(
            [
                torch.randint(
                    -20, 21, (sq, g, HEAD_DIM), dtype=torch.int32, generator=gen
                )
                for sq in seqs_q
            ]
        ).to(torch.int8)
        q = q_flat.npu()
        cu_q = [0]
        for sq in seqs_q:
            cu_q.append(cu_q[-1] + sq)
        w = (
            torch.cat(
                [torch.empty(sq, g).uniform_(-1, 1, generator=gen) for sq in seqs_q]
            )
            .to(torch.float16)
            .npu()
        )
        q_scale = (
            torch.cat(
                [torch.empty(sq, g).uniform_(0.5, 1.0, generator=gen) for sq in seqs_q]
            )
            .to(torch.float16)
            .npu()
        )
        k_dense = torch.randint(
            -20, 21, (b, 1, s2max, HEAD_DIM), dtype=torch.int32, generator=gen
        ).to(torch.int8)
        k_scale_dense = (
            torch.empty(b, 1, s2max).uniform_(0.5, 1.0, generator=gen).to(torch.float16)
        )
        max_blk = math.ceil(s2max / block_size)
        blk_ids = np.random.permutation(np.arange(max_blk * b + 4)).astype(np.int32)
        block_table = np.full((b, max_blk), -1, dtype=np.int32)
        cur = 0
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                block_table[bi][ib] = blk_ids[cur]
                cur += 1
        block_table_t = torch.from_numpy(block_table).npu()
        k_phys = torch.zeros(max_blk * b + 4, block_size, 1, HEAD_DIM, dtype=torch.int8)
        k_scale_phys = torch.zeros(max_blk * b + 4, block_size, 1, dtype=torch.float16)
        for bi in range(b):
            for ib in range(max_blk):
                bid = block_table[bi][ib]
                if bid == -1:
                    continue
                start = ib * block_size
                end = min(start + block_size, s2max)
                k_phys[bid, : end - start, 0, :] = k_dense[bi, 0, start:end, :]
                k_scale_phys[bid, : end - start, 0] = k_scale_dense[bi, 0, start:end]
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32).npu()
        seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32).npu()
        cmp_res_t = (
            torch.tensor(res, dtype=torch.int32).npu() if res is not None else None
        )
        metadata = quant_lightning_indexer_metadata(
            num_heads_q=g,
            num_heads_k=1,
            head_dim=HEAD_DIM,
            topk=topk,
            quant_mode=2,
            cu_seqlens_q=cu_q_t,
            seqused_k=seqused_k_t,
            cmp_residual_k=cmp_res_t,
            batch_size=b,
            max_seqlen_q=max(seqs_q),
            max_seqlen_k=s2max,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=case["mask_mode"],
            cmp_ratio=ratio,
        )
        return {
            "q": q,
            "w": w,
            "q_scale": q_scale,
            "k_dense": k_dense,
            "k_scale_dense": k_scale_dense,
            "k_phys": k_phys.npu(),
            "k_scale_phys": k_scale_phys.npu(),
            "block_table": block_table_t,
            "seqused_k_t": seqused_k_t,
            "cu_q_t": cu_q_t,
            "cu_k_t": None,
            "cmp_res_t": cmp_res_t,
            "metadata": metadata,
            "output_idx_offset_t": None,
            "output_idx_offsets": None,
            "k_pool_blocks": None,
            "table": None,
            "s2max": s2max,
            "seqs_q": seqs_q,
            "cu_q": cu_q,
        }

    # BSND / PA_BBND 基础路径
    q = (
        torch.randint(-20, 21, (b, s1, g, HEAD_DIM), dtype=torch.int32, generator=gen)
        .to(torch.int8)
        .npu()
    )
    w = torch.empty(b, s1, g).uniform_(-1, 1, generator=gen).to(torch.float16).npu()
    q_scale = (
        torch.empty(b, s1, g).uniform_(0.5, 1.0, generator=gen).to(torch.float16).npu()
    )

    # 稠密 k/k_scale (golden 用), 再按 block_table 散射成 PA 物理布局
    s2max = max(seqused_k)
    k_dense = torch.randint(
        -20, 21, (b, 1, s2max, HEAD_DIM), dtype=torch.int32, generator=gen
    ).to(torch.int8)
    k_scale_dense = (
        torch.empty(b, 1, s2max).uniform_(0.5, 1.0, generator=gen).to(torch.float16)
    )

    max_blk_per_batch = math.ceil(s2max / block_size)
    gap = (
        2 if pa_gap else 1
    )  # pa_gap: 每视图块后跟一个 gap 块, tensor stride0 = 2x 紧凑
    pool = case.get("pool") or ((max_blk_per_batch * b + 4) * gap)
    table_w = case.get("table_w") or max_blk_per_batch
    total_blocks = pool
    inputs_cu_k = [0]
    for s2 in seqused_k:
        inputs_cu_k.append(inputs_cu_k[-1] + s2)
    if not pa_gap:
        # 紧凑: block_table 值 = 物理块号 (池内连续编号的置换)
        blk_ids = np.random.permutation(np.arange(total_blocks)).astype(np.int32)
        block_table = np.full((b, table_w), -1, dtype=np.int32)
        cur = 0
        for bi in range(b):
            need = math.ceil(seqused_k[bi] / block_size)
            for ib in range(need):
                block_table[bi][ib] = blk_ids[cur]
                cur += 1
        block_table_t = torch.from_numpy(block_table).npu()

    if pa_gap:
        # 0 轴非连续 (keyStride0 语义, 对齐 arch35):
        # block_table 值 = 视图块号 (0..nBlk-1), tensor 实际 stride0 = gap x 紧凑 (块间全零 gap 段);
        # 正确 kernel 应按 表值 x keyStride0 寻址; 紧凑假设 kernel 按表值 x 紧凑寻址会落错地址
        n_blk = total_blocks // gap
        blk_ids = np.random.permutation(np.arange(n_blk)).astype(
            np.int32
        )  # 视图块号, 不乘 gap
        block_table = np.full((b, table_w), -1, dtype=np.int32)
        cur = 0
        for bi in range(b):
            need = math.ceil(seqused_k[bi] / block_size)
            for ib in range(need):
                block_table[bi][ib] = blk_ids[cur]
                cur += 1
        block_table_t = torch.from_numpy(block_table).npu()
        blk_elems = block_size * 1 * HEAD_DIM
        buf = torch.zeros(n_blk * gap * blk_elems, dtype=torch.int8, device="npu")
        k_phys = torch.as_strided(
            buf,
            (n_blk, block_size, 1, HEAD_DIM),
            (gap * blk_elems, HEAD_DIM, HEAD_DIM, 1),
        )
        sbuf = torch.zeros(n_blk * gap * block_size, dtype=torch.float16, device="npu")
        k_scale_phys = torch.as_strided(
            sbuf, (n_blk, block_size, 1), (gap * block_size, 1, 1)
        )
        kp = torch.zeros(n_blk, block_size, 1, HEAD_DIM, dtype=torch.int8)
        ksp = torch.zeros(n_blk, block_size, 1, dtype=torch.float16)
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                bid = block_table[bi][ib]
                if bid == -1:
                    continue
                start = ib * block_size
                end = min(start + block_size, s2max)
                kp[bid, : end - start, 0, :] = k_dense[bi, 0, start:end, :]
                ksp[bid, : end - start, 0] = k_scale_dense[bi, 0, start:end]
        k_phys.copy_(kp.npu())
        k_scale_phys.copy_(ksp.npu())
    else:
        k_phys = torch.zeros(total_blocks, block_size, 1, HEAD_DIM, dtype=torch.int8)
        k_scale_phys = torch.zeros(total_blocks, block_size, 1, dtype=torch.float16)
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                bid = block_table[bi][ib]
                if bid == -1:
                    continue
                start = ib * block_size
                end = min(start + block_size, s2max)
                k_phys[bid, : end - start, 0, :] = k_dense[bi, 0, start:end, :]
                k_scale_phys[bid, : end - start, 0] = k_scale_dense[bi, 0, start:end]
        k_phys = k_phys.npu()
        k_scale_phys = k_scale_phys.npu()
    seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32).npu()

    cmp_res_t = None
    if res is not None:
        cmp_res_t = torch.tensor(res, dtype=torch.int32).npu()

    # output_idx_offset (A15 测试): 每行 int32 偏移, golden 同步持有
    off_vals = case.get("output_idx_offset")
    output_idx_offset_t = None
    output_idx_offsets = None
    if off_vals is not None:
        total_rows = sum(seqs_q)
        if (
            off_vals == "per_batch"
        ):  # 每 batch 常量: [cu_k[bi] 前缀] (TND 绝对位置还原场景)
            off_list = []
            for bi in range(b):
                off_list.extend(
                    [inputs_cu_k[bi] if layout == "TND" else bi * 1000] * seqs_q[bi]
                )
        elif callable(off_vals):
            off_list = [off_vals(r) for r in range(total_rows)]
        else:
            off_list = (
                list(off_vals) * (total_rows // len(off_vals))
                if len(off_vals) < total_rows
                else list(off_vals)
            )
        output_idx_offsets = off_list
        # host 要求 shape dim=3: BSND [B, maxS1, N2] / TND [T, N2] — 统一 [rows, 1, 1] 补齐 3 维
        output_idx_offset_t = (
            torch.tensor(off_list[:total_rows], dtype=torch.int32)
            .reshape(-1, 1, 1)
            .npu()
        )

    # metadata: 负载均衡分核
    metadata = quant_lightning_indexer_metadata(
        num_heads_q=g,
        num_heads_k=1,
        head_dim=HEAD_DIM,
        topk=topk,
        quant_mode=2,
        seqused_k=seqused_k_t,
        cmp_residual_k=cmp_res_t,
        batch_size=b,
        max_seqlen_q=max(seqs_q),
        max_seqlen_k=s2max,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=case["mask_mode"],
        cmp_ratio=ratio,
    )

    return {
        "q": q,
        "w": w,
        "q_scale": q_scale,
        "k_dense": k_dense,
        "k_scale_dense": k_scale_dense,
        "k_phys": k_phys,
        "k_scale_phys": k_scale_phys,
        "block_table": block_table_t,
        "seqused_k_t": seqused_k_t,
        "cu_q_t": None,
        "cu_k_t": None,
        "cmp_res_t": cmp_res_t,
        "metadata": metadata,
        "output_idx_offset_t": output_idx_offset_t,
        "output_idx_offsets": output_idx_offsets,
        "s2max": s2max,
        "max_blk_per_batch": max_blk_per_batch,
        "seqs_q": seqs_q,
        "k_pool_blocks": total_blocks,
        "table": None,
    }


def npu_run(inputs, case, cand_in_npu=None):
    """按本分支契约分发到两个入口（无 candidate_mode 属性，开关= candidate_topk_blocks）：
      cand_mode=3 -> 现网基础入口（候选关闭；2 输出）
      cand_mode=1 -> 第一级 source（candidate_topk_blocks=2048；3 输出，末元为块级候选索引）
    统一返回 3 元组 (indices, values, candidate)，与下游比对/pin/透传检查约定一致。"""
    layout = case.get("layout", "BSND")
    mode = case["cand_mode"]
    common = dict(
        topk=case["topk"],
        quant_mode=2,
        cu_seqlens_q=inputs.get("cu_q_t"),
        cu_seqlens_k=inputs.get("cu_k_t"),
        seqused_k=inputs["seqused_k_t"],
        cmp_residual_k=inputs.get("cmp_res_t"),
        block_table=inputs["block_table"],
        output_idx_offset=inputs.get("output_idx_offset_t"),
        metadata=inputs["metadata"],
        max_seqlen_q=max(case.get("seqs_q", [case["q_seq"]])),
        layout_q=layout,
        layout_k="PA_BBND",
        mask_mode=case["mask_mode"],
        cmp_ratio=case["cmp_ratio"],
    )
    empty_cand = torch.empty(0, dtype=torch.int32, device=inputs["q"].device)

    if mode == 3:
        idx, vals = quant_lightning_indexer(
            inputs["q"],
            inputs["k_phys"],
            inputs["w"],
            inputs["q_scale"],
            inputs["k_scale_phys"],
            return_value=0,
            **common,
        )
        return idx, vals, empty_cand

    cand_blocks = case.get("cand_blocks", CAND_BLOCKS)
    if cand_blocks != CAND_BLOCKS:
        pytest.skip(
            f"本分支 candidate_topk_blocks 仅支持 {CAND_BLOCKS}（用例给 {cand_blocks}）"
        )

    if mode == 1:
        ret = quant_lightning_indexer_candidate(
            inputs["q"],
            inputs["k_phys"],
            inputs["w"],
            inputs["q_scale"],
            inputs["k_scale_phys"],
            candidate_topk_blocks=CAND_BLOCKS,
            candidate_block_size=CAND_BS,
            **common,
        )
        # 末元为块级候选索引；第 4 元（若存在）是预留空输出 candidate_block_length，比对只覆盖前 3 输出
        return ret[0], ret[1], ret[2]

    raise RuntimeError(f"未知 cand_mode={mode}")


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
    """torch 参考: score (对齐 golden cal_atten_per_batch_int8) + candidate/masked topk;
    额外返回 score_rows (mask 后位置分) 与 blk_rows (amax+pin 后块分) 供官方边界容忍比较。
    TND 输入重整为 per-batch 列表后统一处理; output_idx_offset 逐元素加到 sparse_indices (对齐 arch35 全加语义)。
    need_rows: 行号集合 (全行拼接序) — 大 shape 抽样模式, 仅计算/返回该批行 (逐行 matmul, 避免整 batch 大矩阵)"""
    b = case["batch"]
    g = case["g"]
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res") or [0] * b
    mask_mode = case["mask_mode"]
    cand_mode = case["cand_mode"]
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * b
    q_all = inputs["q"].cpu().to(torch.int32)
    w_all = inputs["w"].cpu()
    qs_all = inputs["q_scale"].cpu()
    k_all = inputs["k_dense"].cpu().to(torch.int32)
    ks_all = inputs["k_scale_dense"].cpu()
    seqused_k = case["seqused_k"]
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
            if mask_mode == 3:
                base = act_k - s1 + i + 1
                if base < 0:
                    valid_lens.append(-1)
                else:
                    vl = int(base // ratio)
                    valid_lens.append(vl)
            else:
                valid_lens.append(s2)
        if need_rows is None:
            # 全量模式: 整 batch matmul (小 shape)
            q_b = q_l[bi].permute(1, 0, 2)  # [G, S1, D]
            qk = torch.matmul(q_b, k_b.t())  # [G, S1, S2] int32
            qk_relu = (qk.to(torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
            cur_w = (w_l[bi] * qs_l[bi]).to(torch.float16)  # [S1, G]
            score = torch.einsum(
                "in,nij->ij", cur_w.to(torch.float32), qk_relu.to(torch.float32)
            )
            score = score * ks_l[bi].to(torch.float32)  # [S1, S2]
            # 行级 mask 原位写回 (mask_mode=3: vl 之后置 -inf; 全无效行整行 -inf)
            for i in range(s1):
                if mask_mode == 3:
                    if valid_lens[i] < 0:
                        score[i, :] = NEG_INF
                    elif valid_lens[i] < s2:
                        score[i, valid_lens[i] :] = NEG_INF
        for i in rows_to_calc:
            if need_rows is None:
                row = score[i]
            else:
                # 抽样模式: 逐行 matmul, 避免大 batch 矩阵
                row = _golden_row_score(q_l[bi], i, w_l[bi], qs_l[bi], k_b, ks_l[bi])
                if mask_mode == 3:
                    vl_r = valid_lens[i]
                    if vl_r < 0:
                        row = torch.full_like(row, NEG_INF)
                    elif vl_r < s2:
                        row[vl_r:] = NEG_INF
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
                nb = math.ceil(s2 / CAND_BS)
                blk_mask = torch.zeros(nb, dtype=torch.bool)
                valid_blk = blk_list[(blk_list >= 0) & (blk_list < nb)]
                blk_mask[valid_blk.to(torch.long)] = True
                pos_mask = blk_mask.repeat_interleave(CAND_BS)[:s2]
                # R11 leak: 候选外可达 → NEG_HUGE (降级但保留入选资格); 不可达 (row 已 -inf) 保持 -inf
                masked = torch.where(
                    pos_mask | (row == NEG_INF), row, torch.full_like(row, NEG_HUGE)
                )
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


def gen_cand_input(case, kind, inputs):
    """mode=2 的候选输入: self=参考选择(自洽) / rand=随机子集; 返回 npu tensor 与 golden 块列表。
    大 shape (行数 > 4096) 的 rand 改用向量化 randint (含重复块, golden 展开幂等; self 不可用)"""
    b = case["batch"]
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * b
    seqused_k = case["seqused_k"]
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res") or [0] * b
    mask_mode = case["mask_mode"]
    cb = case.get("cand_blocks", CAND_BLOCKS)
    total_rows = sum(seqs_q)
    nb_max = (max(seqused_k) + CAND_BS - 1) // CAND_BS
    if total_rows > 4096:
        assert kind != "self", "大 shape 仅支持 randint 候选"
        # 期望 shape: TND [T, 1, cb] (拼接总行) / BSND [B, S1, 1, cb]; 变长 seqs_q 逐批生成
        if case.get("layout") == "TND":
            gen_dev = torch.Generator(device="npu").manual_seed(case.get("seed", 7) + 1)
            parts = [
                torch.randint(
                    0,
                    nb_max,
                    (sq, 1, cb),
                    dtype=torch.int32,
                    device="npu",
                    generator=gen_dev,
                )
                for sq in seqs_q
            ]
            rows_npu = torch.cat(parts, dim=0)
            cand_golden = rows_npu.reshape(-1, cb).cpu()
            return rows_npu, cand_golden
        gen_dev = torch.Generator(device="npu").manual_seed(case.get("seed", 7) + 1)
        parts = [
            torch.randint(
                0,
                nb_max,
                (sq, 1, cb),
                dtype=torch.int32,
                device="npu",
                generator=gen_dev,
            )
            for sq in seqs_q
        ]
        rows_npu = torch.stack(
            parts, dim=0
        )  # [B, S1, 1, cb] (要求各批同长; 变长走下方逐行分支)
        cand_golden = rows_npu.reshape(-1, cb).cpu()
        return rows_npu, cand_golden
    gen = torch.Generator().manual_seed(case.get("seed", 7) + 1)
    rows = []
    for bi in range(b):
        s1 = seqs_q[bi]
        s2 = seqused_k[bi]
        act_k = seqused_k[bi] * ratio + res[bi]
        nb = math.ceil(s2 / CAND_BS)
        for i in range(s1):
            if kind == "self":
                # 复用 golden 的 score 流程 (与 golden_run 一致), 取参考选择
                q_b = inputs["q"].cpu().to(torch.int32)[bi].permute(1, 0, 2)
                k_b = inputs["k_dense"].cpu().to(torch.int32)[bi, 0, :s2, :]
                qk = torch.matmul(q_b, k_b.t())
                qk_relu = (
                    (qk.to(torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
                )
                cur_w = (inputs["w"].cpu()[bi] * inputs["q_scale"].cpu()[bi]).to(
                    torch.float16
                )
                score = torch.einsum(
                    "in,nij->ij", cur_w.to(torch.float32), qk_relu.to(torch.float32)
                )
                score = score * inputs["k_scale_dense"].cpu()[bi, 0, :s2].to(
                    torch.float32
                )
                if mask_mode == 3:
                    base = act_k - s1 + i + 1
                    vl = int(base // ratio) if base >= 0 else -1
                    if vl >= 0 and vl < s2:
                        score[i, vl:] = NEG_INF
                compress_len = vl if mask_mode == 3 else s2
                cand_row, _ = select_candidate_blocks_ref(
                    score[i], compress_len, cb, CAND_BS
                )
                rows.append(cand_row)
            else:
                nb_sel = min(cb, nb)
                sel = torch.randperm(nb, generator=gen)[:nb_sel].to(torch.int32)
                out = torch.full((cb,), -1, dtype=torch.int32)
                out[:nb_sel] = sel
                rows.append(out)
    cand_golden = torch.stack(rows).reshape(b, -1, 1, cb)
    return cand_golden.npu(), cand_golden


# =====================================================================================
# ---- 大 shape 全量比对 (2026-09-10, 取代 §6.6 抽样机制): NPU 设备侧 golden 全行完全匹配 ----
# 两级机制:
#   ① 筛选: NPU fp16 参考全行多重集合比较 — qk 矩阵乘 fp16 单次输出舍入, 误差 ≤ 2^-11 (实测 4.88e-4);
#   ② 仲裁: 筛选不一致的行用 NPU fp32 重算 (int8 值域内 fp32 矩阵乘与 CPU int32 逐位一致, 探测验证)
#      + 官方两级规则 (cmp_indices: 多重集合 + 边界 0.001) — 门禁结论不依赖 fp16 误差与容差的余量,
#      只有与精确参考的真实差异 (> 0.001 相对) 才会 FAIL。
# =====================================================================================


def _gold_idx_det(score, topk):
    """确定性 topk 选择: 值降序, tie 取小索引 (对齐 kernel MrgSort (value,idx) 与 CPU golden 稳定排序)。
    score [n, L] fp32 (不可达=-inf, mode=2 候选外=NEG_HUGE); 返回 [n, topk] int32, 不足 -1 填充。
    比 torch.topk 多一遍小排序 (仅 k+slots 宽), 消除 tie 任意序导致的多重集合假性失配。"""
    n, length = score.shape
    k_eff = min(length, topk)
    topv, topi = torch.topk(score, k_eff, dim=-1)
    bv = topv[:, -1:]  # 第 k 大值 (选择边界)
    above = topv > bv  # topk 槽位中严格大于边界的 (索引正确, 与 tie 序无关)
    colj = torch.arange(length, device=score.device).unsqueeze(0)
    tied_idx = torch.where(score == bv, colj, length)  # tie 位置→索引, 其余→哨兵 L
    slots = (k_eff - above.sum(-1, keepdim=True)).clamp(min=0)  # 需用 tie 补的槽位数
    max_slots = max(1, int(slots.max().item()))
    tied_sorted, _ = torch.topk(
        tied_idx, max_slots, dim=-1, largest=False
    )  # tie 索引升序
    rank = torch.arange(max_slots, device=score.device).unsqueeze(0)
    tie_ok = (
        (rank < slots) & (tied_sorted < length) & (bv > NEG_INF)
    )  # -inf 边界不补 tie
    a_idx = torch.where(above, topi, -1)
    t_idx = torch.where(tie_ok, tied_sorted, -1)
    gi, _ = torch.sort(
        torch.cat([a_idx, t_idx], dim=1), dim=-1, descending=True
    )  # 索引在前, -1 沉底
    if gi.shape[1] < topk:
        gi = torch.nn.functional.pad(gi, (0, topk - gi.shape[1]), value=-1)
    gi = gi[:, :topk]
    # 边界元素换到行尾: 官方 compare_topk_valid 取 gold 行最后有效槽的分数为边界值 (value_bm =
    # topk_value[..., cur_cpu[-1]]), 隐含"行按值降序、行尾=第 k 大"的约定 (kernel/topk_golden 均满足)。
    # 本函数按索引排序会破坏该约定 → 边界值取到无关元素 → 回退判据失真 (实测 0.19 假阳性)。
    b_tie = tied_sorted.gather(
        1, (slots.clamp(min=1) - 1)
    )  # tie 已选中者的最大索引 (值=边界)
    pos_nf = ((topv > NEG_INF).sum(-1, keepdim=True) - 1).clamp(min=0)
    b_last = topi.gather(1, pos_nf)  # bv=-inf 时: 值最小的有限元素
    has_finite = (topv > NEG_INF).any(-1, keepdim=True)
    b_elem = torch.where(
        (slots >= 1) & (bv > NEG_INF),
        b_tie,
        torch.where(has_finite, b_last, torch.full_like(b_tie, -1)),
    )
    n_valid = (gi >= 0).sum(-1, keepdim=True)
    pos_last = (n_valid - 1).clamp(min=0)
    old_last = gi.gather(1, pos_last)
    pos_b = (
        (gi == b_elem).to(torch.uint8).argmax(-1, keepdim=True)
    )  # b_elem 所在列 (唯一)
    gi = gi.scatter(1, pos_b, old_last)  # 原尾部元素挪到 b_elem 原位
    gi = gi.scatter(1, pos_last, b_elem).to(torch.int32)  # 行尾放边界元素
    return gi


def _gold_cand_from_score(score, vl_rows, cb, s2, det=False):
    """score [n, L] → (候选块 golden [n, cb], 块分 [n, nb]): pad→块内 amax→pin 尾块 +inf→topk→-1 槽。
    det=False: 普通 topk (筛选用, tie 任意序 → 失配行由 r16 仲裁兜底);
    det=True: 确定性 idx-asc tie-break (仲裁用, 与 kernel 逐位对齐)。"""
    n = score.shape[0]
    padw = (-score.shape[-1]) % CAND_BS
    scorep = torch.nn.functional.pad(score, (0, padw), value=NEG_INF)
    blocks = scorep.reshape(n, -1, CAND_BS).amax(dim=-1)
    nb_c = blocks.shape[1]
    pin_blk = (vl_rows.clamp(max=s2) - 1).div(CAND_BS, rounding_mode="floor")
    pvalid = (vl_rows > 0) & (pin_blk >= 0) & (pin_blk < nb_c)
    pidx = pin_blk.clamp(min=0, max=nb_c - 1).unsqueeze(1)
    cur = blocks.gather(1, pidx)
    blocks.scatter_(
        1,
        pidx,
        torch.where(pvalid.unsqueeze(1), torch.full_like(cur, float("inf")), cur),
    )
    if nb_c <= cb:
        # 快路径: 块数 ≤ 容器 → 全部有限块入选 (集合与 tie 无关), 免 topk/det
        col = torch.arange(nb_c, device=score.device).unsqueeze(0).expand(n, -1)
        gc = torch.where(blocks > NEG_INF, col, torch.full_like(col, -1)).to(
            torch.int32
        )
        if nb_c < cb:
            gc = torch.nn.functional.pad(gc, (0, cb - nb_c), value=-1)
        # 边界块 (有限块中块分最小者) 换到行尾, 供边界域仲裁定位 (块分含 fp16 舍入, 仅定位用)
        gvals = torch.where(
            gc >= 0,
            blocks.gather(1, gc.clamp(min=0).long()),
            torch.full_like(blocks[:, :1].expand(n, gc.shape[1]), float("inf")),
        )
        _, bmini = torch.topk(gvals, 1, dim=-1, largest=False)  # argmin (NPU argmin 慢)
        nv = (gc >= 0).sum(-1, keepdim=True)
        pos_last = (nv - 1).clamp(min=0)
        old_last = gc.gather(1, pos_last)
        gc = gc.scatter(1, bmini, old_last).scatter(1, pos_last, gc.gather(1, bmini))
        return gc, blocks
    gc = _gold_idx_det(
        blocks, cb
    )  # 确定性 tie-break, 与位置级选择同语义 (块号=索引); 行尾=边界
    return gc, blocks
    if det:
        gc = _gold_idx_det(
            blocks, cb
        )  # 确定性 tie-break, 与位置级选择同语义 (块号=索引)
        return gc, blocks
    cb_eff = min(cb, nb_c)
    topvb, topib = torch.topk(blocks, cb_eff, dim=-1)
    gc = torch.where(topvb > NEG_INF, topib, torch.full_like(topib, -1)).to(torch.int32)
    if cb_eff < cb:
        gc = torch.nn.functional.pad(gc, (0, cb - cb_eff), value=-1)
    return gc, blocks


def _npu_score_rows_exact(q_rows, w_rows, k32, ks, vl_rows, s2, cand_rows, nb_total):
    """NPU fp32 精确 score: q_rows [n,g,D] int8 → [n, s2] fp32 (含因果 mask; mode=2 候选掩蔽 R11 leak)。
    int8 值域内 fp32 矩阵乘为精确整数算术 (|Σ q·k| ≤ 128·127·127 < 2^24), 与 CPU int32 golden 等价。"""
    qk = torch.matmul(q_rows.to(torch.float32), k32.t())  # [n,g,s2] 精确
    r16 = (qk / 1024.0).clamp_min(0.0).to(torch.float16)  # 与 golden 链一致 (fp16 量化)
    score = (
        torch.bmm(w_rows.to(torch.float32).unsqueeze(1), r16.float()).squeeze(1) * ks
    )
    colj = torch.arange(s2, device=score.device)
    score = score.masked_fill(colj >= vl_rows.unsqueeze(1), NEG_INF)
    if cand_rows is not None:
        n = q_rows.shape[0]
        # index 双向 clamp: 变长 batch 下候选 randint 按最大 nb 生成, 小 batch 行含越界块号
        # (≥ 本批 nb_total) — scatter 越界寻址触发 aicpu 异常; 越界块按"非候选"处理 (与 golden
        # valid_blk = (>=0 & < nb) 语义一致)
        ids = cand_rows.clamp(0, nb_total - 1).long()
        validb = (cand_rows >= 0) & (cand_rows < nb_total)
        # scatter_add 而非 bool scatter: 候选 randint 含重复块 + -1 pad clamp 到 0,
        # 重复 index 的 bool scatter 是竞态 (实测 b2_tnd 触发 aicpu ScatterElements 异常 0x2a);
        # scatter_add 重复=累加, 语义确定
        memb = torch.zeros(n, nb_total, dtype=torch.int32, device=score.device)
        memb.scatter_add_(1, ids, validb.to(torch.int32))
        # repeat_interleave 在 NPU 上极慢 (实测 4096x2048→x8 需 9.2s), 用等价 gather (27ms)
        pos_mask = memb[:, torch.arange(s2, device=score.device) // CAND_BS] > 0
        score = torch.where(
            pos_mask | (score == NEG_INF), score, torch.full_like(score, NEG_HUGE)
        )
    return score


def _pin_check_vec(case, cand_npu, seqs_q):
    """pin 向量化全行检查: vl ≥ 1 的行, pin 块 ((vl-1)//8) 必须在 op 候选输出中"""
    cb = case.get("cand_blocks", CAND_BLOCKS)
    cand_flat = cand_npu.reshape(-1, cb)
    fails = []
    row = 0
    for bi in range(case["batch"]):
        s1b = seqs_q[bi]
        s2b = case["seqused_k"][bi]
        act_k = (
            s2b * case["cmp_ratio"] + (case.get("cmp_res") or [0] * case["batch"])[bi]
        )
        if case["mask_mode"] == 3:
            basev = act_k - s1b + torch.arange(s1b) + 1
            vl = torch.div(basev, case["cmp_ratio"], rounding_mode="floor")
            vl = torch.where(basev >= 0, vl, torch.full_like(vl, -1))
        else:
            vl = torch.full((s1b,), s2b, dtype=torch.int64)
        vl = vl.clamp(max=s2b)
        pin_blk = (vl - 1).div(CAND_BS, rounding_mode="floor")
        need = pin_blk >= 0
        if bool(need.any()):
            sub = cand_flat[row : row + s1b]
            has = (sub == pin_blk.npu().unsqueeze(1)).any(dim=1)
            bad = (~has & need.npu()).nonzero().flatten().tolist()
            for i in bad:
                fails.append(f"pin@b{bi}r{i} 块{int(pin_blk[i])} 不在候选")
        row += s1b
    assert not fails, f"[{case['name']}] pin 失败: {fails[:5]}"
    print(f"[{case['name']}] pin OK (向量化全行)")


def _val_gate(score, ga, gb, thres=0.0):
    """值域门 (官方边界容忍规则的向量化, [n,k] 空间无大张量): 索引多重集合失配的行, 若
    (a) 排序后值逐位 |a-b| <= thres*|b| (thres=0 → 精确相等; -1 槽计数硬相等), 且
    (b) 值 > 边界值*(1+thres) 的索引集合相等 (差异只可能发生在边界容忍带内)
    → 判 PASS。thres=0.001 + 精确分数 (r16 fp32 gsum) = 官方 compare_topk_valid 的等价向量化;
    thres=0 + fp16 筛选分数 = 保守快门 (仅精确等值并列通过, 精确域差 ≤ 2^-11 < 0.001, sound)。
    score [n, L]; ga/gb [n, k] 索引。返回 bool [n]。"""
    n, length = score.shape
    va = torch.where(
        (ga >= 0) & (ga < length), score.gather(1, ga.clamp(0, length - 1)), NEG_INF
    )
    vb = torch.where(
        (gb >= 0) & (gb < length), score.gather(1, gb.clamp(0, length - 1)), NEG_INF
    )
    vsa, _ = torch.sort(va, dim=-1, descending=True)
    vsb, _ = torch.sort(vb, dim=-1, descending=True)
    fa = vsa > NEG_INF
    fb = vsb > NEG_INF
    both = fa & fb
    tol_ok = ((vsa - vsb).abs() <= thres * vsb.abs().clamp_min(1e-30)) | ~both
    veq = (~(fa ^ fb)).all(-1) & tol_ok.all(-1)  # -1 槽数相等 + 有限位逐位容忍
    okb = (gb >= 0) & (gb < length)
    nb = okb.sum(-1, keepdim=True)
    bv_g = vsb.gather(1, (nb - 1).clamp(min=0))  # gold 侧边界值 (最小有效值)
    above = bv_g + thres * bv_g.clamp_min(0)  # 边界容忍带上沿 (bv<=0 时 = bv)
    ia, _ = torch.sort(torch.where(va > above, ga, -1), dim=-1, descending=True)
    ib, _ = torch.sort(torch.where(vb > above, gb, -1), dim=-1, descending=True)
    return veq & (ia == ib).all(-1)


def _arb_sparse_rows(w_rows, r16_full, js, ks_seg, vl_rows, cand_memb, ga, gb, col):
    """边界域精确仲裁 (官方 compare_topk_valid 的等价向量化): 对索引多重集合失配的行, 仅对
    差异元素与 gold 边界元素用 r16 列 gather 计算精确分数 (O(32·diff) 微秒级, 免 gsum/det/CPU),
    按官方 0.001 相对容差判定 (bv=0 时要求差值恰为 0, 对齐官方 value_bm==0 特例)。
    w_rows [n,g] fp16 (坏行子集); r16_full [chunk,g,col] fp16 + js [n] 坏行行号 (内部子批拷贝);
    ks_seg [col]; vl_rows [n]; cand_memb [n, nb_total] bool (mode=2) 或 None;
    ga/gb [n,k] gold/kernel 索引 (含 -1 槽; gold 行尾有效槽 = 边界元素, _gold_idx_det 保证)。
    返回 bool [n]。"""
    n, k = ga.shape
    dev = ga.device
    gnum = r16_full.shape[1]
    big = col + 1
    va = ga >= 0
    vb = gb >= 0
    ok = va.sum(-1) == vb.sum(-1)  # -1 槽数硬契约
    sa, _ = torch.sort(torch.where(va, ga, big), dim=-1)  # 有效升序在前, 哨兵沉底
    sb, _ = torch.sort(torch.where(vb, gb, big), dim=-1)
    ok = ok & ~(
        (((sa[:, 1:] == sa[:, :-1]) & (sa[:, 1:] < big)).any(-1))
        | (((sb[:, 1:] == sb[:, :-1]) & (sb[:, 1:] < big)).any(-1))
    )  # 官方 has_duplicate → FAIL
    posa = torch.searchsorted(sb, ga.clamp(min=0))
    in_b = va & (posa < k) & (sb.gather(1, posa.clamp(max=k - 1)) == ga)
    posb = torch.searchsorted(sa, gb.clamp(min=0))
    in_a = vb & (posb < k) & (sa.gather(1, posb.clamp(max=k - 1)) == gb)
    da = torch.where(va & ~in_b, ga, big)  # A 独有差异
    db = torch.where(vb & ~in_a, gb, big)  # B 独有差异
    nd = max(
        1, int(torch.maximum((da < big).sum(-1).max(), (db < big).sum(-1).max()).item())
    )
    dac, _ = torch.topk(da, nd, dim=-1, largest=False)  # 差异索引升序, big 填充
    dbc, _ = torch.topk(db, nd, dim=-1, largest=False)
    bidx = ga.gather(
        1, (va.sum(-1, keepdim=True) - 1).clamp(min=0)
    )  # gold 边界元素 (行尾有效槽)
    all_idx = torch.cat([dac, dbc, bidx], dim=1)  # [n, 2nd+1], 末列 = 边界
    m = all_idx.shape[1]
    idxc = all_idx.clamp(0, col - 1)
    validp = (all_idx < big) & (idxc < vl_rows.unsqueeze(1))  # 真实且可达
    if cand_memb is not None:
        inc = cand_memb.gather(1, (idxc // CAND_BS).clamp(max=cand_memb.shape[1] - 1))
    else:
        inc = torch.ones_like(validp)
    # 一维扁平 take: 仅读取所需 n×g×m 个元素 (r16[js] 行拷贝/三维高级索引均会物化整行, 不可用)
    flat = r16_full.reshape(-1)  # [c·g·col] 视图
    rows = (js * gnum).unsqueeze(1) + torch.arange(gnum, device=dev).unsqueeze(
        0
    )  # [n, g]
    fidx = rows.unsqueeze(2) * col + idxc.unsqueeze(1)  # [n, g, m]
    colsg = flat[fidx]  # [n, g, m] fp16
    vals = torch.bmm(w_rows.to(torch.float32).unsqueeze(1), colsg.float()).squeeze(
        1
    )  # [n, m] 精确 g-sum
    vals = vals * ks_seg[idxc]
    vals = torch.where(
        validp & inc,
        vals,
        torch.where(validp, torch.full_like(vals, NEG_HUGE), NEG_INF),
    )
    bv = vals[:, -1:]  # gold 边界精确值
    vdiff = vals[:, :-1]
    dmask = torch.cat([dac < big, dbc < big], dim=1)
    adiff = (vdiff - bv).abs()
    passm = torch.where(bv == 0, adiff == 0, adiff <= 0.001 * bv.abs()) | ~dmask
    return ok & passm.all(-1)


def _arb_cand_rows(w_rows, r16_full, js, ks_seg, vl_rows, ga, gb, col):
    """候选块输出的边界域精确仲裁 (块值 = 块内可达位置精确分 max; pin 块 +inf; 全不可达 -inf),
    官方 0.001 相对容差。参数含义同 _arb_sparse_rows, ga/gb 为 [n, cb] 块索引;
    gold 行尾有效槽 = 边界块 (_gold_cand_from_score 三路径均保证)。返回 bool [n]。"""
    n, k = ga.shape
    dev = ga.device
    gnum = r16_full.shape[1]
    big = col + 1
    va = ga >= 0
    vb = gb >= 0
    ok = va.sum(-1) == vb.sum(-1)
    sa, _ = torch.sort(torch.where(va, ga, big), dim=-1)
    sb, _ = torch.sort(torch.where(vb, gb, big), dim=-1)
    ok = ok & ~(
        (((sa[:, 1:] == sa[:, :-1]) & (sa[:, 1:] < big)).any(-1))
        | (((sb[:, 1:] == sb[:, :-1]) & (sb[:, 1:] < big)).any(-1))
    )
    posa = torch.searchsorted(sb, ga.clamp(min=0))
    in_b = va & (posa < k) & (sb.gather(1, posa.clamp(max=k - 1)) == ga)
    posb = torch.searchsorted(sa, gb.clamp(min=0))
    in_a = vb & (posb < k) & (sa.gather(1, posb.clamp(max=k - 1)) == gb)
    da = torch.where(va & ~in_b, ga, big)
    db = torch.where(vb & ~in_a, gb, big)
    nd = max(
        1, int(torch.maximum((da < big).sum(-1).max(), (db < big).sum(-1).max()).item())
    )
    dac, _ = torch.topk(da, nd, dim=-1, largest=False)
    dbc, _ = torch.topk(db, nd, dim=-1, largest=False)
    bidx = ga.gather(1, (va.sum(-1, keepdim=True) - 1).clamp(min=0))
    all_idx = torch.cat([dac, dbc, bidx], dim=1)  # [n, m] 块号
    m = all_idx.shape[1]
    # 块号 → 8 位置展开 → 一维扁平 take 精确位置分
    posn = all_idx.unsqueeze(-1) * CAND_BS + torch.arange(
        CAND_BS, device=dev
    )  # [n, m, 8]
    posc = posn.clamp(0, col - 1).reshape(n, m * CAND_BS)
    flat = r16_full.reshape(-1)
    rows = (js * gnum).unsqueeze(1) + torch.arange(gnum, device=dev).unsqueeze(0)
    fidx = rows.unsqueeze(2) * col + posc.unsqueeze(1)  # [n, g, 8m]
    colsg = flat[fidx]  # [n, g, 8m]
    pv = (
        torch.bmm(w_rows.to(torch.float32).unsqueeze(1), colsg.float()).squeeze(1)
        * ks_seg[posc]
    )  # [n, 8m] 精确位置分
    pv = pv.reshape(n, m, CAND_BS)
    vl_c = torch.minimum(vl_rows, torch.tensor(col, device=dev))
    validp = (all_idx.unsqueeze(-1) < big) & (posn < vl_c.view(n, 1, 1)) & (posn < col)
    blkv = torch.where(validp, pv, NEG_INF).amax(dim=-1)  # 块值 = 可达位置 max
    pin_blk = (vl_rows.clamp(min=0) - 1).div(CAND_BS, rounding_mode="floor")
    is_pin = (
        (all_idx == pin_blk.unsqueeze(1)) & (vl_rows > 0).unsqueeze(1) & (all_idx < big)
    )
    blkv = torch.where(is_pin, torch.full_like(blkv, float("inf")), blkv)
    bv = blkv[:, -1:]  # gold 边界块精确值
    vdiff = blkv[:, :-1]
    dmask = torch.cat([dac < big, dbc < big], dim=1)
    adiff = (vdiff - bv).abs()
    passm = torch.where(bv == 0, adiff == 0, adiff <= 0.001 * bv.abs()) | ~dmask
    return ok & passm.all(-1)


def _exact_gsum(w_rows, r16_rows, ks_seg):
    """r16 (与 golden 逐位一致) → fp32 精确 g-sum; 分行子批控制 [n,32,col] fp32 物化 ≤ ~1.5GB。
    FLOPs = qk 矩阵乘的 1/128, fp32 慢路径可承受; 精度 = CPU golden einsum (仅求和顺序 ulp 差)。"""
    n = w_rows.shape[0]
    col = r16_rows.shape[-1]
    sub = max(1, int(1.5e9 / (32 * max(col, 1) * 4)))
    outs = []
    for s0 in range(0, n, sub):
        s1e = min(s0 + sub, n)
        outs.append(
            torch.bmm(
                w_rows[s0:s1e].to(torch.float32).unsqueeze(1), r16_rows[s0:s1e].float()
            ).squeeze(1)
        )
    return torch.cat(outs, dim=0) * ks_seg


def _full_match_npu(inputs, case, idx_npu, cand_npu, cand_golden_npu):
    """大 shape 全量比对主流程, 返回 (fails, 仲裁行数, 耗时秒)"""
    t0 = time.time()
    b = case["batch"]
    g = case["g"]
    topk = case["topk"]
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res") or [0] * b
    mask_mode = case["mask_mode"]
    cand_mode = case["cand_mode"]
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * b
    seqused_k = case["seqused_k"]
    s2max = max(seqused_k)
    is_tnd = case.get("layout") == "TND"
    cu_q = inputs["cu_q"] if is_tnd else None
    offsets = inputs.get("output_idx_offsets") or [0] * sum(seqs_q)
    cb = case.get("cand_blocks", CAND_BLOCKS)
    dev = idx_npu.device
    idx_flat = idx_npu.reshape(-1, topk)
    cand_op = cand_npu.reshape(-1, cb) if cand_mode == 1 else None
    cand_in = None
    if cand_mode == 2:
        cand_in = cand_golden_npu.reshape(-1, cand_golden_npu.shape[-1]).to(dev)
    fails = []
    n_arb = 0
    row_base = 0
    for bi in range(b):
        s1 = seqs_q[bi]
        s2 = seqused_k[bi]
        act_k = s2 * ratio + res[bi]
        if mask_mode == 3:
            basev = act_k - s1 + torch.arange(s1) + 1
            vl = torch.div(basev, ratio, rounding_mode="floor")
            vl = torch.where(basev >= 0, vl, torch.full_like(vl, -1)).to(torch.int64)
        else:
            vl = torch.full((s1,), s2, dtype=torch.int64)
        if is_tnd:
            q_b = inputs["q"][cu_q[bi] : cu_q[bi + 1]]
            w_b = (
                inputs["w"][cu_q[bi] : cu_q[bi + 1]]
                * inputs["q_scale"][cu_q[bi] : cu_q[bi + 1]]
            )
        else:
            q_b = inputs["q"][bi]
            w_b = inputs["w"][bi] * inputs["q_scale"][bi]
        k16 = (
            inputs["k_dense"][bi, 0, :s2].to(torch.float16).npu()
        )  # [s2, D] int8→fp16 精确
        k32 = k16.to(torch.float32)  # int8 值域下等价 CPU int32
        ks = inputs["k_scale_dense"][bi, 0, :s2].to(torch.float32).npu()
        nb_total = (s2 + CAND_BS - 1) // CAND_BS
        # fp16 筛选值域保护: |qk| ≤ |q|max·|k|max·D 需 < 65504 (本 harness ±20 数据 = 51200)
        qmax = int(q_b.abs().max().item())
        kmax = int(inputs["k_dense"][bi, 0, :s2].abs().max().item())
        assert qmax * kmax * HEAD_DIM < 65504, (
            f"fp16 筛选值域超限 ({qmax}·{kmax}·{HEAD_DIM}={qmax * kmax * HEAD_DIM} ≥ 65504), 需改 fp32 路径"
        )
        vl_n = vl.npu()
        idx8 = (
            torch.arange(s2, device=dev) // CAND_BS
        )  # 位置→块号 (repeat_interleave 替代, NPU 慢路径)
        # 行块大小: 每行 ≈ (2g+8)·s2 字节 (qk/r16 fp16 原地链 + score fp32 + 掩码), 预算 ~2.5GB
        # (mode=1 还要容纳 cand 输出/比较张量, 预算过高会触发分配器抖动/长跑碎片累积 OOM)
        rows_per = max(1, min(4096, int(2.5e9 / ((2 * g + 8) * max(s2, 1)))))
        progress = max(1, s1 // 4)
        _prof = (
            {"mm": 0.0, "chain": 0.0, "det": 0.0, "cand": 0.0, "cmp": 0.0, "arb": 0.0}
            if os.environ.get("QLI_FM_PROF")
            else None
        )
        for r0 in range(0, s1, rows_per):
            r1 = min(r0 + rows_per, s1)
            c = r1 - r0
            vl_c = vl_n[r0:r1]
            col_lim = max(0, min(s2, int(vl_c.max().item())))
            if _prof:
                torch.npu.synchronize()
                _pt = time.time()
            if col_lim == 0:
                gi = torch.full((c, topk), -1, dtype=torch.int32, device=dev)
                gc = (
                    torch.full((c, cb), -1, dtype=torch.int32, device=dev)
                    if cand_mode == 1
                    else None
                )
            else:
                # ① fp16 筛选参考 (qk 单次输出舍入; /1024 为指数移位精确, 链路与 golden 一致)
                # 2D 展平: 3D×2D bmm 广播会把 B 按 batch 物化 (实测 20× 慢), 展平后走单次大 GEMM
                cRows = r1 - r0
                q2 = q_b[r0:r1].to(torch.float16).reshape(cRows * g, HEAD_DIM)
                # B 必须连续: 大 M (≥8192) 时转置视图走慢速 GEMM (实测 11 vs 116 TFLOPS)
                kb = k16[:col_lim].t().contiguous()
                qk = torch.matmul(q2, kb).reshape(cRows, g, col_lim)
                if _prof:
                    torch.npu.synchronize()
                    _prof["mm"] += time.time() - _pt
                    _pt = time.time()
                r16 = qk.div_(1024.0).clamp_min_(0.0)  # fp16 原地
                score = (
                    torch.bmm(w_b[r0:r1].unsqueeze(1), r16).squeeze(1).to(torch.float32)
                    * ks[:col_lim]
                )  # [c,col]
                colj = torch.arange(col_lim, device=dev)
                score = score.masked_fill(colj >= vl_c.unsqueeze(1), NEG_INF)
                if cand_mode == 2:
                    cc = cand_in[row_base + r0 : row_base + r1]
                    ids = cc.clamp(
                        0, nb_total - 1
                    ).long()  # 越界块号 clamp (变长 batch)
                    validb = (cc >= 0) & (cc < nb_total)
                    memb = torch.zeros(c, nb_total, dtype=torch.int32, device=dev)
                    memb.scatter_add_(
                        1, ids, validb.to(torch.int32)
                    )  # 重复 index 累加, 免竞态
                    pos_mask = memb[:, idx8[:col_lim]] > 0
                    score = torch.where(
                        pos_mask | (score == NEG_INF),
                        score,
                        torch.full_like(score, NEG_HUGE),
                    )
                if _prof:
                    torch.npu.synchronize()
                    _prof["chain"] += time.time() - _pt
                    _pt = time.time()
                gi = _gold_idx_det(score, topk)
                if _prof:
                    torch.npu.synchronize()
                    _prof["det"] += time.time() - _pt
                    _pt = time.time()
                off_rows = offsets[row_base + r0 : row_base + r1]
                if any(off_rows):
                    offt = torch.tensor(
                        off_rows, dtype=torch.int32, device=dev
                    ).unsqueeze(1)
                    gi = gi + offt  # 全元素加, 对齐 A15/golden 语义
                gc = None
                blocks_c = None
                if cand_mode == 1:
                    gc, blocks_c = _gold_cand_from_score(score, vl_c, cb, s2)
                if _prof:
                    torch.npu.synchronize()
                    _prof["cand"] += time.time() - _pt
                    _pt = time.time()
            # ② 多重集合筛选 (整行排序后相等) → 值域门 (边界并列等值交换, sound) → 边界域精确仲裁
            op = idx_flat[row_base + r0 : row_base + r1]
            eq = (torch.sort(gi, dim=-1).values == torch.sort(op, dim=-1).values).all(
                dim=-1
            )
            if not bool(eq.all()):
                eq = eq | _val_gate(score, gi, op)
            j_list = (~eq).nonzero(as_tuple=False).flatten().tolist()
            if _prof:
                torch.npu.synchronize()
                _prof["cmp"] += time.time() - _pt
                _pt = time.time()
            if j_list:
                n_arb += len(j_list)
                js = torch.tensor(j_list, dtype=torch.long, device=dev)
                grs = torch.tensor(
                    [row_base + r0 + j for j in j_list], dtype=torch.long, device=dev
                )
                if col_lim > 0 and not any(offsets[gr] for gr in grs.tolist()):
                    # 边界域精确仲裁 (官方规则等价): 差异元素+边界元素 r16 列 gather 精确分, 微秒级
                    memb_bad = None
                    if cand_mode == 2:
                        ccb = cand_in[grs]
                        idsb = ccb.clamp(
                            0, nb_total - 1
                        ).long()  # 越界块号 clamp (变长 batch)
                        vb = (ccb >= 0) & (ccb < nb_total)
                        memb_bad = torch.zeros(
                            len(j_list), nb_total, dtype=torch.bool, device=dev
                        )
                        memb_bad.scatter_(1, idsb, vb)
                    arb_ok = _arb_sparse_rows(
                        w_b[r0 + js],
                        r16,
                        js,
                        ks[:col_lim],
                        vl_n[r0 + js],
                        memb_bad,
                        gi[js],
                        idx_flat[grs],
                        col_lim,
                    )
                    res_list = (~arb_ok).nonzero(as_tuple=False).flatten().tolist()
                else:
                    res_list = list(range(len(j_list)))  # 全无效行/带偏移: 逐行走旧链
                for t in res_list:
                    gr = int(grs[t])
                    j = j_list[t]
                    if col_lim == 0:
                        sc_e = _npu_score_rows_exact(
                            q_b[r0 + j : r0 + j + 1],
                            w_b[r0 + j : r0 + j + 1],
                            k32,
                            ks,
                            vl_n[r0 + j : r0 + j + 1],
                            s2,
                            cand_in[gr : gr + 1] if cand_in is not None else None,
                            nb_total,
                        )
                        col_e = s2
                    else:
                        sc_e = _exact_gsum(
                            w_b[r0 + j : r0 + j + 1], r16[j : j + 1], ks[:col_lim]
                        )
                        colj_e = torch.arange(col_lim, device=dev)
                        sc_e = sc_e.masked_fill(
                            colj_e >= vl_n[r0 + j : r0 + j + 1].unsqueeze(1), NEG_INF
                        )
                        if cand_mode == 2:
                            ccb = cand_in[gr : gr + 1]
                            idsb = ccb.clamp(
                                0, nb_total - 1
                            ).long()  # 越界块号 clamp (变长 batch)
                            vb = (ccb >= 0) & (ccb < nb_total)
                            mb = torch.zeros(1, nb_total, dtype=torch.int32, device=dev)
                            mb.scatter_add_(
                                1, idsb, vb.to(torch.int32)
                            )  # 重复 index 累加, 免竞态
                            pmb = mb[:, idx8[:col_lim]] > 0
                            sc_e = torch.where(
                                pmb | (sc_e == NEG_INF),
                                sc_e,
                                torch.full_like(sc_e, NEG_HUGE),
                            )
                        col_e = col_lim
                    gi_e = _gold_idx_det(sc_e, topk)
                    if offsets[gr]:
                        gi_e = gi_e + offsets[gr]
                    op_bad = idx_flat[gr : gr + 1]
                    ok_r, msg_r = cmp_indices(
                        op_bad.cpu(),
                        gi_e.cpu(),
                        f"idx@r{gr}",
                        torch.nn.functional.pad(
                            sc_e, (0, s2max - sc_e.shape[-1]), value=NEG_INF
                        )
                        .reshape(1, 1, 1, -1)
                        .cpu()
                        .numpy(),
                        1,
                    )
                    if not ok_r:
                        fails.append(msg_r)
            if _prof:
                torch.npu.synchronize()
                _prof["arb"] += time.time() - _pt
            if cand_mode == 1:
                opc = cand_op[row_base + r0 : row_base + r1]
                eqc = (
                    torch.sort(gc, dim=-1).values == torch.sort(opc, dim=-1).values
                ).all(dim=-1)
                if not bool(eqc.all()):
                    eqc = eqc | _val_gate(blocks_c, gc, opc)
                jc_list = (~eqc).nonzero(as_tuple=False).flatten().tolist()
                if jc_list:
                    n_arb += len(jc_list)
                    jsc = torch.tensor(jc_list, dtype=torch.long, device=dev)
                    grsc = torch.tensor(
                        [row_base + r0 + j for j in jc_list],
                        dtype=torch.long,
                        device=dev,
                    )
                    if col_lim > 0 and not any(offsets[gr] for gr in grsc.tolist()):
                        # 边界域精确仲裁 (候选块): 差异块+边界块 8 位置 r16 扁平 take, 微秒级
                        carc_ok = _arb_cand_rows(
                            w_b[r0 + jsc],
                            r16,
                            jsc,
                            ks[:col_lim],
                            vl_n[r0 + jsc],
                            gc[jsc],
                            cand_op[grsc],
                            col_lim,
                        )
                        cres_list = (
                            (~carc_ok).nonzero(as_tuple=False).flatten().tolist()
                        )
                    else:
                        cres_list = list(range(len(jc_list)))
                    for t in cres_list:
                        gr = int(grsc[t])
                        j = jc_list[t]
                        if col_lim == 0:
                            sc_ec = _npu_score_rows_exact(
                                q_b[r0 + j : r0 + j + 1],
                                w_b[r0 + j : r0 + j + 1],
                                k32,
                                ks,
                                vl_n[r0 + j : r0 + j + 1],
                                s2,
                                None,
                                nb_total,
                            )
                        else:
                            sc_ec = _exact_gsum(
                                w_b[r0 + j : r0 + j + 1], r16[j : j + 1], ks[:col_lim]
                            )
                            colj_c = torch.arange(col_lim, device=dev)
                            sc_ec = sc_ec.masked_fill(
                                colj_c >= vl_n[r0 + j : r0 + j + 1].unsqueeze(1),
                                NEG_INF,
                            )
                        gc_e, blocks_e = _gold_cand_from_score(
                            sc_ec, vl_n[r0 + j : r0 + j + 1], cb, s2, det=True
                        )
                        v4b = (
                            torch.nn.functional.pad(
                                blocks_e[0],
                                (0, nb_total - blocks_e.shape[1]),
                                value=NEG_INF,
                            )
                            .reshape(1, 1, 1, -1)
                            .cpu()
                            .numpy()
                        )
                        ok_c, msg_c = cmp_indices(
                            cand_op[gr].cpu().unsqueeze(0),
                            gc_e.cpu(),
                            f"cand@r{gr}",
                            v4b,
                            1,
                        )
                        if not ok_c:
                            fails.append(msg_c)
            if r1 % progress < rows_per:
                print(
                    f"    [{case['name']}] b{bi} {r1}/{s1} 行 ({time.time() - t0:.0f}s)"
                )
                sys.stdout.flush()
        row_base += s1
    if _prof:
        print(
            "    [FM_PROF] mm=%.1fs chain=%.1fs det=%.1fs cand=%.1fs cmp=%.1fs arb=%.1fs"
            % (
                _prof["mm"],
                _prof["chain"],
                _prof["det"],
                _prof["cand"],
                _prof["cmp"],
                _prof["arb"],
            )
        )
    return fails, n_arb, time.time() - t0


def cmp_indices(npu_rows, gold_rows, name, values4d=None, s1=1):
    """官方比对规则 (对齐 tests/pytest/result_compare_method.py::check_result):
    1) 门限: 整行排序后多重集合相等 (值/-1/重复均敏感, 顺序不敏感) -> PASS
    2) 回退: compare_topk_valid — 有效前缀集合比较, 差异元素按边界值相对误差 <= thres(0.001) 容忍
    values4d: [B, 1, S1, score_size] 位置分 (mode=2 为 mask 后) / 块分矩阵, 供回退查分"""
    npu_m = (
        npu_rows.cpu().numpy() if torch.is_tensor(npu_rows) else np.asarray(npu_rows)
    )
    gold_m = (
        gold_rows.cpu().numpy() if torch.is_tensor(gold_rows) else np.asarray(gold_rows)
    )
    assert npu_m.shape == gold_m.shape, f"{name} shape {npu_m.shape} vs {gold_m.shape}"
    rows = npu_m.shape[0]
    n_multiset = 0
    n_boundary = 0
    fails = []
    for r in range(rows):
        g = np.asarray(gold_m[r], dtype=np.int64)
        n = np.asarray(npu_m[r], dtype=np.int64)
        if np.array_equal(np.sort(g), np.sort(n)):
            n_multiset += 1
            continue
        valid_len = int((g != -1).sum())
        npu_valid_len = int((n != -1).sum())
        if valid_len == 0:
            fails.append((r, "gold 全 -1 但 npu 有有效值"))
            continue
        if npu_valid_len != valid_len:
            # 官方 check_result 仅按 gold valid_len 切片, 会放过 -1 槽计数差异;
            # candidate 的 -1 槽是硬契约 (numBlocks < topk_blocks 时必须填 -1), 此处收紧
            fails.append(
                (r, f"-1 槽计数不匹配 gold_valid={valid_len} npu_valid={npu_valid_len}")
            )
            continue
        oob = bool((n < -1).any())
        if values4d is not None:
            oob = oob or bool((n >= values4d.shape[-1]).any())
        if oob:
            fails.append((r, "npu 越界索引"))
            continue
        b_idx, i_idx = r // s1, r % s1
        diff_npu, diff_cpu = [], []
        ok_row, max_re = compare_topk_valid(
            g[:valid_len],
            n[:valid_len],
            values4d,
            (b_idx, i_idx, 0),
            diff_npu,
            diff_cpu,
            None,
            None,
            thres=0.001,
            return_value_flag=False,
        )
        if ok_row:
            n_boundary += 1
        else:
            fails.append(
                (
                    r,
                    f"boundary max_re={max_re:.6f} npu_only={diff_npu[:4]} gold_only={diff_cpu[:4]}",
                )
            )
    ok = not fails
    msg = f"{name}: rows={rows} multiset_eq={n_multiset} boundary_ok={n_boundary} fail={len(fails)}"
    if fails:
        msg += " | " + "; ".join(f"r{r}:{info}" for r, info in fails[:3])
    return ok, msg


CASES = [
    # name, batch, q_seq, seqused_k, cmp_ratio, cmp_res, mask, cand_mode, g, topk, cand_blocks(可选, 默认2048)
    # 用户确认: 全部用例统一 q_head_num=32 (g=32, mBase=128)
    ("r1_m1_decode", 1, 1, [2048], 1, None, 3, 1, 32, 512),
    ("r1_m0_decode", 1, 1, [2048], 1, None, 0, 1, 32, 512),
    ("r1_m1_prefill_m3", 2, 8, [1024, 2048], 1, None, 3, 1, 32, 512),
    ("r1_m1_tail", 2, 4, [1000, 2000], 1, None, 3, 1, 32, 512),
    # R6 回归: prefill s1>1 的 mode=2 (每 AIV 2 行共核, candBuf 按行分区前该场景必错)
    ("r2_m1_decode", 1, 1, [1024], 2, [1], 3, 1, 32, 512),
    ("r2_m1_prefill_m3", 1, 16, [1024], 2, [1], 3, 1, 32, 512),
    ("r1_m1_g64_regress", 1, 1, [2048], 1, None, 0, 1, 64, 512),
    ("r1_m1_sparse2048", 1, 1, [4096], 1, None, 0, 1, 32, 2048),
    # pin 生效场景: candBlocks=64 < 总块数 512 (s2=4096, bs=8), 最新块必须被 pin 进候选
    # pin + 多 tile: s2=8192 (4 tiles, 1024 块), candBlocks=64
    # ---- 缺口补齐 (2026-09-07, 对照设计矩阵 §6.3) ----
    # B=4 变长 batch (矩阵要求 1/4)
    ("r1_m1_b4_varlen", 4, 4, [512, 1024, 1536, 2048], 1, None, 3, 1, 32, 512),
    # S1%4 != 0 prefill 尾块 (5/6/7: cuS1ProcNum 尾块路径)
    ("r1_m1_s1tail", 1, 6, [2048], 1, None, 3, 1, 32, 512),
    ("r1_m1_s1tail7", 2, 7, [1024, 2048], 1, None, 3, 1, 32, 512),
    # ratio=2 + mask=0 (无 residual, decode)
    ("r2_m1_decode_m0", 1, 1, [1024], 2, None, 0, 1, 32, 512),
    # 极小 S2: actS2Size=1 (单块单位置, 块数=1)
    ("r1_m1_tiny", 1, 1, [1], 1, None, 0, 1, 32, 512),
    # candBlocks=64 x mode=2 (掩码窗口 + topk_blocks<候选数的部分选)
    # R11 leak 回归 (v41 issue 场景门禁): prefill + candBlocks=64 (128 块缺 64) +
    # vl<=topk 的行 (0..511) → 输出必须完整 0..vl (候选只降级排序不取消资格);
    # 行 512+ 覆盖 regime 2: 候选外按索引升序泄漏填充
    # g=64 prefill 扩展 (此前仅 decode 一例)
    ("r1_m1_g64_prefill", 1, 8, [2048], 1, None, 3, 1, 64, 512),
]


def case_dict(t):
    if isinstance(t, dict):
        return t
    name, b, s1, sk, ratio, res, mask, mode, g, topk = t[:10]
    d = {
        "name": name,
        "batch": b,
        "q_seq": s1,
        "seqused_k": sk,
        "cmp_ratio": ratio,
        "cmp_res": res,
        "mask_mode": mask,
        "cand_mode": mode,
        "g": g,
        "topk": topk,
    }
    if len(t) > 10:
        d["cand_blocks"] = t[10]
    return d


# ---- 大 shape (16K/128K/1M) 直入 pytest (设计文档 §6.6 抽样比对机制) + TND/key非连续/offset 用例 ----
NEW_CASES = [
    # (id, case dict, xfail reason 或 None)
    # ---- 大 shape 全遍历 (§6.6): q_seq {16K,128K} x layout {BSND,TND} x cmp_ratio {1,2} ----
    #      (1M 组与 cand_blocks=64 用例 2026-10-08 按约定删除)
    # 生产规格: key pool [7936,128,1,128], block_table [1,1055]; 1M pool 默认 8192+4; ratio2 带 residual=1
    (
        "big16k_bsnd_r1",
        {
            "name": "big16k_bsnd_r1",
            "batch": 1,
            "q_seq": 16384,
            "seqused_k": [16384],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "big16k_bsnd_r2",
        {
            "name": "big16k_bsnd_r2",
            "batch": 1,
            "q_seq": 16384,
            "seqused_k": [16384],
            "cmp_ratio": 2,
            "cmp_res": [1],
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "big16k_tnd_r1",
        {
            "name": "big16k_tnd_r1",
            "batch": 1,
            "q_seq": 16384,
            "seqs_q": [16384],
            "seqused_k": [16384],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    (
        "big16k_tnd_r2",
        {
            "name": "big16k_tnd_r2",
            "batch": 1,
            "q_seq": 16384,
            "seqs_q": [16384],
            "seqused_k": [16384],
            "cmp_ratio": 2,
            "cmp_res": [1],
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    (
        "big128k_bsnd_r1",
        {
            "name": "big128k_bsnd_r1",
            "batch": 1,
            "q_seq": 131072,
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "big128k_bsnd_r2",
        {
            "name": "big128k_bsnd_r2",
            "batch": 1,
            "q_seq": 131072,
            "seqused_k": [131072],
            "cmp_ratio": 2,
            "cmp_res": [1],
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "big128k_tnd_r1",
        {
            "name": "big128k_tnd_r1",
            "batch": 1,
            "q_seq": 131072,
            "seqs_q": [131072],
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    (
        "big128k_tnd_r2",
        {
            "name": "big128k_tnd_r2",
            "batch": 1,
            "q_seq": 131072,
            "seqs_q": [131072],
            "seqused_k": [131072],
            "cmp_ratio": 2,
            "cmp_res": [1],
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    # 生产规格: key pool [7936,128,1,128], block_table [B,1055]; 1M pool 8192/表 8192
    # ---- 大 shape mode=2 遍历补缺 (2026-09-08): TND / ratio2 / 1M / 16K ----
    (
        "tnd_m1_decode",
        {
            "name": "tnd_m1_decode",
            "batch": 2,
            "q_seq": 1,
            "seqs_q": [1, 1],
            "seqused_k": [1024, 2048],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "layout": "TND",
        },
        None,
    ),
    (
        "tnd_m1_prefill",
        {
            "name": "tnd_m1_prefill",
            "batch": 2,
            "q_seq": 8,
            "seqs_q": [8, 4],
            "seqused_k": [1024, 2048],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "layout": "TND",
        },
        None,
    ),
    (
        "tnd_r2_m1",
        {
            "name": "tnd_r2_m1",
            "batch": 1,
            "q_seq": 1,
            "seqs_q": [1],
            "seqused_k": [1024],
            "cmp_ratio": 2,
            "cmp_res": [1],
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "layout": "TND",
        },
        None,
    ),
    # ---- key 0 轴非连续 (§11.2, A11 已实施 2026-09-08: attr 显式传 stride) ----
    # s1=1 + s2=128K decode 组合 (生产长上下文 decode): s2 半区并行 v5 门控命中形态,
    # 多 tile (131072/2048=64 tiles) + 大 topk 抽样比对
    (
        "dec_128k_m3",
        {
            "name": "dec_128k_m3",
            "batch": 1,
            "q_seq": 1,
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 0,
            "cand_mode": 3,
            "g": 32,
            "topk": 512,
            "pool": 2048,
            "table_w": 1024,
            "sample_rows": True,
            "sample_rand": 6,
        },
        None,
    ),
    (
        "dec_128k_m3_mask3",
        {
            "name": "dec_128k_m3_mask3",
            "batch": 1,
            "q_seq": 1,
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 3,
            "g": 32,
            "topk": 512,
            "pool": 2048,
            "table_w": 1024,
            "sample_rows": True,
            "sample_rand": 6,
        },
        None,
    ),
    (
        "dec_128k_m1",
        {
            "name": "dec_128k_m1",
            "batch": 1,
            "q_seq": 1,
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 2048,
            "table_w": 1024,
            "sample_rows": True,
            "sample_rand": 6,
        },
        None,
    ),
    (
        "pa_gap_m1_decode",
        {
            "name": "pa_gap_m1_decode",
            "batch": 1,
            "q_seq": 1,
            "seqused_k": [2048],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pa_gap": True,
        },
        None,
    ),
    (
        "pa_gap_128k",
        {
            "name": "pa_gap_128k",
            "batch": 1,
            "q_seq": 131072,
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 15872,
            "table_w": 1055,
            "pa_gap": True,
            "sample_rows": True,
        },
        None,
    ),
    (
        "pa_gap_16k_m1",
        {
            "name": "pa_gap_16k_m1",
            "batch": 1,
            "q_seq": 16384,
            "seqused_k": [16384],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 15872,
            "table_w": 1055,
            "pa_gap": True,
            "sample_rows": True,
        },
        None,
    ),
    # ---- B=2 变长大 shape (2026-09-08): batch 前缀/变长 vl/block_table 独立置换的大 shape 验证 ----
    (
        "big128k_b2_varlen",
        {
            "name": "big128k_b2_varlen",
            "batch": 2,
            "q_seq": 16384,
            "seqused_k": [98304, 131072],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 2048,
            "table_w": 1024,
            "sample_rows": True,
            "force_rows": [64, 73, 95, 16448, 16457, 16479],
        },
        None,
    ),
    (
        "big128k_b2_mask0_r2",
        {
            "name": "big128k_b2_mask0_r2",
            "batch": 2,
            "q_seq": 16384,
            "seqused_k": [98304, 131072],
            "cmp_ratio": 2,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 2048,
            "table_w": 1024,
            "sample_rows": True,
        },
        None,
    ),
    # ---- B=4 变长大 shape (2026-09-08): 4 batch 变长 vl + 非对齐尾 tile (98432) + R9 窗口行锚点 ----
    (
        "big128k_b4_varlen",
        {
            "name": "big128k_b4_varlen",
            "batch": 4,
            "q_seq": 16384,
            "seqused_k": [65536, 98432, 131072, 81920],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 4100,
            "table_w": 1024,
            "sample_rows": True,
            "force_rows": [
                64,
                73,
                95,
                18368,
                18377,
                18399,
                32832,
                32841,
                32863,
                49216,
                49225,
                49247,
            ],
        },
        None,
    ),
    (
        "big128k_b4_mask0_r2",
        {
            "name": "big128k_b4_mask0_r2",
            "batch": 4,
            "q_seq": 16384,
            "seqused_k": [65536, 98432, 131072, 81920],
            "cmp_ratio": 2,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 4100,
            "table_w": 1024,
            "sample_rows": True,
        },
        None,
    ),
    # ---- mask_mode=0 大 shape 全遍历 (2026-09-08): {16k,128k,1m} x {bsnd,tnd} x {m1,m2} x {r1,r2} ----
    (
        "mask0_16k_bsnd_m1_r1",
        {
            "name": "mask0_16k_bsnd_m1_r1",
            "batch": 1,
            "q_seq": 16384,
            "seqused_k": [16384],
            "cmp_ratio": 1,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "mask0_16k_bsnd_m1_r2",
        {
            "name": "mask0_16k_bsnd_m1_r2",
            "batch": 1,
            "q_seq": 16384,
            "seqused_k": [16384],
            "cmp_ratio": 2,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "mask0_16k_tnd_m1_r1",
        {
            "name": "mask0_16k_tnd_m1_r1",
            "batch": 1,
            "q_seq": 16384,
            "seqs_q": [16384],
            "seqused_k": [16384],
            "cmp_ratio": 1,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    (
        "mask0_16k_tnd_m1_r2",
        {
            "name": "mask0_16k_tnd_m1_r2",
            "batch": 1,
            "q_seq": 16384,
            "seqs_q": [16384],
            "seqused_k": [16384],
            "cmp_ratio": 2,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    (
        "mask0_128k_bsnd_m1_r1",
        {
            "name": "mask0_128k_bsnd_m1_r1",
            "batch": 1,
            "q_seq": 131072,
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "mask0_128k_bsnd_m1_r2",
        {
            "name": "mask0_128k_bsnd_m1_r2",
            "batch": 1,
            "q_seq": 131072,
            "seqused_k": [131072],
            "cmp_ratio": 2,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
        },
        None,
    ),
    (
        "mask0_128k_tnd_m1_r1",
        {
            "name": "mask0_128k_tnd_m1_r1",
            "batch": 1,
            "q_seq": 131072,
            "seqs_q": [131072],
            "seqused_k": [131072],
            "cmp_ratio": 1,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    (
        "mask0_128k_tnd_m1_r2",
        {
            "name": "mask0_128k_tnd_m1_r2",
            "batch": 1,
            "q_seq": 131072,
            "seqs_q": [131072],
            "seqused_k": [131072],
            "cmp_ratio": 2,
            "mask_mode": 0,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "pool": 7936,
            "table_w": 1055,
            "sample_rows": True,
            "sample_rand": 4,
            "layout": "TND",
        },
        None,
    ),
    # ---- output_idx_offset (§11.6, A15 未实施) ----
    (
        "off_m1_decode",
        {
            "name": "off_m1_decode",
            "batch": 1,
            "q_seq": 1,
            "seqused_k": [2048],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "output_idx_offset": [1000],
        },
        None,
    ),
    (
        "off_zero_regress",
        {
            "name": "off_zero_regress",
            "batch": 1,
            "q_seq": 4,
            "seqused_k": [2048],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "output_idx_offset": [0, 0, 0, 0],
        },
        None,
    ),  # 零偏移等价回归: 应与不传一致
    # R13 契约门禁: prefill (行级 vl < topk → 输出含 -1 槽) + 非零 offset →
    # 有效槽 = idx+offset, -1 槽必须保持 -1 (对齐模型 where(idxs < compress_lens, idxs+offset, -1);
    # 修复前 kernel/golden 全元素加 → -1 槽变 off-1)
    (
        "off_m1_prefill",
        {
            "name": "off_m1_prefill",
            "batch": 1,
            "q_seq": 128,
            "seqused_k": [1024],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "output_idx_offset": [700],
        },
        None,
    ),
    (
        "off_tnd_m1",
        {
            "name": "off_tnd_m1",
            "batch": 2,
            "q_seq": 1,
            "seqs_q": [1, 1],
            "seqused_k": [1024, 2048],
            "cmp_ratio": 1,
            "mask_mode": 3,
            "cand_mode": 1,
            "g": 32,
            "topk": 512,
            "layout": "TND",
            "output_idx_offset": "per_batch",
        },
        None,
    ),
]

ARGVALUES = [tuple(c) for c in CASES] + [
    pytest.param(d, id=nid, marks=[pytest.mark.xfail(reason=reason, strict=True)])
    if reason
    else pytest.param(d, id=nid)
    for nid, d, reason in NEW_CASES
]
ALL_IDS = [c[0] for c in CASES] + [nid for nid, _, _ in NEW_CASES]


def _sample_rows(case):
    """抽样行集合: 固定行 (首/等分/尾, 覆盖 vl 极小到极大) + 固定 seed 随机行"""
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * case["batch"]
    n_rand = case.get("sample_rand", 8)
    rows = set()
    for bi in range(case["batch"]):
        s1 = seqs_q[bi]
        base = sum(seqs_q[:bi])
        rows.update(
            {
                base,
                base + 1,
                base + 2,
                base + s1 // 3,
                base + s1 // 2,
                base + (2 * s1) // 3,
                base + s1 - 2,
                base + s1 - 1,
            }
        )
        rng = np.random.default_rng(case.get("seed", 7) + 100 + bi)
        rows.update((base + r % s1) for r in rng.integers(0, s1, n_rand))
    # R10 回归锚点: 确定性加入指定行 (大 shape 随机抽样可能漏掉 vl 尾 tile cuS2Len 在 (64,96] 的 stale 窗口行)
    rows.update(case.get("force_rows", ()))
    return sorted(r for r in rows if 0 <= r < sum(seqs_q))


@pytest.mark.parametrize("t", ARGVALUES, ids=ALL_IDS)
def test_qli_cand(t):
    # 跨用例显存清理: NPU caching allocator 碎片会随大 shape 用例累积 (实测 54 用例后
    # big128k_b2_tnd_m2 处 aclnnCat 507018 OOM, 其后 37 用例连锁失败) — 每用例开始归还空闲块
    torch.npu.empty_cache()
    case = case_dict(t)
    inputs = gen_case_inputs(case)
    cand_golden_npu, cand_golden = (None, None)
    if case["cand_mode"] == 2:
        kind = "self" if "self" in case["name"] else "rand"
        cand_golden_npu, cand_golden = gen_cand_input(case, kind, inputs)
    idx_npu, _, cand_npu = npu_run(inputs, case, cand_golden_npu)
    torch.npu.synchronize()
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * case["batch"]
    total_rows = sum(seqs_q)
    s2max = max(case["seqused_k"])
    topk = case["topk"]

    sample = _sample_rows(case) if case.get("sample_rows") else None
    if sample is None:
        # ---- 全量模式 (小 shape): 全行 golden + 官方两级比对 ----
        idx_golden, cand_golden_rows, score_rows, blk_rows = golden_run_ext(
            inputs, case, cand_golden
        )
        b = case["batch"]
        if case.get("layout") == "TND":
            # TND 变长行: 统一 [B,1,S1,S2] 无法对位 (boundary 回退的 b_idx/i_idx 除法失效), 逐行比对
            fails = []
            for r in range(sum(seqs_q)):
                v4 = score_rows[r].reshape(1, 1, 1, -1).numpy()
                ok_r, msg_r = cmp_indices(
                    idx_npu.reshape(-1, topk)[r].cpu().unsqueeze(0),
                    idx_golden.reshape(-1, topk)[r].unsqueeze(0),
                    f"idx@r{r}",
                    v4,
                    1,
                )
                if not ok_r:
                    fails.append(msg_r)
                if case["cand_mode"] == 1:
                    cb = case.get("cand_blocks", CAND_BLOCKS)
                    v4b = blk_rows[r].reshape(1, 1, 1, -1).numpy()
                    ok_c, msg_c = cmp_indices(
                        cand_npu.reshape(-1, cb)[r].cpu().unsqueeze(0),
                        cand_golden_rows[r].unsqueeze(0),
                        f"cand@r{r}",
                        v4b,
                        1,
                    )
                    if not ok_c:
                        fails.append(msg_c)
            assert not fails, f"[{case['name']}] TND 逐行比对失败: {fails[:3]}"
            print(f"[{case['name']}] TND 逐行 {sum(seqs_q)} 行全过 (官方两级规则)")
            if case["cand_mode"] == 1:
                cand_rows_for_pin = cand_golden_rows
            else:
                cand_rows_for_pin = None
            pin_rows = [
                (bi, i) for bi in range(case["batch"]) for i in range(seqs_q[bi])
            ]
            if case["cand_mode"] == 2:
                assert cand_npu.numel() == 0
            print(f"[{case['name']}] PASS")
            return
        # BSND 统一矩阵路径: values4d [B,1,S1,s2max] (每批行数统一 q_seq; TND 已走上方逐行分支)
        values4d_pos = (
            torch.stack(score_rows)
            .reshape(case["batch"], case["q_seq"], s2max)
            .unsqueeze(1)
            .numpy()
        )
        ok_idx, msg_idx = cmp_indices(
            idx_npu.reshape(-1, topk),
            idx_golden.reshape(-1, topk),
            "sparse_indices",
            values4d_pos,
            case["q_seq"],
        )
        print(msg_idx)
        assert ok_idx, f"[{case['name']}] {msg_idx}"
        if case["cand_mode"] == 1:
            cb = case.get("cand_blocks", CAND_BLOCKS)
            values4d_blk = (
                torch.stack(blk_rows)
                .reshape(case["batch"], case["q_seq"], -1)
                .unsqueeze(1)
                .numpy()
            )
            ok_cand, msg_cand = cmp_indices(
                cand_npu.reshape(-1, cb),
                torch.stack(cand_golden_rows),
                "candidate_topk_index",
                values4d_blk,
                case["q_seq"],
            )
            print(msg_cand)
            assert ok_cand, f"[{case['name']}] {msg_cand}"
        cand_rows_for_pin = cand_golden_rows
        pin_rows = [(bi, i) for bi in range(case["batch"]) for i in range(seqs_q[bi])]
    else:
        # ---- 全量比对模式 (大 shape, 2026-09-10 起): NPU 设备侧 golden 全行完全匹配 (取代 §6.6 抽样) ----
        nb_max = (s2max + CAND_BS - 1) // CAND_BS
        idx_all = idx_npu.reshape(-1, topk).cpu().to(torch.int64).numpy()
        oob = int(((idx_all < -1) | (idx_all >= s2max)).sum())
        assert oob == 0, f"[{case['name']}] 全行有效性: sparse_indices 越界 {oob}"
        if case["cand_mode"] == 1:
            cb = case.get("cand_blocks", CAND_BLOCKS)
            cand_all = cand_npu.reshape(-1, cb).cpu().to(torch.int64).numpy()
            oob_c = int(((cand_all < -1) | (cand_all >= nb_max)).sum())
            assert oob_c == 0, f"[{case['name']}] 全行有效性: candidate 越界 {oob_c}"
        fails, n_arb, el = _full_match_npu(
            inputs, case, idx_npu, cand_npu, cand_golden_npu
        )
        assert not fails, f"[{case['name']}] 全量比对失败: {fails[:3]}"
        print(
            f"[{case['name']}] 全量 {total_rows} 行完全匹配 "
            f"(NPU golden {el:.0f}s, fp16 筛选 + fp32 精确仲裁 {n_arb} 行)"
        )
        if case["cand_mode"] == 1:
            _pin_check_vec(case, cand_npu, seqs_q)
        pin_rows = []  # pin 已向量化检查, 跳过下方逐行循环
        cand_rows_for_pin = None

    if case["cand_mode"] == 1 and pin_rows:
        cb = case.get("cand_blocks", CAND_BLOCKS)
        cand_flat = cand_npu.reshape(-1, cb).cpu()
        for bi, i in pin_rows:
            act_k = (
                case["seqused_k"][bi] * case["cmp_ratio"]
                + (case.get("cmp_res") or [0] * case["batch"])[bi]
            )
            vl = (
                (act_k - seqs_q[bi] + i + 1) // case["cmp_ratio"]
                if case["mask_mode"] == 3
                else case["seqused_k"][bi]
            )
            last_blk = (min(vl, case["seqused_k"][bi]) - 1) // CAND_BS
            row_g = sum(seqs_q[:bi]) + i
            if last_blk < 0:
                continue
            assert last_blk in cand_flat[row_g].tolist(), (
                f"[{case['name']}] pin 失败: 行 {row_g} 块 {last_blk} 不在候选中"
            )
        print(f"[{case['name']}] pin OK")
    if case["cand_mode"] == 2:
        # 消费不改变候选输入, 校验透传无关性: 输出第三元为空
        assert cand_npu.numel() == 0
    print(f"[{case['name']}] PASS")


# =====================================================================================
# ---- P13: GetKeyScale 页跨度 int32 乘法溢出回归 (2026-09-18) ----
# 缺陷: k_scale 页跨度 (stride0=73856 half/块) 下, blockId * stride 的 int32 乘法在
# 物理块号 >= ceil(2^31/73856) = 29077 时溢出, 负偏移符号扩展越过张量基址 →
# 读到基址前内存 (静默错读, 不报错)。真机 vLLM 中该内存是 KV cache 其它映射 → 偶发乱码。
# 复现布局 (对齐部署观测): 显存 = [guard 毒值段][k_scale 池], 池经 storage_offset 置于
# guard 之后; block_table 把逻辑块映射到跨 29077 边界的物理块号 (29076~29079)。
# 修复 (P13): kScaleBlkStride 提升为 int64, 乘法前提升 (同型防御已覆盖
# vector GetKeyScale 的 blockTableBatchOffset 与 cube KeyNd2NzForPA 的表寻址)。
# 门禁语义: int64 修复后高页号 Top-K 必须与低页号 (同逻辑数据) 完全一致;
# int32 缺陷版高页号选中项分数下界显著劣于参考 (< 下界的 10%) — 本测试同时能抓回归。
# =====================================================================================

P13_STRIDE0 = 73856  # half 元素/块 (真实部署观测值)
P13_FIRST_OVERFLOW = 2147483647 // P13_STRIDE0 + 1  # 29077
P13_GUARD_HALF = P13_FIRST_OVERFLOW * P13_STRIDE0 + 4096  # guard 覆盖最大负偏移绝对值


def _p13_build(block_ids, with_guard, seed=7):
    """构造 P13 复现输入: guard 毒值段 + k_scale 池 (storage_offset 布局)。
    k/k_scale 逻辑数据与 harness 同风格; 返回 (npu 输入 dict, CPU 参考分数 [S2])"""
    B, S2, NBLK, BS, G = 1, 4096, 32, 128, 32
    gen = torch.Generator().manual_seed(seed)
    nphys = max(block_ids) + 1

    q = (
        torch.randint(-20, 21, (B, 1, G, HEAD_DIM), dtype=torch.int32, generator=gen)
        .to(torch.int8)
        .npu()
    )
    w = torch.empty(B, 1, G).uniform_(-1, 1, generator=gen).to(torch.float16).npu()
    q_scale = (
        torch.empty(B, 1, G).uniform_(0.5, 1.0, generator=gen).to(torch.float16).npu()
    )
    k_dense = torch.randint(
        -20, 21, (B, 1, S2, HEAD_DIM), dtype=torch.int32, generator=gen
    ).to(torch.int8)
    k_scale_dense = (
        torch.empty(B, 1, S2).uniform_(0.5, 1.0, generator=gen).to(torch.float16)
    )

    pool_elems = nphys * P13_STRIDE0
    total = (P13_GUARD_HALF + pool_elems) if with_guard else pool_elems
    flat = torch.full(
        (total,), float("nan"), dtype=torch.float16
    )  # 毒值: 错误读取必然污染
    for slot, bid in enumerate(block_ids[:NBLK]):  # 仅 NBLK 个逻辑块携带数据
        flat[
            (P13_GUARD_HALF if with_guard else 0) + bid * P13_STRIDE0 : (
                P13_GUARD_HALF if with_guard else 0
            )
            + bid * P13_STRIDE0
            + BS
        ] = k_scale_dense[0, 0, slot * BS : (slot + 1) * BS]
    flat_t = flat.npu()
    s_off = P13_GUARD_HALF if with_guard else 0
    k_scale_phys = torch.as_strided(
        flat_t, (nphys, BS, 1), (P13_STRIDE0, 1, 1), storage_offset=s_off
    )
    kphys = torch.zeros(nphys, BS, 1, HEAD_DIM, dtype=torch.int8)
    for slot, bid in enumerate(block_ids[:NBLK]):
        kphys[bid, :, 0, :] = k_dense[0, 0, slot * BS : (slot + 1) * BS, :]
    kphys = kphys.npu()
    bt = torch.zeros(B, nphys, dtype=torch.int32)
    bt[0, :NBLK] = torch.tensor(block_ids[:NBLK], dtype=torch.int32)
    bt = bt.npu()
    seqused_k = torch.tensor([S2] * B, dtype=torch.int32).npu()
    metadata = quant_lightning_indexer_metadata(
        num_heads_q=G,
        num_heads_k=1,
        head_dim=HEAD_DIM,
        topk=512,
        quant_mode=2,
        cu_seqlens_q=None,
        seqused_k=seqused_k,
        cmp_residual_k=None,
        batch_size=B,
        max_seqlen_q=1,
        max_seqlen_k=S2,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )

    # CPU 参考分数 (逻辑数据与物理布局无关; 真实 64 位寻址语义)
    qf = q[0, 0].cpu().to(torch.float32)
    wf = (w[0, 0].cpu() * q_scale[0, 0].cpu()).to(torch.float32)
    sc = torch.zeros(S2)
    for slot in range(NBLK):
        kb = k_dense[0, 0, slot * BS : (slot + 1) * BS, :].to(torch.float32)
        ksb = k_scale_dense[0, 0, slot * BS : (slot + 1) * BS].to(torch.float32)
        qk = qf @ kb.t()
        qk_relu = (qk / 1024.0).clamp_min(0.0).to(torch.float16).to(torch.float32)
        sc[slot * BS : (slot + 1) * BS] = (wf @ qk_relu) * ksb.unsqueeze(0)
    inputs = {
        "q": q,
        "w": w,
        "q_scale": q_scale,
        "k_phys": kphys,
        "k_scale_phys": k_scale_phys,
        "block_table": bt,
        "seqused_k": seqused_k,
        "metadata": metadata,
    }
    return inputs, sc


def _p13_run(inputs):
    # 现网路径（候选关闭）: 走基础入口; 本用例与候选无关, 仅考 k_scale 页跨度寻址
    ret = quant_lightning_indexer(
        inputs["q"],
        inputs["k_phys"],
        inputs["w"],
        inputs["q_scale"],
        inputs["k_scale_phys"],
        topk=512,
        quant_mode=2,
        cu_seqlens_q=None,
        seqused_k=inputs["seqused_k"],
        cmp_residual_k=None,
        block_table=inputs["block_table"],
        metadata=inputs["metadata"],
        max_seqlen_q=1,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
        return_value=0,
    )
    return ret[0]


def test_qli_p13_kscale_int32_overflow():
    """P13: k_scale 页跨度 int32 溢出 — 高物理块号 (跨 29077) Top-K 必须与低页号一致。

    int32 缺陷版特征: 高页号组选中项分数下界 < 参考下界的 10% (毒值 NaN/垃圾入选);
    修复版必须与低页号组一致 (官方比对语义, 此处用下界等价判定 + 集合位等价双保险)。"""
    torch.npu.empty_cache()
    low_ids = list(range(1, 33))  # 1..32 全低页号
    first = P13_FIRST_OVERFLOW  # 29077
    high_ids = [first - 1, first, first + 1, first + 2] + list(range(5, 37))

    ins_l, sc_ref = _p13_build(low_ids, with_guard=False)
    idx_l = _p13_run(ins_l).reshape(-1).cpu()
    ref_min = torch.topk(sc_ref.flatten(), 512).values.min().item()
    # 低页号组: 正确实现下选中项最低分 = 参考下界
    low_min = sc_ref[idx_l.to(torch.long)].min().item()
    assert abs(low_min - ref_min) < 1e-3 * max(1.0, abs(ref_min)), (
        f"低页号组异常 (基线能力): {low_min} vs 参考 {ref_min}"
    )
    del ins_l
    torch.npu.empty_cache()

    ins_h, _ = _p13_build(high_ids, with_guard=True)  # 同逻辑数据, 高物理页号
    idx_h = _p13_run(ins_h).reshape(-1).cpu()
    high_min = sc_ref[idx_h.to(torch.long)].min().item()
    assert abs(high_min - ref_min) < 1e-3 * max(1.0, abs(ref_min)), (
        f"P13 溢出回归: 高物理块号 (>= {first}) Top-K 分数下界 {high_min} 严重偏离参考 {ref_min} "
        f"(int32 页跨度乘法溢出, 读到 guard 毒值); 确认 kScaleBlkStride 是否为 int64 修复版"
    )

    # 双保险: 高页号组 Top-K 集合与低页号组位等价 (同逻辑数据的确定性输出)
    assert torch.equal(idx_l, idx_h), (
        f"P13 溢出回归: 高/低页号 Top-K 集合不一致 (int32 寻址差异); "
        f"首异处 = {(idx_l != idx_h).nonzero()[0].item()}"
    )
    print(
        f"[p13_kscale_int32_overflow] PASS: 低/高页号 (跨 {first}) Top-K 位等价, "
        f"分数下界 {low_min:.6f} == 参考 {ref_min:.6f}"
    )
    del ins_h
    torch.npu.empty_cache()
