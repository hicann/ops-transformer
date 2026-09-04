#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Batch-invariance test driver for aclnnQuantMatmulReduceScatterV2.

Literature BI definition (Thinking Machines, 2026):
    y[i] = f(x[i], W)   即 row i 的输出只取决于输入 row i + 共享权重 W

方法学（关键，详见 README.md §2 Methodology Note）：
    每个 cell 三步走，缺一不可：
      1. **HCCL warm-up**：worker 启动后先做 N_WARMUP 次 dummy 调用，稳定 HCCL state
      2. **Multi-trial determinism baseline**：同 input N_TRIALS 次，所有 SHA 必须一致
         （catch trial 非确定性 — 否则 A vs B 不一致无法判别是 BI bug 还是噪声）
      3. **A vs B BI check**：只有 baseline 过了，才比较 variant A 和 B 的 SHA
         同 → BI_PASS；不同 → BI_FAIL；任一 baseline 不一致 → NON_DETERMINISTIC

该测试为 issue #2956 参考实现。
"""

import hashlib
import os
import sys

import torch
import torch.distributed as dist
import torch_npu

BLOCK_SIZE = 32  # MX 量化的 K-axis block size
N_WARMUP = 2  # warm-up 次数（稳定 HCCL state）
N_TRIALS = 3  # determinism baseline 每个 variant 的 trial 数


def sha_row(t, row_idx):
    """计算 tensor 第 row_idx 行的 SHA256（前 16 hex 字符）。"""
    return hashlib.sha256(
        t[row_idx].cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def build_x2_scale_k_varying(n_pack, N):
    """构造 K-axis varying 的 E8M0 scale（已知的 BI 触发条件）。"""
    bs = torch.empty(n_pack, N, 2, dtype=torch.uint8)
    for p in range(n_pack):
        for c in range(N):
            for slot in (0, 1):
                bs[p, c, slot] = 0x78 + ((p * 2 + slot) % 8)
    return bs


def build_x1_scale_varying(M_total, n_pack):
    """构造 M×K-block varying 的 E8M0 x1_scale（额外 BI 触发条件覆盖）。"""
    bs = torch.empty(M_total, n_pack, 2, dtype=torch.uint8)
    for m in range(M_total):
        for p in range(n_pack):
            for slot in (0, 1):
                bs[m, p, slot] = 0x78 + ((m + p * 2 + slot) % 8)
    return bs


def make_and_call(
    dev, hcom, ws, M_total, K, N, target_row, seed, x1_scale_varying=False
):
    """构造 input + 调用 op，返回输出 tensor。

    x1_scale_varying=False (默认): x1_scale 恒为 1.0 (0x7F)，仅 x2_scale 沿 K 轴 varying
    x1_scale_varying=True:        x1_scale 沿 M+K 轴 varying，覆盖 x1_scale 维度 BI
    """
    # MX 每 64 个 K 元素打包成 1 个 n_pack（每 pack 含 2 个 slot，每 slot 覆盖 BLOCK_SIZE=32 个元素）
    n_pack = (K + BLOCK_SIZE * 2 - 1) // (BLOCK_SIZE * 2)

    g_t = torch.Generator()
    g_t.manual_seed(12345)
    target_row_bytes = torch.randint(0x10, 0x50, (K,), dtype=torch.uint8, generator=g_t)

    g_w = torch.Generator()
    g_w.manual_seed(9000)
    x2_bytes = torch.randint(0x10, 0x50, (K, N), dtype=torch.uint8, generator=g_w)
    x2 = x2_bytes.to(dev).reshape(K, N).view(torch.float8_e4m3fn)

    x2s_bytes = build_x2_scale_k_varying(n_pack, N)
    x2s = x2s_bytes.to(dev).reshape(n_pack, N, 2).view(torch.float8_e8m0fnu)

    g_o = torch.Generator()
    g_o.manual_seed(seed)
    x1_bytes = torch.randint(0x10, 0x50, (M_total, K), dtype=torch.uint8, generator=g_o)
    x1_bytes[target_row] = target_row_bytes  # 固定 target row 的内容
    if x1_scale_varying:
        x1s_bytes = build_x1_scale_varying(M_total, n_pack)
    else:
        x1s_bytes = torch.full((M_total, n_pack, 2), 0x7F, dtype=torch.uint8)
    x1 = x1_bytes.to(dev).reshape(M_total, K).view(torch.float8_e4m3fn)
    x1s = x1s_bytes.to(dev).reshape(M_total, n_pack, 2).view(torch.float8_e8m0fnu)

    out = torch.ops.npu.npu_quant_mm_reduce_scatter(
        x1,
        x2,
        hcom,
        ws,
        reduce_op="sum",
        bias=None,
        x1_scale=x1s,
        x2_scale=x2s,
        quant_scale=None,
        block_size=BLOCK_SIZE,
        comm_turn=0,
        group_sizes=None,
        amax_output=False,
        y_dtype=torch_npu.bfloat16,
        x1_dtype=torch_npu.float8_e4m3fn,
        x2_dtype=torch_npu.float8_e4m3fn,
        x1_scale_dtype=torch_npu.float8_e8m0fnu,
        x2_scale_dtype=torch_npu.float8_e8m0fnu,
    )
    torch_npu.npu.synchronize()
    return out[0] if isinstance(out, tuple) else out


def rigorous_bi_cell(
    dev, hcom, ws, M_total, K, N, target_row, seed_a, seed_b, x1_scale_varying=False
):
    """三态判定 BI cell。

    返回 (status, sha_a, sha_b, shas_a_all, shas_b_all)
    status ∈ {BI_PASS, BI_FAIL, NON_DET, ERROR, SKIP}
    """
    shas_a = []
    shas_b = []
    try:
        # Phase 1: determinism baseline for variant A (same input seed_a, N_TRIALS times)
        for _ in range(N_TRIALS):
            y = make_and_call(
                dev, hcom, ws, M_total, K, N, target_row, seed_a, x1_scale_varying
            )
            if target_row >= y.shape[0]:
                return ("SKIP", None, None, shas_a, shas_b)
            shas_a.append(sha_row(y, target_row))
        # Phase 2: determinism baseline for variant B
        for _ in range(N_TRIALS):
            y = make_and_call(
                dev, hcom, ws, M_total, K, N, target_row, seed_b, x1_scale_varying
            )
            shas_b.append(sha_row(y, target_row))
    except Exception:
        return ("ERROR", None, None, shas_a, shas_b)

    det_a = len(set(shas_a)) == 1
    det_b = len(set(shas_b)) == 1
    if not det_a or not det_b:
        return (
            "NON_DET",
            shas_a[0] if shas_a else None,
            shas_b[0] if shas_b else None,
            shas_a,
            shas_b,
        )
    sha_a, sha_b = shas_a[0], shas_b[0]
    return ("BI_PASS" if sha_a == sha_b else "BI_FAIL", sha_a, sha_b, shas_a, shas_b)


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dev = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl", world_size=ws, rank=rank)
    hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(
        local_rank
    )

    if rank == 0:
        print(
            f"=== matmul_reduce_scatter_v2 BI rigorous WS={ws} N_WARMUP={N_WARMUP} N_TRIALS={N_TRIALS} ===",
            flush=True,
        )

    # === Phase 0: HCCL warm-up (critical — first worker cold start often non-deterministic) ===
    if rank == 0:
        print("  Phase 0: HCCL warm-up...", flush=True)
    try:
        for _ in range(N_WARMUP):
            _ = make_and_call(dev, hcom, ws, 16, 256, 64, 0, 42)
    except Exception as e:
        if rank == 0:
            print(f"    warmup ERROR (continuing): {str(e)[:100]}", flush=True)

    # === BI matrix ===
    M_TOTALS = [8, 16, 32]
    KS = [256, 512, 1024]
    NS = [64, 128]
    TARGET_ROW = 0
    # x1_scale varying coverage: cell index 满足 % 4 == 0 时启用 x1_scale 沿 M+K 轴 varying
    # （在 18 cell 矩阵里随机覆盖 ~4-5 个 cell，补强 x1_scale 维度 BI 检测）
    X1_VARYING_MOD = 4

    n_pass = n_fail = n_nondet = n_err = n_skip = 0
    fail_examples = []
    nondet_examples = []

    cell_idx = 0
    for M_total in M_TOTALS:
        if M_total % ws != 0:
            continue
        for K in KS:
            for N in NS:
                # 不同 cell 用不同 seed 对 — 不同数据分布触发不同 cube/HCCL 路径
                shape_hash = hash((M_total, K, N)) % 10000
                seed_a = 100 + shape_hash
                seed_b = 200 + shape_hash
                x1_var = cell_idx % X1_VARYING_MOD == 0
                cell_idx += 1
                status, sa, sb, sas, sbs = rigorous_bi_cell(
                    dev,
                    hcom,
                    ws,
                    M_total,
                    K,
                    N,
                    TARGET_ROW,
                    seed_a=seed_a,
                    seed_b=seed_b,
                    x1_scale_varying=x1_var,
                )
                if rank == 0:
                    if status == "BI_PASS":
                        n_pass += 1
                    elif status == "BI_FAIL":
                        n_fail += 1
                        fail_examples.append((M_total, K, N, sa, sb))
                    elif status == "NON_DET":
                        n_nondet += 1
                        nondet_examples.append((M_total, K, N, sas, sbs))
                    elif status == "ERROR":
                        n_err += 1
                    elif status == "SKIP":
                        n_skip += 1

    if rank == 0:
        print(
            f"  result: BI_PASS={n_pass} BI_FAIL={n_fail} NON_DET={n_nondet} ERROR={n_err} SKIP={n_skip}",
            flush=True,
        )
        if fail_examples:
            print("  BI_FAIL (real BI 违例 — deterministic but A != B):", flush=True)
            for ex in fail_examples[:5]:
                print(
                    f"    M_total={ex[0]} K={ex[1]} N={ex[2]} A_sha={ex[3]} B_sha={ex[4]}",
                    flush=True,
                )
        if nondet_examples:
            print("  NON_DET (同 input 跨 trial 不一致 — 无法判别 BI):", flush=True)
            for ex in nondet_examples[:5]:
                print(
                    f"    M_total={ex[0]} K={ex[1]} N={ex[2]} A_trials={ex[3]} B_trials={ex[4]}",
                    flush=True,
                )

    dist.destroy_process_group()
    # 仅 BI_FAIL 算 failure；NON_DET 是「需进一步排查 HCCL state / 硬件因素」的信号
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
