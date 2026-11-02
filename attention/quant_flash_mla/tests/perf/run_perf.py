#!/usr/bin/env python3
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


"""quant_flash_mla(全量化 MLA FA)NPU 性能测试入口。

逐 case 独立 msprof 进程(one-by-one), 规避批跑下 profiler 条目归属歧义。
链路: perf_cases(用例) -> perf_codegen(生成脚本) -> perf_profiler(msprof 采集)
      -> perf_parser(过滤 OP Type=QuantFlashMla, 热启动均值) -> MFU 估算 -> perf.csv

用法:
    conda activate fzj
    source /home/fanzijian/Ascend1008/cann-9.3.0/set_env.sh    # 基础 CANN 包(按实际安装路径)
    source /home/fanzijian/custom/vendors/custom_transformer/bin/set_env.bash  # custom 算子包
    python3 run_perf.py --runs 5
    python3 run_perf.py --runs 1                      # 只测一次(含首次运行开销)
    python3 run_perf.py --runs 5 --discard-first      # 丢弃预热后取平稳均值
    python3 run_perf.py --runs 5 --case-filter Decode,PA_NZ
    python3 run_perf.py --runs 5 --case-filter Decode,PA_NZ --pipe-timeline   # 性能 + pipe 时间线
    python3 run_perf.py --list
"""

import argparse
import csv
import os
import time
from pathlib import Path

# 隔离 torch JIT 扩展编译缓存: 共享的 ~/.cache/torch_extensions 可能被其它
# 环境(如 base conda 的旧版 cann_ops_transformer)污染, 导致 wrapper 与编译出的
# .so 参数签名不一致(metadata 13参 vs 14参 TypeError)。独立目录保证 .so 始终
# 从当前环境(fzj)的 csrc 编译。
HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".torch_ext"))

from perf_cases import load_cases
from perf_codegen import D_V, build_script
from perf_profiler import run_msprof, run_msopprof_pipe
from perf_parser import parse_op_summary, compute_hot_avg

# MFU 相关常量
QK_HEAD_DIM = 576  # MLA: nope 512 + rope 64
V_HEAD_DIM = D_V  # MLA: 512, 与 perf_codegen 保持同一来源
FP8_CUBE_FLOPS_PER_CYCLE = 2 * 16 * 16 * 32
FP8_FREQ_GHZ = 1.6
DEFAULT_CORE_NUM = 32


def get_core_num():
    """设备 cube 核数(纯 CPU 环境回退到 32)。"""
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401

        return int(torch.npu.get_device_properties(0).cube_core_num)
    except Exception:
        return DEFAULT_CORE_NUM


def fp8_peak_tflops(core_num):
    return FP8_CUBE_FLOPS_PER_CYCLE * FP8_FREQ_GHZ * core_num * 0.001


def calc_compute_num(case):
    """按 batch 逐条累加计算量, QK 与 PV 的 head dim 分开计。

    sp0:            2*kvs*qs*(qkd+vd)*qn
    sp3 (qs<kvs):   2*(kvs-qs+1+kvs)*qs/2*(qkd+vd)*qn
    sp3 (qs>=kvs):  2*kvs*kvs/2*(qkd+vd)*qn
    """
    qn = case["N_q"]
    is_causal = case["sparse_mode"] == 3
    head_dim_sum = QK_HEAD_DIM + V_HEAD_DIM
    total = 0.0
    for qs, kvs in zip(case["seqused_q"], case["cache_seqlens"]):
        if is_causal:
            valid = (kvs - qs + 1 + kvs) * qs / 2 if qs < kvs else kvs * kvs / 2
        else:
            valid = kvs * qs
        total += 2 * valid * head_dim_sum * qn
    return total


def calc_mfu(case, duration_us, peak_tflops):
    """返回 MFU 百分比; duration 无效时返回 None。"""
    if duration_us <= 0 or peak_tflops <= 0:
        return None
    flops = calc_compute_num(case) / (duration_us * 1e-6)
    return flops * 1e-12 / peak_tflops * 100


def main():
    ap = argparse.ArgumentParser(description="QuantFlashMla perf benchmark")
    ap.add_argument(
        "--runs",
        type=int,
        default=5,
        help="每 case 迭代次数(>=1); 默认对全部条目取均值",
    )
    ap.add_argument(
        "--discard-first",
        action="store_true",
        help="丢弃第一条(预热)后取均值, 需 --runs >= 2",
    )
    ap.add_argument(
        "--case-filter", type=str, default=None, help="逗号分隔的用例名子串过滤"
    )
    ap.add_argument(
        "--output", type=str, default=None, help="输出目录(默认 results/<时间戳>)"
    )
    ap.add_argument(
        "--pipe-timeline",
        action="store_true",
        help="对 --case-filter 命中的全部用例额外采集 PipeTimeline"
        "(产物 visualize_data.bin 用 MindStudio Insight 打开); "
        "如需缩小范围请用 --case-filter 过滤",
    )
    ap.add_argument("--list", action="store_true", help="只列出用例名")
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs 需 >= 1")
    if args.discard_first and args.runs < 2:
        ap.error("--discard-first 需配合 --runs >= 2")

    cases = load_cases()
    if args.case_filter:
        filters = [f.strip() for f in args.case_filter.split(",") if f.strip()]
        cases = {k: v for k, v in cases.items() if any(f in k for f in filters)}
        print(f"[info] Filtered to {len(cases)} cases")
    if not cases:
        print("[error] no cases matched")
        return 2

    if args.list:
        for name in cases:
            print(name)
        return 0

    print(
        f"[info] Loaded {len(cases)} cases, runs={args.runs}, "
        f"averaging={'discard 1st' if args.discard_first else 'keep all'}"
    )
    if args.pipe_timeline:
        print("[info] PipeTimeline enabled for all filtered cases")
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else HERE / "results" / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output -> {output_dir}")
    core_num = get_core_num()
    peak_tflops = fp8_peak_tflops(core_num)
    print(f"[info] cube_core_num={core_num}, fp8_peak={peak_tflops:.2f} TFLOPS")

    results = {}  # name -> (avg_us, mfu_percent, samples, status, message)
    pipe_bins = {}  # name -> visualize_data.bin 路径
    total = len(cases)
    for idx, (name, case) in enumerate(cases.items()):
        print(f"\n{'=' * 70}")
        print(f"[{idx + 1}/{total}] [HOT] {name}")
        print(f"{'=' * 70}")
        case_dir = output_dir / name
        try:
            script = build_script(name, case, args.runs)
            run_msprof(script, case_dir, label=name)
            entries = parse_op_summary(case_dir / "op_summary.csv")
            durations = [d for _, d in entries]
            if len(entries) < args.runs:
                message = (
                    f"captured {len(entries)}/{args.runs} entries, "
                    f"check {case_dir / name}.msprof.log"
                )
                print(f"  => FAILED: {message}")
                results[name] = (-1.0, None, len(entries), "MISSING", message)
                continue
            avg = compute_hot_avg(durations, args.discard_first)
            samples = len(durations) - (1 if args.discard_first else 0)
            mfu = calc_mfu(case, avg, peak_tflops)
            results[name] = (avg, mfu, samples, "PASS", "")
            mode = "discard 1st" if args.discard_first else "keep all"
            print(f"  => {avg:.2f} us  MFU={mfu:.2f}%  ({samples} samples, {mode})")
            if args.pipe_timeline:
                opprof_dir = run_msopprof_pipe(script, case_dir, label=name)
                vis = list(opprof_dir.glob("visualize_data.bin")) or list(
                    opprof_dir.rglob("visualize_data.bin")
                )
                if vis:
                    pipe_bins[name] = str(vis[0])
                    print(f"  [pipe] {vis[0]}")
                else:
                    print(f"  [pipe] WARNING: no visualize_data.bin in {opprof_dir}")
        except Exception as e:
            print(f"  => FAILED: {e}")
            results[name] = (-1.0, None, 0, "ERROR", str(e))

    perf_csv = output_dir / "perf.csv"
    with open(perf_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_name",
                "duration_us",
                "mfu",
                "samples",
                "status",
                "message",
                "pipe_visualize_bin",
            ]
        )
        for name, (avg, mfu, samples, status, message) in results.items():
            writer.writerow(
                [
                    name,
                    f"{avg:.2f}" if avg > 0 else "",
                    f"{mfu:.2f}" if mfu is not None else "",
                    samples,
                    status,
                    message,
                    pipe_bins.get(name, ""),
                ]
            )

    n_failed = sum(1 for r in results.values() if r[3] != "PASS")
    print(f"\n{'=' * 70}")
    print(f"[SUMMARY] {len(results)} cases, {n_failed} failed, results -> {perf_csv}")
    print(f"{'=' * 70}")
    for name, (avg, mfu, _samples, status, _msg) in results.items():
        shown = f"{avg:.2f} us" if avg > 0 else f"FAILED({status})"
        mfu_shown = f"{mfu:.2f}%" if mfu is not None else "-"
        print(f"  [HOT] {name:<64s} {shown:>12s}  MFU={mfu_shown}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
