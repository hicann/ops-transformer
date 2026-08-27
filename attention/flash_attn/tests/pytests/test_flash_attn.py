#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import sys
import os
import gc
import logging
import argparse
from pathlib import Path

# ── 新架构 modules ──
from core.data import flash_attn_inputs
from core.case_loader import load_case_modules, normalize_params, resolve_case_ids
from core.orchestrator import run_case, run_case_load
from core.reporter import Reporter

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_DEFAULT_CASE_FILES = [
    os.path.splitext(f)[0]
    for f in os.listdir(os.path.join(os.path.dirname(__file__), "test_cases"))
    if f.endswith(".py") and not f.startswith("_")
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlashAttn precision test (new architecture)"
    )
    parser.add_argument("--case_files", type=str, default=",".join(_DEFAULT_CASE_FILES))
    parser.add_argument("--case_id", type=str, default="all")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument(
        "--verbose_diff",
        action="store_true",
        help="Print all failed elements when precision fails (default: first 10 only)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate heatmap PNG when precision fails",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="./viz_output",
        help="heatmap/analysis report output directory",
    )
    parser.add_argument(
        "--fail_analysis",
        action="store_true",
        help="Output multi-dimension fail distribution analysis report when precision fails",
    )
    parser.add_argument(
        "--save_golden",
        type=str,
        default=None,
        help="Normal comparison flow + save inputs/golden to specified directory",
    )
    parser.add_argument(
        "--load_golden",
        type=str,
        default=None,
        help="Load golden from specified directory, run NPU/GPU comparison only",
    )

    parser.add_argument("--meta_only", action="store_true")
    parser.add_argument("--load_gpu_dump", type=str, default=None)
    parser.add_argument("--load_npu_dump", type=str, default=None)
    parser.add_argument("--compare_mode", action="store_true")
    parser.add_argument("--graph_mode", action="store_true")
    parser.add_argument("--report_interval", type=int, default=20)
    parser.add_argument("--perf_mode", action="store_true")
    parser.add_argument("--perf_runs", type=int, default=5)
    parser.add_argument("--perf_cold_thr", type=int, default=16)
    parser.add_argument("--perf_output", type=str, default="./perf_output")
    parser.add_argument("--skip_accuracy", action="store_true")
    parser.add_argument(
        "--result_csv",
        type=str,
        default="attention/flash_attn/tests/pytests/result.csv",
        help="precision result CSV output path",
    )
    parser.add_argument(
        "--one_by_one",
        action="store_true",
        help="run cases one by one, pause after each",
    )
    args = parser.parse_args()

    if args.save_golden and args.load_golden:
        logger.error("[ERROR] --save_golden and --load_golden cannot be used together")
        sys.exit(1)

    if args.compare_mode and not args.load_golden:
        logger.error(
            "[ERROR] --compare_mode requires --load_golden to specify GPU result directory"
        )
        sys.exit(1)

    # ── 加载用例 ──
    module_names = ["test_cases." + x.strip() for x in args.case_files.split(",")]
    all_cases = load_case_modules(module_names)
    run_cases = resolve_case_ids(args.case_id, all_cases)
    if not run_cases:
        logger.error("[ERROR] No runnable cases.")
        sys.exit(1)

    # ── 性能测试分支 ──
    if args.perf_mode:
        from utils.perf_parser import parse_op_summary, compute_stats

        cases_list = [normalize_params(all_cases[name]) for name in run_cases]
        out = Path(args.perf_output)

        if args.one_by_one:
            from utils.perf_runner import run_all_perf

            csv_path = run_all_perf(
                cases_list,
                args.perf_runs,
                args.perf_cold_thr,
                out,
                case_names=run_cases,
                one_by_one=True,
            )
        else:
            from utils.perf_runner import build_batch_script, run_batch_msprof

            script = build_batch_script(
                cases_list,
                args.perf_runs,
                args.perf_cold_thr,
                case_names=run_cases,
                output_dir=str(out.resolve()),
            )
            csv_path = run_batch_msprof(script, out)

        if csv_path is None or not csv_path.exists():
            logger.error("[ERROR] msprof collection failed")
            sys.exit(1)

        from utils.perf_parser import _read_perf_log
        import csv as csv_mod

        with open(str(csv_path)) as f:
            all_rows = list(csv_mod.DictReader(f))
        flash_rows = [r for r in all_rows if r.get("OP Type") == "FlashAttn"]
        fieldnames = (
            list(all_rows[0].keys())
            if all_rows
            else ["Task Start Time(us)", "Task Duration(us)"]
        )

        with open(str(csv_path), "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames)
            w.writeheader()
            for row in flash_rows:
                w.writerow(row)

        perf_log = _read_perf_log(str(out))
        log_path = out / "_perf_log.jsonl"
        logger.info(
            f"  [perf] log: {log_path} exists={log_path.exists()} entries={len(perf_log)}"
        )

        hot_idxs = [
            i
            for i, c in enumerate(cases_list)
            if c.get("S1", 9999) > args.perf_cold_thr
        ]
        cold_idxs = [
            i
            for i, c in enumerate(cases_list)
            if c.get("S1", 9999) <= args.perf_cold_thr
        ]
        ordered = [(i, "hot") for i in hot_idxs] + [(i, "cold") for i in cold_idxs]

        fieldnames = (
            list(all_rows[0].keys())
            if all_rows
            else ["Task Start Time(us)", "Task Duration(us)"]
        )
        pos = 0
        for idx_ord, (case_idx, mode) in enumerate(ordered):
            tag = f"{mode}{idx_ord}"
            info = perf_log.get(tag, {})
            case_dir = out / run_cases[case_idx].replace("/", "_")
            case_dir.mkdir(parents=True, exist_ok=True)
            sub_csv = case_dir / "op_summary.csv"

            if pos + args.perf_runs <= len(flash_rows) and info.get("ok", False):
                chunk = flash_rows[pos : pos + args.perf_runs]
                pos += args.perf_runs
                with open(sub_csv, "w", newline="") as f:
                    w = csv_mod.DictWriter(f, fieldnames)
                    w.writeheader()
                    for row in chunk:
                        w.writerow(row)
            else:
                with open(case_dir / "ERROR", "w") as ef:
                    ef.write(info.get("error", "case crashed or missing entries"))
                with open(sub_csv, "w", newline="") as f:
                    w = csv_mod.DictWriter(f, fieldnames)
                    w.writeheader()

        entries = parse_op_summary(str(csv_path))
        stats, crashed = compute_stats(
            entries, cases_list, args.perf_runs, args.perf_cold_thr, log_dir=str(out)
        )

        SEP = "─" * 110
        max_name = max((len(n) for n in run_cases if n), default=55)
        logger.info(f"\n┌{SEP}┐")
        logger.info(f"│  perf test  ({len(stats)} cases, {len(crashed)} crashed)")
        logger.info(f"├{SEP}┤")
        logger.info(
            f"│  {'Case':<{max_name}}  {'Avg(us)':>9}  {'Min(us)':>9}  {'Max(us)':>9}  {'Mode':>6}  │"
        )
        logger.info(f"├{SEP}┤")
        for idx, name in enumerate(run_cases):
            if idx >= len(stats):
                continue
            s = stats[idx]
            is_crash = s["mode"] == "CRASH"
            if is_crash:
                logger.info(
                    f"│  {name:<{max_name}}  {'-':>9}  {'-':>9}  {'-':>9}  {'CRASH':>6}  │"
                )
            else:
                logger.info(
                    f"│  {name:<{max_name}}  {s['avg_us']:>9.2f}  {s['min_us']:>9.2f}  "
                    f"{s['max_us']:>9.2f}  {s['mode']:>6}  │"
                )
        logger.info(f"└{SEP}┘")

        if not args.one_by_one:
            perf_csv = out / "perf.csv"
            with open(perf_csv, "w", newline="") as f:
                w = csv_mod.DictWriter(
                    f, ["#", "case", "dir", "avg_us", "mode", "error"]
                )
                w.writeheader()
                for idx, name in enumerate(run_cases):
                    if idx >= len(stats):
                        continue
                    s = stats[idx]
                    is_crash = s["mode"] == "CRASH"
                    w.writerow(
                        dict(
                            **{"#": idx + 1},
                            case=name,
                            dir=f"case_{idx:04d}",
                            avg_us="" if is_crash else s["avg_us"],
                            mode=s["mode"],
                            error=s.get("error", "") if is_crash else "",
                        )
                    )

        logger.info(f"[perf] {out / 'perf.csv'}")
        if crashed:
            logger.info(f"\n[perf] {len(crashed)} cases crashed:")
            for idx in crashed:
                if idx < len(run_cases):
                    logger.info(f"  case_{idx:04d}  {run_cases[idx]}")

        sys.exit(0)

    # ── compare 模式复用正常流程，只改对比目标 ──
    # 不单独分支，通过 compare_dir 参数传入 run_case 即可

    # ── load_golden 模式 ──
    if args.load_golden and not args.compare_mode:
        if args.use_gpu:
            try:
                from core.backends.gpu import GPUBackend

                primary = GPUBackend(args.gpu_device)
            except ImportError:
                logger.error("[ERROR] GPU backend unavailable")
                sys.exit(1)
        else:
            try:
                from core.backends.npu import NPUBackend

                primary = NPUBackend(args.device_id, graph_mode=args.graph_mode)
            except ImportError:
                logger.error("[ERROR] NPU backend unavailable")
                sys.exit(1)

        if not primary.is_available():
            logger.error(f"[ERROR] primary backend ({primary.name}) unavailable")
            sys.exit(1)

        reporter = Reporter()
        if args.result_csv:
            csv_path = Path(args.result_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            reporter.set_csv(csv_path)

        for i, name in enumerate(run_cases):
            case_params = normalize_params(all_cases[name]) if name in all_cases else {}
            dtype = case_params.get("Dtype", "fp16")
            logger.info(f"\n{'=' * 66}")
            logger.info(f"  [load] Case: {name}  dtype={dtype}")
            logger.info(f"{'=' * 66}")

            try:
                result = run_case_load(
                    case_name=name,
                    golden_dir=args.load_golden,
                    primary=primary,
                    device_id=args.device_id,
                    verbose_diff=args.verbose_diff,
                    visualize=args.visualize,
                    viz_dir=args.viz_dir,
                    fail_analysis=args.fail_analysis,
                )
                err = None
            except FileNotFoundError:
                logger.info(f"  [warn] golden data not found, skipping: {name}")
                continue
            except Exception as e:
                import traceback

                err = f"{e}\n{traceback.format_exc()}"
                logger.error(f"[ERROR] {name} runtime error: {e}")
                result = {
                    "attn": {
                        "passed": False,
                        "max_abs": float("nan"),
                        "mean_abs": float("nan"),
                        "fail_cnt": -1,
                        "total": -1,
                        "fail_ratio": float("nan"),
                    },
                    "lse": {
                        "passed": False,
                        "max_abs": float("nan"),
                        "mean_abs": float("nan"),
                        "fail_cnt": -1,
                        "total": -1,
                        "fail_ratio": float("nan"),
                    },
                }

            reporter.record(name, result, error=err, dtype=dtype)

            if (
                args.report_interval > 0
                and (i + 1) % args.report_interval == 0
                and (i + 1) < len(run_cases)
            ):
                reporter.print_interim(len(run_cases))

        fail_cnt = reporter.print_final()
        if args.result_csv:
            logger.info(f"[result] {args.result_csv}")
        sys.exit(0 if fail_cnt == 0 else 1)

    # ── 选择后端（默认/save 模式）──
    if args.use_gpu:
        try:
            from core.backends.gpu import GPUBackend

            primary = GPUBackend(args.gpu_device)
        except ImportError:
            logger.error("[ERROR] GPU backend unavailable")
            sys.exit(1)
    else:
        try:
            from core.backends.npu import NPUBackend

            primary = NPUBackend(
                args.device_id, meta_only=args.meta_only, graph_mode=args.graph_mode
            )
        except ImportError:
            logger.error("[ERROR] NPU backend unavailable")
            sys.exit(1)

    from core.backends.cpu import CPUBackend

    golden = CPUBackend()
    comparators = []

    # ── 逐 case 执行 ──
    reporter = Reporter()
    if args.result_csv:
        csv_path = Path(args.result_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        reporter.set_csv(csv_path)
    for i, name in enumerate(run_cases):
        params = normalize_params(all_cases[name])
        layout = params.get("input_layout", "BNSD")
        dtype = params.get("Dtype", "fp16")

        logger.info(f"\n{'=' * 66}")
        logger.info(
            f"  Case: {name}  "
            f"B={params.get('B')} N1={params['N1']} N2={params.get('N2')} "
            f"S1={params.get('S1')} S2={params.get('S2')} D={params['D']} "
            f"layout={layout} dtype={dtype}"
        )
        logger.info(f"{'=' * 66}")

        primary_available = primary.is_available()

        n1 = params.get("N1", 1)
        n2 = params.get("N2", n1)
        if not (n1 >= n2 and n1 % n2 == 0):
            logger.info(
                f"  skip: N1={n1}, N2={n2} does not satisfy N1>=N2 and N1%N2==0"
            )
            reporter.record(
                name,
                {
                    "attn": {
                        "passed": False,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "fail_cnt": 0,
                        "total": 0,
                        "fail_ratio": 0.0,
                    },
                    "lse": {
                        "passed": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "fail_cnt": 0,
                        "total": 0,
                        "fail_ratio": 0.0,
                    },
                },
                error="skipped: N1<N2 or N1%N2!=0",
                dtype=dtype,
            )
            if (
                args.report_interval > 0
                and (i + 1) % args.report_interval == 0
                and (i + 1) < len(run_cases)
            ):
                reporter.print_interim(len(run_cases))
            continue
        if not primary_available:
            logger.error("[ERROR] primary backend unavailable")
            err = "primary backend not available"
            reporter.record(
                name,
                {
                    "attn": {
                        "passed": False,
                        "max_abs": float("nan"),
                        "mean_abs": float("nan"),
                        "fail_cnt": -1,
                        "total": -1,
                        "fail_ratio": float("nan"),
                    },
                    "lse": {
                        "passed": False,
                        "max_abs": float("nan"),
                        "mean_abs": float("nan"),
                        "fail_cnt": -1,
                        "total": -1,
                        "fail_ratio": float("nan"),
                    },
                },
                error=err,
                dtype=dtype,
            )
            if (
                args.report_interval > 0
                and (i + 1) % args.report_interval == 0
                and (i + 1) < len(run_cases)
            ):
                reporter.print_interim(len(run_cases))
            continue

        try:
            result = run_case(
                params=params,
                primary=primary,
                golden=golden,
                spec=flash_attn_inputs,
                comparators=comparators if comparators else None,
                golden_dir=args.save_golden,
                case_name=name,
                verbose_diff=args.verbose_diff,
                visualize=args.visualize,
                viz_dir=args.viz_dir,
                fail_analysis=args.fail_analysis,
                compare_dir=args.load_golden if args.compare_mode else None,
            )
            err = None
        except Exception as e:
            import traceback

            err = f"{e}\n{traceback.format_exc()}"
            logger.error(f"[ERROR] {name} runtime error: {e}")
            traceback.print_exc()
            result = {
                "attn": {
                    "passed": False,
                    "max_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "fail_cnt": -1,
                    "total": -1,
                    "fail_ratio": float("nan"),
                },
                "lse": {
                    "passed": False,
                    "max_abs": float("nan"),
                    "mean_abs": float("nan"),
                    "fail_cnt": -1,
                    "total": -1,
                    "fail_ratio": float("nan"),
                },
            }

        reporter.record(name, result, error=err, dtype=dtype)

        del result, params
        gc.collect()
        primary.clear_cache()

        if (
            args.report_interval > 0
            and (i + 1) % args.report_interval == 0
            and (i + 1) < len(run_cases)
        ):
            reporter.print_interim(len(run_cases))

    fail_cnt = reporter.print_final()
    if args.result_csv:
        logger.info(f"[result] {args.result_csv}")
    sys.exit(0 if fail_cnt == 0 else 1)
