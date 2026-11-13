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
"""跑一批性能用例，只保留原始 msprof 数据和最终逐条表。

_perf_sweep.py 和 _perf_log.jsonl 放在 out-dir 下，跑完不删除。
"""

import argparse
import csv
import os
import time
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS_DIR))

from core.case_loader import load_case_modules, normalize_params  # noqa: E402
from utils.perf_runner import build_batch_script  # noqa: E402

CASE_PREFIX = re.compile(r"test_cases\.[^/]+/")


def _clean_line(line):
    return CASE_PREFIX.sub("", line.rstrip("\n"))


def _keep_line(line):
    text = line.strip()
    if text.startswith("[batch]") or text.startswith("[CASE_END]"):
        return True
    return any(
        key in text
        for key in (
            "CRASH",
            "Traceback",
            "Error",
            "ERROR",
            "Exception",
            "failed",
            "FAILED",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-files", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cold-thr", type=int, default=16)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="按 N 条用例分段跑，段间 sleep --gap-ms（默认 0=不分段）；分段时每段独立msprof目录，最后合并",
    )
    parser.add_argument(
        "--gap-ms",
        type=int,
        default=0,
        help="段间空闲毫秒，用于让设备状态恢复，避免持续满载影响测量",
    )
    parser.add_argument(
        "--big-case-regex",
        default=None,
        help="匹配到的 case 单独成段（逐条跑），例：'KVS(32768|131072)'",
    )
    parser.add_argument(
        "--big-gap-ms",
        type=int,
        default=0,
        help="大 case 段后的空闲毫秒（0 则用 --gap-ms）",
    )
    args = parser.parse_args()

    modules = [
        x.strip() if x.strip().startswith("test_cases.") else "test_cases." + x.strip()
        for x in args.case_files.split(",")
    ]
    all_cases = load_case_modules(modules)
    names = sorted(all_cases)
    cases = [normalize_params(all_cases[n]) for n in names]

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
        out_dir = out_dir.resolve()
    else:
        out_dir = (
            Path(__file__).resolve().parent / f"perf_{datetime.now():%Y%m%d_%H%M%S}"
        )
    if out_dir.exists():
        raise SystemExit(f"out dir already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    if args.chunk_size and args.chunk_size > 0:
        chunks = _build_chunks(names, args.chunk_size, args.big_case_regex)
    else:
        # 不分段：单次 msprof 会话，大 case 的间隔由脚本内的 time.sleep 承担
        chunks = [(list(range(len(names))), False)]
    # 单会话时把间隔放到 msprof 会话内部（比"分段重启 msprof"快一个数量级）
    args._single_session = len(chunks) == 1
    if len(chunks) > 1:
        plan = ", ".join(f"{len(c)}cases{'*' if big else ''}" for c, big in chunks)
        print(
            f"[plan] {len(chunks)} chunks: {plan}  (gap {args.gap_ms}ms / big {args.big_gap_ms or args.gap_ms}ms)",
            flush=True,
        )
    metas = []
    for ci, (idxs, is_big) in enumerate(chunks):
        chunk_dir = out_dir if len(chunks) == 1 else out_dir / f"chunk_{ci:02d}"
        chunk_names = [names[i] for i in idxs]
        chunk_cases = [cases[i] for i in idxs]
        t0 = datetime.now()
        _run_chunk(chunk_cases, chunk_names, chunk_dir, args)
        metas.append(
            dict(
                chunk=ci,
                n_cases=len(chunk_names),
                big=int(is_big),
                dir=chunk_dir.name,
                start=t0.strftime("%H:%M:%S"),
                end=datetime.now().strftime("%H:%M:%S"),
            )
        )
        if len(chunks) > 1:
            print(
                f"[chunk {ci + 1}/{len(chunks)}] {len(chunk_names)} cases{' (big)' if is_big else ''}",
                flush=True,
            )
        gap = args.big_gap_ms if (is_big and args.big_gap_ms > 0) else args.gap_ms
        if ci + 1 < len(chunks) and gap > 0:
            print(f"[gap] {gap} ms ...", flush=True)
            time.sleep(gap / 1000.0)

    if len(chunks) > 1:
        _merge_chunks(out_dir, len(chunks))
        _write_chunk_meta(out_dir, metas)
    print(f"done: {out_dir / 'profiling_raw.csv'}")
    print(f"summary: {out_dir / 'profiling_summary.csv'}")
    if len(chunks) > 1:
        print(f"chunk meta: {out_dir / 'chunk_meta.csv'}")


def _build_chunks(names, chunk_size, big_regex):
    """普通 case 按 chunk_size 成段；匹配 big_regex 的 case 单独成段（逐条跑）。"""
    import re

    big = re.compile(big_regex) if big_regex else None
    chunks = []
    pending = []
    for i, name in enumerate(names):
        if big and big.search(name):
            if pending:
                chunks.append((pending, False))
                pending = []
            chunks.append(([i], True))
        else:
            pending.append(i)
            if len(pending) >= chunk_size:
                chunks.append((pending, False))
                pending = []
    if pending:
        chunks.append((pending, False))
    return chunks or [([], False)]


def _run_chunk(cases, names, out_dir, args):
    """跑一个分段：独立 msprof 目录 + 本段 raw/summary。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    msprof_dir = out_dir / "msprof_output"
    script = out_dir / "_perf_sweep.py"
    single_session = getattr(args, "_single_session", False)
    script.write_text(
        build_batch_script(
            cases,
            args.runs,
            args.cold_thr,
            case_names=names,
            device_id=args.device_id,
            output_dir=str(out_dir),
            gap_ms=args.gap_ms if single_session else 0,
            big_gap_ms=args.big_gap_ms if single_session else 0,
            big_case_regex=args.big_case_regex if single_session else None,
        )
    )
    cmd = [
        "msprof",
        f"--output={msprof_dir}",
        "--aic-mode=task-based",
        sys.executable,
        "-u",
        str(script),
    ]
    print(
        f"start: {len(names)} cases x {args.runs} runs -> {out_dir}",
        flush=True,
    )
    log_path = out_dir / "run.log"
    full_log_path = out_dir / "run_full.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    output = []
    with open(log_path, "w") as log, open(full_log_path, "w") as full_log:
        log.write(f"start: {len(names)} cases x {args.runs} runs -> {out_dir}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            full_log.write(line)
            full_log.flush()
            output.append(line)
            if _keep_line(line):
                clean = _clean_line(line)
                print(clean, flush=True)
                log.write(clean + "\n")
                log.flush()
        returncode = proc.wait()

    if returncode != 0 or "PERF_DONE" not in "".join(output):
        raise SystemExit(
            f"perf run failed: log={log_path}, full_log={full_log_path}, script={script}"
        )

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "export_perf_raw.py"),
            "--perf-dir",
            str(out_dir),
            "--case-files",
            args.case_files,
            "--runs",
            str(args.runs),
            "--cold-thr",
            str(args.cold_thr),
            "--case-names",
            ",".join(names),
            "--out",
            str(out_dir / "profiling_raw.csv"),
        ],
        check=True,
        cwd=str(TESTS_DIR),
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "summarize_perf_raw.py"),
            "--raw",
            str(out_dir / "profiling_raw.csv"),
            "--out",
            str(out_dir / "profiling_summary.csv"),
        ],
        check=True,
        cwd=str(TESTS_DIR),
    )


def _merge_chunks(out_dir, n_chunks):
    """把各段的 raw/summary 合并成顶层文件（保留一个表头）。"""
    for fname in ("profiling_raw.csv", "profiling_summary.csv"):
        merged = []
        header = None
        for ci in range(n_chunks):
            part = out_dir / f"chunk_{ci:02d}" / fname
            with open(part) as f:
                lines = f.read().splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
                merged.extend(lines)
            else:
                merged.extend(lines[1:])
        with open(out_dir / fname, "w") as f:
            f.write("\n".join(merged) + "\n")


def _write_chunk_meta(out_dir, metas):
    if not metas:
        return
    fields = sorted({k for m in metas for k in m})
    with open(out_dir / "chunk_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metas)


if __name__ == "__main__":
    main()
