#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""msprof 性能采集(最小版, 抽取自 ops-transformer-testkit core/profiler.py)。

以 output_dir 为 cwd 运行生成的脚本, task-based 模式采集 AI Core kernel,
然后从最新的 PROF_*/mindstudio_profiler_output 拷出 op_summary CSV。
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_msprof(
    script: str, output_dir: Path, label: str = "case", total_timeout: int = 7200
) -> Path:
    """在 output_dir 下写脚本并用 msprof 执行, 返回拷出的 op_summary.csv 路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{label}.py"
    script_path.write_text(script)

    msprof_dir = output_dir / "msprof"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = [
        "msprof",
        f"--output={msprof_dir}",
        "--aic-mode=task-based",
        sys.executable,
        str(script_path),
    ]
    log_path = output_dir / f"{label}.msprof.log"
    print(f"[msprof] {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        subprocess.run(
            cmd,
            cwd=output_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=total_timeout,
            check=False,
        )

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        raise FileNotFoundError(f"No PROF_* found in {msprof_dir}, see {log_path}")
    csv_files = sorted(prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No op_summary CSV in {prof_dirs[0]}, see {log_path}")

    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] {label} -> {dst}")
    return dst


def run_msopprof_pipe(
    script: str,
    output_dir: Path,
    label: str = "case",
    kernel_name: str = "QuantFlashMlaWithKvcache",
    launch_count: int = 1,
    total_timeout: int = 7200,
) -> Path:
    """msprof op --aic-metrics=PipeTimeline 采集 pipe 时间线。

    采集 PMU 硬件计数级别的 Cube/Vector/MTE pipe 时间线, 产物为 OPPROF_* 目录,
    其中 visualize_data.bin 用 MindStudio Insight 打开可视化。
    返回最新的 OPPROF_* 目录路径。
    """
    pipe_dir = output_dir / "pipe_timeline"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    script_path = pipe_dir / f"{label}_pipe.py"
    script_path.write_text(script)

    cmd = [
        "msprof",
        "op",
        f"--output={pipe_dir}",
        "--aic-metrics=PipeTimeline",
        f"--launch-count={launch_count}",
        f"--kernel-name={kernel_name}",
        sys.executable,
        str(script_path),
    ]
    log_path = pipe_dir / f"{label}.msopprof.log"
    print(f"[msopprof] {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        subprocess.run(
            cmd,
            cwd=pipe_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=total_timeout,
            check=False,
        )

    opprof_dirs = sorted(pipe_dir.glob("OPPROF_*"), reverse=True)
    if not opprof_dirs:
        raise FileNotFoundError(f"No OPPROF_* in {pipe_dir}, see {log_path}")
    print(f"[msopprof] {label} -> {opprof_dirs[0]}")
    return opprof_dirs[0]
