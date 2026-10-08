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
"""Run a process tree on <=10 CPUs with an 18 GiB RSS watchdog.

The watchdog is not a cgroup hard limit. Shared pages are counted per process,
so the measurement is conservative; short allocation spikes can be missed.
"""

import os
import signal
import subprocess
import sys
import time

import psutil


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: limit_resources.py COMMAND [ARG ...]")
    cpus = sorted(os.sched_getaffinity(0))[:10]
    os.sched_setaffinity(0, cpus)
    limit = 18 * 1024**3
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    root = psutil.Process(child.pid)
    peak = 0

    def stop(signum, _frame):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"[resources] CPUs={cpus}; RSS watchdog=18 GiB (not a hard limit)", flush=True
    )
    try:
        while child.poll() is None:
            rss = 0
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            if rss >= limit:
                print("[resources] RSS limit reached; terminating task", flush=True)
                stop(signal.SIGTERM, None)
            time.sleep(0.1)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        print(
            f"[resources] peak sampled process-tree RSS={peak / 1024**3:.2f} GiB",
            flush=True,
        )
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    sys.exit(main())
