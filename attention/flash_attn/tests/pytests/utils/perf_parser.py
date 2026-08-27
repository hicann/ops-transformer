# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parse msprof op_summary.csv, group by case, cold/hot stats with crash detection."""

import json
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def parse_op_summary(
    csv_path: str, operator_name: str = "FlashAttn"
) -> List[Tuple[float, float]]:
    """Return [(start_time_us, duration_us), ...] sorted by start time."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    entries = [
        (float(r["Task Start Time(us)"]), float(r["Task Duration(us)"]))
        for r in rows
        if r["OP Type"] == operator_name
    ]
    entries.sort(key=lambda x: x[0])
    return entries


def _read_perf_log(log_dir: str) -> dict:
    """Read _perf_log.jsonl, return {tag: {"ok": bool, "error": str}}."""
    log_path = Path(log_dir) / "_perf_log.jsonl"
    if not log_path.exists():
        return {}
    result = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                result[d["tag"]] = d
            except json.JSONDecodeError:
                pass
    return result


def compute_stats(
    entries: List[Tuple[float, float]],
    cases: List[Dict],
    runs: int,
    cold_thr: int,
    log_dir: str = None,
) -> Tuple[List[Dict], List[int]]:
    """Group entries by case and compute statistics.

    Returns:
        results: list of {avg_us, min_us, max_us, mode, error} indexed by case order
        crashed: list of case indices that crashed
    """
    hot = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) > cold_thr]
    cold = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) <= cold_thr]
    total = len(cases)

    # Build index→tag mapping
    idx_to_tag = {}
    for tag_idx, (orig_idx, c) in enumerate(hot):
        idx_to_tag[orig_idx] = f"hot{tag_idx}"
    for tag_idx, (orig_idx, c) in enumerate(cold):
        idx_to_tag[orig_idx] = f"cold{tag_idx}"

    perf_log = _read_perf_log(log_dir) if log_dir else {}

    # Determine crashed case indices
    crashed_tags = {
        tag: info for tag, info in perf_log.items() if not info.get("ok", False)
    }
    crashed_idxs = set()
    for orig_idx, tag in idx_to_tag.items():
        if tag in crashed_tags:
            crashed_idxs.add(orig_idx)

    # Build expected entries, skipping crashed cases
    expected = []
    for tag_idx, (orig_idx, c) in enumerate(hot):
        if orig_idx in crashed_idxs:
            continue
        for _ in range(runs):
            expected.append((orig_idx, "hot"))
    for tag_idx, (orig_idx, c) in enumerate(cold):
        if orig_idx in crashed_idxs:
            continue
        for _ in range(runs):
            expected.append((orig_idx, "cold"))

    if len(entries) > len(expected) and expected:
        ratio = len(entries) // len(expected)
        if ratio > 1:
            logger.info(
                f"[perf] entries={len(entries)} expected={len(expected)} ratio={ratio}"
            )
            entries = [entries[i * ratio] for i in range(len(expected))]

    n = min(len(entries), len(expected))
    raw = defaultdict(lambda: {"hot": [], "cold": []})

    for i in range(n):
        orig_idx, mode = expected[i]
        raw[orig_idx][mode].append(entries[i][1])

    results = []
    crashed = []
    for orig_idx in range(total):
        d = raw.get(orig_idx, {"hot": [], "cold": []})
        if orig_idx in crashed_idxs or (not d["hot"] and not d["cold"]):
            error_msg = ""
            tag = idx_to_tag.get(orig_idx, "")
            info = crashed_tags.get(tag, {})
            error_msg = info.get("error", "").replace('"', "'")
            results.append(
                {
                    "avg_us": 0.0,
                    "min_us": 0.0,
                    "max_us": 0.0,
                    "mode": "CRASH",
                    "error": error_msg,
                }
            )
            crashed.append(orig_idx)
            continue

        if d["hot"]:
            keep = d["hot"][1:]
            avg = sum(keep) / len(keep) if keep else 0.0
            mode = "hot"
            all_durs = d["hot"]
        elif d["cold"]:
            vals = sorted(d["cold"])
            keep = vals[1:-1] if len(vals) >= 3 else vals
            avg = sum(keep) / len(keep) if keep else 0.0
            mode = "cold"
            all_durs = d["cold"]
        else:
            results.append(
                {
                    "avg_us": 0.0,
                    "min_us": 0.0,
                    "max_us": 0.0,
                    "mode": "CRASH",
                    "error": "",
                }
            )
            crashed.append(orig_idx)
            continue

        results.append(
            {
                "avg_us": round(avg, 2),
                "min_us": round(min(all_durs), 2) if all_durs else 0.0,
                "max_us": round(max(all_durs), 2) if all_durs else 0.0,
                "mode": mode,
                "error": "",
            }
        )

    return results, crashed
