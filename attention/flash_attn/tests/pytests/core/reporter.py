# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""终端表格 + CSV 报告。"""

import csv
import logging
import sys
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class Reporter:
    """收集每条 case 的结果，支持中间统计和最终 CSV 输出。"""

    def __init__(self):
        self._results: Dict[str, dict] = {}
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None

    def set_csv(self, path: Path):
        """设置增量 CSV 输出路径，每条 record 自动追加。"""
        self._csv_path = path
        exists = path.exists()
        self._csv_file = open(path, "a", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            [
                "case",
                "dtype",
                "attn_pass",
                "attn_max_abs",
                "attn_fail_ratio",
                "lse_pass",
                "lse_max_abs",
                "lse_fail_ratio",
                "error",
            ],
        )
        if not exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def record(self, name: str, result: dict, error: str = None, dtype: str = ""):
        if error:
            result["_error"] = error
        if dtype:
            result["_dtype"] = dtype
        self._results[name] = result

        if self._csv_writer:
            a = result.get("attn", {})
            l = result.get("lse", {})
            self._csv_writer.writerow(
                {
                    "case": name,
                    "dtype": dtype,
                    "attn_pass": a.get("passed"),
                    "attn_max_abs": a.get("max_abs", ""),
                    "attn_fail_ratio": _pct(a.get("fail_ratio")),
                    "lse_pass": l.get("passed") if l.get("total", 0) > 0 else "",
                    "lse_max_abs": l.get("max_abs", ""),
                    "lse_fail_ratio": _pct(l.get("fail_ratio"))
                    if l.get("total", 0) > 0
                    else "",
                    "error": error or "",
                }
            )
            self._csv_file.flush()

    def print_interim(self, total: int):
        self._print_table(self._results, total, is_final=False)

    def print_final(self) -> int:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        return self._print_table(self._results, len(self._results), is_final=True)

    @staticmethod
    def _print_table(results: dict, total: int, is_final: bool) -> int:
        max_len = max((len(n) for n in results.keys()), default=28)
        width = max_len + 100
        SEP = "─" * width
        title = (
            "  summary"
            if is_final
            else f"  intermediate stats  ({len(results)}/{total})"
        )
        logger.info(f"\n┌{SEP}┐")
        logger.info(f"│{title}")
        logger.info(f"├{SEP}┤")
        hdr = (
            f"│  {'Case':<{max_len}}  {'Dtype':>5}  "
            f"{'Attn':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │  "
            f"{'LSE':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │"
        )
        logger.info(hdr)
        logger.info(f"├{SEP}┤")
        pass_cnt = fail_cnt = 0
        for name, res in results.items():
            a = res.get("attn", {})
            l = res.get("lse", {})
            dt = res.get("_dtype", "")
            a_tag = "PASS" if a.get("passed") else "FAIL"
            l_tag = (
                ("PASS" if l.get("passed") else "FAIL")
                if l.get("total", 0) > 0
                else "---"
            )
            a_max = f"{a.get('max_abs', 0):.6f}" if a.get("total", 0) > 0 else "N/A"
            a_fr = (
                f"{a.get('fail_ratio', 0) * 100:.4f}%"
                if a.get("total", 0) > 0
                else "N/A"
            )
            l_max = f"{l.get('max_abs', 0):.6f}" if l.get("total", 0) > 0 else "-"
            l_fr = (
                f"{l.get('fail_ratio', 0) * 100:.4f}%" if l.get("total", 0) > 0 else "-"
            )
            logger.info(
                f"│  {name:<{max_len}}  {dt:>5}  "
                f"{a_tag:>8}  {a_max:>12}  {a_fr:>10}  │  "
                f"{l_tag:>8}  {l_max:>12}  {l_fr:>10}  │"
            )
            if a.get("passed") and l.get("passed"):
                pass_cnt += 1
            else:
                fail_cnt += 1
        logger.info(f"├{SEP}┤")
        label = f"│  pass: {pass_cnt}   fail: {fail_cnt}   total: {len(results)}"
        logger.info(label)
        logger.info(f"└{SEP}┘")
        return fail_cnt


def _pct(v) -> str:
    if v is None or v == float("nan") or v < 0:
        return ""
    return f"{v * 100:.4f}%"
