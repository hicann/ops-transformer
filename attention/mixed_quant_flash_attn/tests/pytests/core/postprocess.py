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

import os
import csv
import datetime


class PostProcessor:
    def __init__(self):
        self._results = []

    def record(self, case_name, result, error=None):
        entry = {
            "case_name": case_name,
            "attn_passed": result.get("attn", {}).get("passed", False),
            "attn_max_abs": result.get("attn", {}).get("max_abs", 0.0),
            "attn_mean_abs": result.get("attn", {}).get("mean_abs", 0.0),
            "attn_fail_cnt": result.get("attn", {}).get("fail_cnt", 0),
            "attn_total": result.get("attn", {}).get("total", 0),
            "attn_fail_ratio": result.get("attn", {}).get("fail_ratio", 0.0),
            "lse_passed": result.get("lse", {}).get("passed", True),
            "error": error,
        }
        self._results.append(entry)

    def print_summary(self):
        if not self._results:
            print("No results to summarize.")
            return 0

        max_len = max(len(r["case_name"]) for r in self._results)
        max_len = max(max_len, 20)
        width = max_len + 80
        SEP = "─" * width

        print(f"\n┌{SEP}┐")
        print(f"│  Summary ({len(self._results)} cases)")
        print(f"├{SEP}┤")
        hdr = (
            f"│  {'Case':<{max_len}}  {'Attn':>6}  {'MaxAbsErr':>12}  "
            f"{'FailRatio':>10}  {'LSE':>6}  {'Error':>20}"
        )
        print(hdr)
        print(f"├{SEP}┤")

        pass_cnt = 0
        fail_cnt = 0
        for r in self._results:
            a_tag = "PASS" if r["attn_passed"] else "FAIL"
            l_tag = "PASS" if r["lse_passed"] else "FAIL"
            a_max = f"{r['attn_max_abs']:.6f}" if r["attn_total"] > 0 else "N/A"
            a_fr = (
                f"{r['attn_fail_ratio'] * 100:.4f}%" if r["attn_total"] > 0 else "N/A"
            )
            err = (
                (r["error"][:18] + "..")
                if r["error"] and len(r["error"]) > 20
                else (r["error"] or "")
            )
            print(
                f"│  {r['case_name']:<{max_len}}  {a_tag:>6}  {a_max:>12}  "
                f"{a_fr:>10}  {l_tag:>6}  {err:>20}"
            )
            if r["attn_passed"] and r["lse_passed"]:
                pass_cnt += 1
            else:
                fail_cnt += 1

        print(f"├{SEP}┤")
        total = len(self._results)
        pass_rate = pass_cnt / total * 100 if total > 0 else 0
        print(
            f"│  Pass: {pass_cnt}   Fail: {fail_cnt}   Total: {total}   PassRate: {pass_rate:.1f}%"
        )
        print(f"└{SEP}┘")
        return fail_cnt

    def save_csv(self, csv_path):
        os.makedirs(
            os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
            exist_ok=True,
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                [
                    "case_name",
                    "attn_passed",
                    "attn_max_abs",
                    "attn_mean_abs",
                    "attn_fail_cnt",
                    "attn_total",
                    "attn_fail_ratio",
                    "lse_passed",
                    "error",
                ],
            )
            writer.writeheader()
            for r in self._results:
                writer.writerow(r)
        print(f"Results saved to {csv_path}")
