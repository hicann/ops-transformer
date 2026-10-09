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

"""TTK e2e custom compare for mixed_quant_flash_attn.

精度报告风格 ported from pytests/core/compare.py::check_result_:
  - bf16: atol=0.0001, rtol=0.0078125
  - fp16: atol=0.000025, rtol=0.005
  - diff_thd=0.005  →  pct_thd = 99.5%
  - max_diff_hd=10.0 (max relative error threshold)
  - PASS when fulfill_percent >= 99.5% AND max_rel_err < 10.0
打印格式对齐 pytest check_result_：框线报告 + 失败点位明细。
"""

import sys
import numpy as np

DIFF_THD = 0.005
MAX_DIFF_HD = 10.0
_SEP = "─" * 64


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_tolerance(dtype_str):
    if "bfloat16" in dtype_str:
        return 0.0001, 0.0078125
    return 0.000025, 0.005


def _check(
    atol,
    rtol,
    npu_out,
    golden_out,
    label,
    except_label="CPU",
    comp_label="NPU",
    verbose_diff=False,
    max_err_show=10,
):
    """精度对比 + 详细报告打印。返回 TTK compare 结果 dict。

    打印风格对齐 pytests/core/compare.py::check_result_：
      - 框线报告（Shape/MaxAbsErr/MeanAbsErr/MaxRelErr/MeanRelErr/FailElems/PctRlt/Threshold/结论）
      - 失败时打印前 max_err_show 个失败点位（idx/expect/real/absErr/relErr）
    """
    real = _to_numpy(npu_out)
    expect = _to_numpy(golden_out)

    # None / shape / size 校验
    if real is None and expect is None:
        print(
            f"\n┌{_SEP}┐\n│  精度报告: {label}  ({except_label} vs {comp_label})  [both None → PASS]\n└{_SEP}┘",
            flush=True,
        )
        return {
            "pass": True,
            "precision": 100.0,
            "error_info": None,
            "metrics": {"rows": 0},
        }
    if real is None or expect is None:
        print(
            f"\n┌{_SEP}┐\n│  精度报告: {label}  ({except_label} vs {comp_label})\n│  [ERROR] one side is None\n└{_SEP}┘",
            file=sys.stderr,
            flush=True,
        )
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{label}: one side is None",
            "metrics": {},
        }
    if real.shape != expect.shape:
        print(
            f"\n┌{_SEP}┐\n│  精度报告: {label}  ({except_label} vs {comp_label})\n│  [ERROR] shape不匹配: expect={tuple(expect.shape)}  {comp_label}={tuple(real.shape)}\n└{_SEP}┘",
            file=sys.stderr,
            flush=True,
        )
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{label}: shape mismatch npu={real.shape} gold={expect.shape}",
            "metrics": {},
        }

    real = real.astype(np.float32).reshape(-1)
    expect = expect.astype(np.float32).reshape(-1)
    total = real.size
    if total == 0:
        print(
            f"\n┌{_SEP}┐\n│  精度报告: {label}  ({except_label} vs {comp_label})  [空输出 → PASS]\n└{_SEP}┘",
            flush=True,
        )
        return {
            "pass": True,
            "precision": 100.0,
            "error_info": None,
            "metrics": {"rows": 0},
        }

    # isclose 判定
    diff_result = np.isclose(real, expect, rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != True)[0]
    fail_cnt = int(err_idx.size)
    fulfill_percent = (total - fail_cnt) / total * 100.0
    fail_ratio = fail_cnt / total

    # abs / rel err
    diff_abs = np.abs(expect - real)
    max_abs = float(diff_abs.max()) if diff_abs.size > 0 else 0.0
    mean_abs = float(diff_abs.mean()) if diff_abs.size > 0 else 0.0

    b1 = np.maximum(np.abs(real), np.abs(expect))
    b2 = float((1.0 / (1 << 14)) / DIFF_THD)
    b = np.add(np.maximum(b1, b2), 10e-10)
    rel_err = diff_abs / (b + 10e-10)
    err_diff = rel_err[err_idx] if err_idx.size > 0 else np.array([0.0])
    max_rel = float(err_diff.max()) if err_diff.size > 0 else 0.0
    mean_rel = float(rel_err.mean()) if rel_err.size > 0 else 0.0

    pct_thd = (1 - DIFF_THD) * 100.0
    passed = fulfill_percent >= pct_thd
    if err_diff.size > 0 and float(err_diff.max()) >= MAX_DIFF_HD:
        passed = False

    # ===== 框线报告（对齐 pytest check_result_）=====
    print(f"\n┌{_SEP}┐", flush=True)
    print(f"│  精度报告: {label}  ({except_label} vs {comp_label})", flush=True)
    print(f"├{_SEP}┤", flush=True)
    print(f"│  Shape       : {tuple(expect.shape)}", flush=True)
    print(f"│  MaxAbsErr   : {max_abs:.8f}", flush=True)
    print(f"│  MeanAbsErr  : {mean_abs:.8f}", flush=True)
    print(f"│  MaxRelErr   : {max_rel:.8f}", flush=True)
    print(f"│  MeanRelErr  : {mean_rel:.8f}", flush=True)
    print(
        f"│  FailElems   : {fail_cnt} / {total}  ({fail_ratio * 100:.4f}%)", flush=True
    )
    print(f"│  PctRlt      : {fulfill_percent:.6f}%", flush=True)
    print(
        f"│  Threshold   : atol={atol}  rtol={rtol * 100:.2f}%  pctThd≥{pct_thd:.2f}%  maxRelErr<{MAX_DIFF_HD}",
        flush=True,
    )
    print(f"│  结论        : {'✓ PASS' if passed else '✗ FAIL'}", flush=True)

    # ===== 失败点位明细 =====
    if fail_cnt > 0:
        print(f"├{_SEP}┤", flush=True)
        show_cnt = len(err_idx) if verbose_diff else min(max_err_show, len(err_idx))
        if verbose_diff:
            print(
                f"│  全部 {len(err_idx)} 个超阈値元素 (rtol={rtol * 100:.2f}%):",
                flush=True,
            )
        else:
            print(f"│  前{show_cnt}个不通过元素:", flush=True)
        print(
            f"│  {'idx':>10}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>12}",
            flush=True,
        )
        for i in err_idx[:show_cnt]:
            print(
                f"│  {int(i):>10}  {float(expect[i]):>+14.8f}  {float(real[i]):>+14.8f}"
                f"  {float(diff_abs[i]):>12.8f}  {float(rel_err[i]):>12.6f}",
                flush=True,
            )
        if not verbose_diff and len(err_idx) > max_err_show:
            print(
                f"│  ... (共 {len(err_idx)} 个失败点，仅显示前 {max_err_show} 个)",
                flush=True,
            )

        # max rel err 点位（即使不 verbose 也打印）
        if err_idx.size > 0:
            max_rel_idx = int(err_idx[np.argmax(err_diff)])
            print(f"├{_SEP}┤", flush=True)
            print("│  MaxRelErr 点位:", flush=True)
            print(
                f"│  idx={max_rel_idx}  {except_label}={float(expect[max_rel_idx]):>+14.8f}"
                f"  {comp_label}={float(real[max_rel_idx]):>+14.8f}"
                f"  absErr={float(diff_abs[max_rel_idx]):>12.8f}"
                f"  relErr={float(rel_err[max_rel_idx]):>12.6f}",
                flush=True,
            )
    print(f"└{_SEP}┘", flush=True)

    # 失败时同时打到 stderr 便于 TTK log 抓取
    if not passed:
        print(
            f"[compare:{label}] FAIL fulfill={fulfill_percent:.4f}% (thd={pct_thd:.2f}%) "
            f"fail={fail_cnt}/{total} max_abs={max_abs:.8f} max_rel={max_rel:.8f} (thd={MAX_DIFF_HD})",
            file=sys.stderr,
            flush=True,
        )

    return {
        "pass": bool(passed),
        "precision": float(fulfill_percent),
        "error_info": None
        if passed
        else (
            f"{label}: fail={fail_cnt}/{total} ({fail_ratio * 100:.4f}%) "
            f"max_abs={max_abs:.8f} max_rel={max_rel:.8f}"
        ),
        "metrics": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "max_rel": max_rel,
            "mean_rel": mean_rel,
            "fail_cnt": fail_cnt,
            "total": total,
            "fail_ratio": float(fail_ratio),
            "fulfill_percent": float(fulfill_percent),
            "atol": atol,
            "rtol": rtol,
        },
    }


def compare(*outputs, **kwargs):
    """TTK custom compare entry.

    outputs layout: [npu_out_0, npu_out_1, ..., golden_out_0, golden_out_1, ...]
    mixed_quant_flash_attn has 2 outputs: attn_out and softmax_lse.
    softmax_lse may be empty when return_softmax_lse=False; skip it then.

    可通过 kwargs 控制打印：
      - verbose_diff: True 打印全部失败点（默认 False，只打前 10 个）
      - max_err_show: 非 verbose 时显示的失败点数（默认 10）
    """
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    verbose_diff = bool(kwargs.get("verbose_diff", False))
    max_err_show = int(kwargs.get("max_err_show", 10))

    npu_attn = npu_outputs[0]
    dtype_str = str(getattr(npu_attn, "dtype", "float32"))
    atol, rtol = _get_tolerance(dtype_str)

    attn_stats = _check(
        atol,
        rtol,
        npu_attn,
        golden_outputs[0],
        "attn_out",
        verbose_diff=verbose_diff,
        max_err_show=max_err_show,
    )

    lse_stats = {
        "pass": True,
        "precision": 100.0,
        "error_info": None,
        "metrics": {"rows": 0},
    }
    if half >= 2:
        npu_lse_np = _to_numpy(npu_outputs[1])
        gold_lse_np = _to_numpy(golden_outputs[1])
        if (
            npu_lse_np is not None
            and gold_lse_np is not None
            and npu_lse_np.size > 0
            and gold_lse_np.size > 0
        ):
            lse_stats = _check(
                atol,
                rtol,
                npu_outputs[1],
                golden_outputs[1],
                "softmax_lse",
                verbose_diff=verbose_diff,
                max_err_show=max_err_show,
            )

    return [attn_stats, lse_stats]
