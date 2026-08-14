#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import datetime
import os
import sys
import logging


np.set_printoptions(suppress=True)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# ================================================================
#  精度阈值配置
# ================================================================
#   rtol/atol:    np.isclose 的相对/绝对容差
#   pct_thd:      通过率阈值，0.005 表示 99.5% 的点位需满足 rtol/atol
#   diff_thd:     相对误差归一化分母系数（b = max(|real|,|expect|, 1/(2^14)/diff_thd)）
#   max_diff_hd:  单点最大允许相对误差阈值，超过即判 Failed
PRECISION = {
    "bfloat16": {
        "rtol": 0.0078125,
        "atol": 0.0001,
        "pct_thd": 0.005,
        "diff_thd": 0.005,
        "max_diff_hd": 10.0,
    },
    "float16": {
        "rtol": 0.005,
        "atol": 0.000025,
        "pct_thd": 0.005,
        "diff_thd": 0.005,
        "max_diff_hd": 10.0,
    },
    "float32": {
        "rtol": 0.005,
        "atol": 0.000025,
        "pct_thd": 0.005,
        "diff_thd": 0.005,
        "max_diff_hd": 10.0,
    },
}


# ================================================================
#  输出定义
# ================================================================
# 算子所有输出列表（顺序固定，便于汇总统计）：
# 反向 4 个 + 正向 3 个 + state_cache 4 个；通路 2/3 的 result 无正向/state 字段，
# 打印/汇总时自动跳过
ALL_OUTPUTS = [
    "d_wkv",
    "d_wgate",
    "d_ape",
    "d_x",
    "cmp_kv",
    "softmax_score",
    "kv",
    "kv_state_update",
    "score_state_update",
    "kv_state_origin",
    "score_state_origin",
]

# 输出名 → result dict 中的 status/pct key 映射
_OUTPUT_KEY_MAP = {
    "d_wkv": ("dwkvPct", "dwkvStatus"),
    "d_wgate": ("dwgatePct", "dwgateStatus"),
    "d_ape": ("apePct", "apeStatus"),
    "d_x": ("dxPct", "dxStatus"),
    "cmp_kv": ("cmpkvPct", "cmpkvStatus"),
    "softmax_score": ("smPct", "smStatus"),
    "kv": ("kvPct", "kvStatus"),
    "kv_state_update": ("kvUpdPct", "kvUpdStatus"),
    "score_state_update": ("scoreUpdPct", "scoreUpdStatus"),
    "kv_state_origin": ("kvOrgPct", "kvOrgStatus"),
    "score_state_origin": ("scoreOrgPct", "scoreOrgStatus"),
}


def get_output_keys(out_name):
    """返回输出名对应的 (pct_key, status_key)。"""
    return _OUTPUT_KEY_MAP[out_name]


def get_pct_thd(data_type):
    """根据 dtype 字符串返回通过率阈值 pct_thd。"""
    return PRECISION[data_type]["pct_thd"]


def _format_threeway_row(seq, a_val, b_val, c_val, re_a, re_b):
    """格式化单行点位：A(NPU)/B(小算子)/C(高精度) 三值 + 相对 C 的两列相对误差。
    C 为 inf/nan 时用字符串列（与两方 _format_loop_row 同款处理）。
    数值列右对齐宽度 10（负数多占负号位，保证小数点对齐）。"""
    if "inf" in str(c_val) or "nan" in str(c_val):
        return f"{seq:08d} \t {a_val:>10} \t {b_val:>10} \t {c_val:>10} \t {'inf/nan':>10} \t {'inf/nan':>10}"
    return f"{seq:08d} \t {a_val:>10.7f} \t {b_val:>10.7f} \t {c_val:>10.7f} \t {re_a:>10.7f} \t {re_b:>10.7f}"


def _display_threeway_table(a_flat, b_flat, c_flat, re_a, re_b, name):
    """三方点位表：Loop 风格（对齐两方 display_output_np_isclose，首尾各 10 行）。"""
    print_log(_ROW_DIVIDER)
    print_log(f"[three-way] {name}:")
    print_log("Loop \t A(NPU) \t B(same) \t C(high) \t A_RE \t B_RE")
    print_log(_ROW_DIVIDER)
    split_count = int(a_flat.numel())
    if split_count <= 20:
        shown = range(split_count)
    else:
        shown = list(range(10)) + list(range(split_count - 10, split_count))
    for idx in shown:
        print_log(
            _format_threeway_row(
                idx,
                a_flat[idx].item(),
                b_flat[idx].item(),
                c_flat[idx].item(),
                re_a[idx].item(),
                re_b[idx].item(),
            )
        )
    if split_count > 20:
        print_log("... \t ... \t ... \t ... \t ... \t ...")
    print_log(_ROW_DIVIDER)


def three_way_report(name, npu_out, golden_out, ref_out, data_type):
    """三方精度判定：metric_A / max(metric_B, err) < ratio（MARE=10/MERE=2/RMSE=2）。

    小值域兼容：|C| >= err 的元素走原 MARE/MERE/RMSE；|C| < err 的元素走
    errorcount 比较（errorcount_A / max(errorcount_B, 1) < 2）。

    打印格式（对齐两方列表风格）：
      1. 点位表：每行 A(NPU)/B(小算子)/C(高精度) 三值 + A_RE/B_RE 两列相对误差
         （分别相对 C），首尾各 10 行；
      2. MARE/MERE/RMSE 分块打印：每类单独一块，A 值/B 值/ratio/阈值/状态各一行，
         块间用标题分隔线区分；
      3. SMALL 小值域 errorcount 兼容块。
    """
    ERR_BY_DTYPE = {"float16": 2e-11, "bfloat16": 2e-8, "float32": 2e-14}
    ERROR_BY_DTYPE = {"float16": 2e-16, "bfloat16": 2e-16, "float32": 2e-30}
    RATIO = {"MARE": 10.0, "MERE": 2.0, "RMSE": 2.0}
    err = ERR_BY_DTYPE[data_type]
    error = ERROR_BY_DTYPE[data_type]
    a = npu_out.detach().cpu().float()
    b = golden_out.detach().cpu().float()
    c = ref_out.detach().cpu().float()
    if a.numel() == 0:
        print_log(f"[three-way] {name}: empty output, skip")
        return {"status": "PASS", "ratios": {"MARE": 0.0, "MERE": 0.0, "RMSE": 0.0}}
    da = (a - c).abs()
    db = (b - c).abs()
    large = c.abs() >= err
    denom = c.abs().clamp(min=1e-7)
    if large.any():
        mare_a = (da[large] / denom[large]).max().item()
        mare_b = (db[large] / denom[large]).max().item()
        mere_a = (da[large] / denom[large]).mean().item()
        mere_b = (db[large] / denom[large]).mean().item()
        rmse_a = da[large].pow(2).mean().sqrt().item()
        rmse_b = db[large].pow(2).mean().sqrt().item()
        ratios = {
            "MARE": mare_a / max(mare_b, err),
            "MERE": mere_a / max(mere_b, err),
            "RMSE": rmse_a / max(rmse_b, err),
        }
        ok_main = all(ratios[m] < RATIO[m] for m in RATIO)
    else:
        mare_a = mare_b = mere_a = mere_b = rmse_a = rmse_b = 0.0
        ratios = {"MARE": 0.0, "MERE": 0.0, "RMSE": 0.0}
        ok_main = True
    small = ~large
    err_cnt_a = int((small & (da > error)).sum())
    err_cnt_b = int((small & (db > error)).sum())
    small_ratio = err_cnt_a / max(err_cnt_b, 1)
    ok_small = small_ratio < 2
    ok = ok_main and ok_small

    # ── 1. 点位表（Loop 风格，A/B/C 三值 + 相对 C 的两列相对误差，首尾各 10 行）──
    a_flat = a.flatten()
    b_flat = b.flatten()
    c_flat = c.flatten()
    denom_pt = c_flat.abs() + 1e-9
    re_a = (a_flat - c_flat).abs() / denom_pt
    re_b = (b_flat - c_flat).abs() / denom_pt
    _display_threeway_table(a_flat, b_flat, c_flat, re_a, re_b, name)

    # ── 2. MARE / MERE / RMSE 分块打印（每类单独一块，A/B/ratio/阈值/状态各一行）──
    metric_vals = [
        ("MARE", mare_a, mare_b),
        ("MERE", mere_a, mere_b),
        ("RMSE", rmse_a, rmse_b),
    ]
    for mname, val_a, val_b in metric_vals:
        print_log(f"================ {mname} ================")
        print_log(f"  A      = {val_a:.6e}")
        print_log(f"  B      = {val_b:.6e}")
        print_log(f"  ratio  = {ratios[mname]:.6f}")
        print_log(f"  thd    = {RATIO[mname]:.0f}")
        print_log(f"  status = {'PASS' if ratios[mname] < RATIO[mname] else 'FAIL'}")

    # ── 3. SMALL 小值域 errorcount 兼容块 ──
    print_log("================ SMALL ================")
    print_log(f"  err_cnt_A = {err_cnt_a}")
    print_log(f"  err_cnt_B = {err_cnt_b}")
    print_log(f"  ratio     = {small_ratio:.3f}")
    print_log("  thd       = 2")
    print_log(f"  status    = {'PASS' if ok_small else 'FAIL'}")
    print_log(_ROW_DIVIDER)
    print_log(f"[three-way] {name}: Result {'PASS' if ok else 'FAIL'}")
    print_log(_ROW_DIVIDER)
    return {"status": "PASS" if ok else "FAIL", "ratios": ratios}


# ================================================================
#  日志输出
# ================================================================
def print_log(data=None, level="INFO"):
    stamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    caller = sys._getframe().f_back
    caller_file = os.path.basename(caller.f_code.co_filename)
    caller_line = str(caller.f_lineno).zfill(4)
    print(f"[{stamp}] [{level}]-{caller_file}:{caller_line} - {data}")


# ================================================================
#  单点位精度对比展示
# ================================================================
# 精度对比表格的共用分隔线与表头（display/check 系列函数共享）
_ROW_DIVIDER = "-" * 87
_ERROR_DIVIDER = "Error Line" + "-" * 77
_MAX_RE_DIVIDER = "Max-RE line:" + "-" * 75
_LOOP_HEADER = "Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff"


def cal_relative_diff_np_isclose(real_data, expect_data):
    """单点相对误差：|real - expect| / (|expect| + 1e-9)。"""
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + 1e-9)


def _format_loop_row(seq, expect_val, real_val):
    """格式化单行点位：期望值为 inf/nan 时用字符串列，否则用 7 位小数。"""
    diff_rate = cal_relative_diff_np_isclose(real_val, expect_val)
    if "inf" in str(expect_val) or "nan" in str(expect_val):
        diff_abs = "inf" if "inf" in str(expect_val) else "nan"
        return f"{seq:08d} \t {expect_val:<7} \t {real_val:<7} \t {diff_abs:<7} \t {diff_rate:<7}"
    diff_abs = abs(np.float64(expect_val) - np.float64(real_val))
    return f"{seq:08d} \t {expect_val:.7f} \t {real_val:.7f} \t {diff_abs:.7f} \t {diff_rate:.7f}"


def _format_error_row(seq, expect_val, real_val, diff_rate):
    """格式化错误点位明细行（7 位小数定点）。"""
    diff_abs = abs(np.float64(expect_val) - np.float64(real_val))
    return f"{seq:08d} \t {expect_val:.7f} \t {real_val:.7f} \t {diff_abs:.7f} \t {diff_rate:.7f}"


def _display_row_range(real_data, expect_data, start, row_range):
    """打印 [start + lo, start + hi) 区间的点位行（绝对序号 = start + idx + 1）。"""
    for idx in row_range:
        seq = start + idx + 1
        print_log(
            _format_loop_row(seq, expect_data[idx + start], real_data[idx + start])
        )


def display_output_np_isclose(real_data, expect_data, start, end):
    """输出每个点位的对比结果（≤20 点位全显示，>20 显示首尾各 10）。"""
    print_log(_ROW_DIVIDER)
    print_log(_LOOP_HEADER)
    print_log(_ROW_DIVIDER)
    split_count = int(end - start)
    if split_count <= 20:
        _display_row_range(real_data, expect_data, start, range(split_count + 1))
    else:
        _display_row_range(real_data, expect_data, start, range(10))
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        _display_row_range(
            real_data, expect_data, start, range(split_count - 10 + 1, split_count + 1)
        )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    """输出错误点位明细 + 最大相对误差点位。"""
    print_log(_ERROR_DIVIDER)
    print_log(_LOOP_HEADER)
    print_log(_ROW_DIVIDER)
    err_num = len(err_idx)
    for count, i in enumerate(err_idx, start=1):
        if count < 10 or (90 < count < 100):
            print_log(
                _format_error_row(
                    i, expect_data[i], real_data[i], relative_diff[count - 1]
                )
            )
        elif count == 10 or (count == 100 and err_num > 100):
            print_log(
                f"{'...':>8} \t {'...':>7} \t {'...':>7} \t {'...':>7} \t {'...':>7}"
            )
        elif count > 100:
            break

    print_log(_MAX_RE_DIVIDER)
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    for m_idx in m_idx_list[:4]:
        print_log(
            _format_error_row(m_idx, expect_data[m_idx], real_data[m_idx], max_error)
        )
    print_log(_ROW_DIVIDER)


def _print_verdict(rtol, atol, pct_thd, fulfill_percent, result_str):
    """打印 Rtol/Atol/PctThd/PctRlt/Result 汇总表格（check_result 两分支共用）。"""
    print_log(_ROW_DIVIDER)
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(_ROW_DIVIDER)
    print_log(
        f"{rtol:.4f}    \t {atol:.6f}  \t {pct_thd:.2f}%   \t {fulfill_percent:.6f}%   \t {result_str}"
    )


# ================================================================
#  单输出精度对比主函数
# ================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    """单输出精度对比，复用自正向 compressor_golden.check_result。

    Args:
        expect:    CPU golden 结果 (torch.Tensor)
        result:    NPU 反向算子输出 (torch.Tensor)
        data_type: dtype 字符串 ('bfloat16' / 'float16' / 'float32')
        pct_thd:   通过率阈值（默认 0.005 表示 99.5%）

    Returns:
        fulfill_percent: 满足精度要求的百分比
        result_str:      "Pass" / "Failed"
    """
    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()
    if real_data.size == 0 and data_compe.size == 0:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return 100.0, "Pass"
    max_error = 0
    result_str = "Failed"
    start, end = 0, max(real_data.size - 1, 0)

    if real_data.size != data_compe.size:
        print_log(
            f"Error,the size of npu output[{real_data.size}] and benchmark[{data_compe.size}] is not equal."
        )
        return 0.0, result_str
    overflows_count = (
        data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    )

    if overflows_count > 0:
        print_log(
            f"Overflow,size:{overflows_count},benchmark_output:"
            f"{data_compe[np.isinf(data_compe)][0:10]}, {data_compe[np.isnan(data_compe)][0:10]}"
        )

    # 检测 NPU 输出 (real_data) 中的 NaN/Inf；同时计算“两边同坏”（golden 与 NPU
    # 同位置都是 NaN 或都是 Inf）——同坏视为一致（输入语义一致时输出应一致），
    # 不判错；单边坏（NPU 坏但 golden 对应位置不坏）才是真实 bug，强制 Failed。
    real_data_f32 = (
        real_data.astype(np.float32)
        if str(real_data.dtype) == "bfloat16"
        else real_data
    )
    data_compe_f32 = (
        data_compe.astype(np.float32)
        if str(data_compe.dtype) == "bfloat16"
        else data_compe
    )
    real_nan_mask = np.isnan(real_data_f32)
    real_inf_mask = np.isinf(real_data_f32)
    real_overflow_mask = real_nan_mask | real_inf_mask
    real_overflow_count = int(real_overflow_mask.sum())
    both_bad = (np.isnan(real_data_f32) & np.isnan(data_compe_f32)) | (
        np.isinf(real_data_f32) & np.isinf(data_compe_f32)
    )
    unmatched_bad = real_overflow_mask & ~both_bad
    unmatched_bad_count = int(unmatched_bad.sum())
    if real_overflow_count > 0:
        print_log(
            f"NPU output has NaN/Inf, count:{real_overflow_count}, nan:{int(real_nan_mask.sum())}, "
            f"inf:{int(real_inf_mask.sum())}, sample_values:{real_data_f32[real_overflow_mask][0:10]}"
        )

    # 仅支持 bfloat16 / float16 / float32，阈值统一从 PRECISION 配置获取
    cfg = PRECISION[data_type]
    diff_thd = cfg["diff_thd"]
    max_diff_hd = cfg["max_diff_hd"]
    rtol = cfg["rtol"]
    atol = cfg["atol"]
    max_error_idx = 10000000

    split_count = int(end - start) + 1
    print_log(f"split_count:{float(split_count)}; max_diff_hd:{max_diff_hd};")

    # bfloat16 需转 float32 再 isclose；float16 / float32 直接 isclose
    # 注意：使用 equal_nan=False，使 NPU 输出中的 NaN 与 CPU 期望不匹配时判为错误；
    # 但两边同坏（both_bad：同位置都是 NaN 或都是 Inf）视为匹配
    if str(real_data.dtype) == "bfloat16":
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=False
        )
    diff_result = diff_result | both_bad
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if data_compe.dtype == np.bool_:
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = np.abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), np.abs(data_compe))
    b2 = (1.0 / (1 << 14)) / diff_thd
    b = np.maximum(b1, b2) + 1e-9
    err_diff = diff_abs / (b + 1e-9)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    # NPU 输出含 NaN/Inf 且 golden 对应位置不同样坏（unmatched）时判 Failed
    # （NaN 会传染，结果不可信）；两边同坏（both_bad）视为一致，不强制 Failed
    if unmatched_bad_count > 0:
        result_str = "Failed"
        max_error = float("inf")
        display_output_np_isclose(real_data, data_compe, start, end)
        pct_thd = (1 - pct_thd) * 100.0
        _print_verdict(rtol, atol, pct_thd, fulfill_percent, result_str)
        print_log(
            f"NPU output has NaN/Inf unmatched with golden (count={unmatched_bad_count}, "
            f"total nan/inf={real_overflow_count}). Force Failed."
        )
        if len(err_diff) > 0:
            display_error_output(
                real_data, data_compe, err_idx, err_diff[0:max_error_idx]
            )
        return fulfill_percent, result_str

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result_str = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        # 过滤 NaN/Inf 后再取 max，避免 NaN 导致 max_error 比较失效
        finite_err_diff = err_diff[np.isfinite(err_diff)]
        if len(finite_err_diff) > 0:
            max_error = float(np.max(finite_err_diff[0:max_error_idx]))
            if max_error >= max_diff_hd:
                result_str = "Failed"
    _print_verdict(rtol, atol, pct_thd, fulfill_percent, result_str)
    if len(err_diff) > 0:
        print_log(f"Max-RelativeError is: {max_error}. Threshold is: {max_diff_hd}.")
    if result_str == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return fulfill_percent, result_str


def check_one_output(
    name, expect, result, data_type, enabled, total_valid, pct_thd=0.005
):
    """对比单个反向输出，未开启对比的返回 SKIP 标记。

    Args:
        name:        输出名 (如 'd_ape', 'd_x', 'd_wkv', 'd_wgate')
        expect:      CPU golden 结果 (torch.Tensor)
        result:      NPU 反向算子输出 (torch.Tensor)
        data_type:   dtype 字符串 ('bfloat16' / 'float16' / 'float32')
        enabled:     是否进行精度对比
        total_valid: 该用例的有效压缩块数（0 时无有效数据，自动 SKIP）
        pct_thd:     通过率阈值

    Returns:
        dict: {diff, pct, status}
    """
    if not enabled:
        return {"diff": float("nan"), "pct": 100.0, "status": "SKIP"}

    print_log("=" * 80)
    print_log(f"check {name}:")
    print_log("--------------------------------------------------------------")
    try:
        if expect is None or result is None:
            print_log(f"{name}: expect or result is None, skip")
            return {"diff": float("nan"), "pct": 100.0, "status": "SKIP"}
        if expect.shape != result.shape:
            print_log(
                f"{name}: shape mismatch expect={expect.shape} result={result.shape}, flatten compare"
            )
        fulfill_percent, result_str = check_result(expect, result, data_type, pct_thd)
        diff = expect.cpu().float().flatten() - result.cpu().float().flatten()
        max_abs_diff = diff.abs().max().item() if diff.numel() > 0 else 0.0
        return {
            "diff": max_abs_diff,
            "pct": fulfill_percent,
            "status": "PASS" if result_str == "Pass" else "FAIL",
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"diff": float("nan"), "pct": 0.0, "status": "ERROR"}


# ================================================================
#  结果格式化与汇总统计
# ================================================================
def format_output_status(result, out_name):
    """格式化单个输出的状态字符串（正反向统一格式，用于用例行展示）。

    输出格式（正反向完全一致）：`<名>=<pct>%[<STATUS>]`；
    result 中不存在该输出的 status 字段（如通路 2/3 无正向字段）时返回 None，
    由调用方跳过，不参与展示。
    """
    pct_key, stat_key = get_output_keys(out_name)
    if stat_key not in result:
        return None
    st = result[stat_key]
    pct = result.get(pct_key, 100.0)
    if st == "SKIP":
        return f"{out_name}=SKIP"
    elif st == "ERROR":
        return f"{out_name}=ERROR"
    else:
        return f"{out_name}={pct:.2f}%[{st}]"


def format_case_line(result):
    """格式化单个用例的完整结果行（正反向统一；按 result 实际含有的输出展示）。"""
    parts = []
    for out_name in ALL_OUTPUTS:
        s = format_output_status(result, out_name)
        if s is not None:
            parts.append(s)
    detail = "  ".join(parts)
    return f"  {result.get('status', 'N/A'):5s}  {detail}  (validBlocks={result.get('totalValid', 0)})"


def build_error_result(name):
    """构建异常用例的 result dict，所有输出标记为 ERROR。"""
    err_result = dict(name=name, status="ERROR")
    for out_name in ALL_OUTPUTS:
        _, stat_key = get_output_keys(out_name)
        err_result[stat_key] = "ERROR"
    return err_result


def print_summary(results):
    """输出汇总统计：用例通过个数 + 每个用例每个输出的对比结果（以 case 为粒度）。

    输出两部分：
      1. SUMMARY: 总体 PASS/FAIL/SKIP/ERROR 个数统计
      2. Per-Case Per-Output Summary: 以 case 为粒度的表格，
         每行一个用例，每列一个输出，明确看出每个 case 的对比结果
    """
    print("\n" + "=" * 100)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    errors = sum(1 for r in results if r.get("status") == "ERROR")
    print(
        f"SUMMARY: {passed} PASS, {failed} FAIL, {skipped} SKIP, "
        f"{errors} ERROR out of {len(results)}"
    )
    print("=" * 100)

    # 以 case 为粒度的每个输出对比结果
    # 列由该批 result 实际含有的输出决定（数据驱动，不区分正反向）：
    # 通路 1 含正向字段 → 7 列；通路 2/3 仅反向 → 4 列
    cols = [o for o in ALL_OUTPUTS if any(get_output_keys(o)[1] in r for r in results)]
    print("\nPer-Case Per-Output Summary:")
    print("-" * 100)
    header = f"{'Case':<45}" + "".join(f"{o:<18}" for o in cols) + "Overall"
    print(header)
    print("-" * 100)
    for r in results:
        case_name = r.get("name", "unknown")
        row = f"{case_name:<45}"
        for out_name in cols:
            pct_key, stat_key = get_output_keys(out_name)
            st = r.get(stat_key, "SKIP")
            if st == "SKIP":
                row += f"{'SKIP':<18}"
            elif st == "ERROR":
                row += f"{'ERROR':<18}"
            else:
                pct = r.get(pct_key, 0.0)
                row += f"{pct:.2f}%[{st}]".ljust(18)
        row += r.get("status", "N/A")
        print(row)
    print("=" * 100)
