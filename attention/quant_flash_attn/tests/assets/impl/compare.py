#!/usr/bin/python3
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

import logging
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod
import quant_flash_attn_hif8_golden as hif8_golden_mod

logger = logging.getLogger(__name__)


_EPS = 1e-10


def cal_relative_diff_torch(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + _EPS)


def display_output_torch(real_data, expect_data, start, end, expect_fp32_data=None):
    has_fp32 = expect_fp32_data is not None

    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info(
        "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        if has_fp32
        else "Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff"
    )
    logger.info(
        "---------------------------------------------------------------------------------------"
    )

    split_count = int(end - start)

    def display_row(i):
        j = start + i
        real_value = float(real_data[j])
        expect_value = float(expect_data[j])
        diff_rate = cal_relative_diff_torch(real_value, expect_value)
        idx = start + i + 1

        if not torch.isfinite(expect_data[j]).item():
            diff_abs = "inf" if torch.isinf(expect_data[j]).item() else "nan"
            if has_fp32:
                logger.info(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s",
                    idx,
                    float(expect_fp32_data[j]),
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
            else:
                logger.info(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s",
                    idx,
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
        else:
            diff_abs = abs(expect_value - real_value)
            if has_fp32:
                logger.info(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f",
                    idx,
                    float(expect_fp32_data[j]),
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
            else:
                logger.info(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f",
                    idx,
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )

    if split_count <= 20:
        for i in range(split_count + 1):
            display_row(i)
    else:
        for i in range(10):
            display_row(i)
        logger.info("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_row(i)


def display_error_output_torch(real_data, expect_data, err_idx, relative_diff):
    logger.info(
        "Error Line-----------------------------------------------------------------------------"
    )
    logger.info("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    logger.info(
        "---------------------------------------------------------------------------------------"
    )

    len_err = int(err_idx.numel())
    for count, i in enumerate(err_idx.tolist(), 1):
        if count < 10 or (90 < count < 100):
            expect_value = float(expect_data[i])
            real_value = float(real_data[i])
            logger.info(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f",
                i,
                expect_value,
                real_value,
                abs(expect_value - real_value),
                float(relative_diff[count - 1].item()),
            )
        elif count == 10 or (count == 100 and len_err > 100):
            logger.info(
                "%08s \t %07s \t %07s \t %07s \t %07s",
                "...",
                "...",
                "...",
                "...",
                "...",
            )
        elif count > 100:
            break

    logger.info(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = (
        float(torch.max(relative_diff).item()) if relative_diff.numel() > 0 else 0.0
    )
    m_idx_list = err_idx[relative_diff == max_error]
    for m_count, m_idx in enumerate(m_idx_list.tolist()):
        if m_count >= 3:
            break
        expect_value = float(expect_data[m_idx])
        real_value = float(real_data[m_idx])
        logger.info(
            "%08d \t %.7f \t %.7f \t %.7f \t %.7f",
            m_idx,
            expect_value,
            real_value,
            abs(expect_value - real_value),
            max_error,
        )
    logger.info(
        "---------------------------------------------------------------------------------------"
    )


def check_result(
    expect,
    npu_result,
    rtol=None,
    atol=None,
    diff_thd=None,
    pct_thd=None,
    max_diff_hd=None,
):
    # 5 个阈值均可经 CSV attributes['precision_tolerances'] 5-tuple 注入:
    #   (rtol, atol, diff_thd, pct_thd, max_diff_hd)
    # 未传入 (None) 时取默认值: diff_thd=0.005, pct_thd=0.005, max_diff_hd=10,
    # rtol/atol 按 dtype 分支 (bf16 0.0078125/0.0001, 否则 0.005/0.000025)
    _diff_thd_default = 0.005
    _pct_thd_default = 0.005
    _max_diff_hd_default = 10
    _rtol_default = 0.005
    _atol_default = 0.000025

    diff_thd = _diff_thd_default if diff_thd is None else diff_thd
    pct_thd = _pct_thd_default if pct_thd is None else pct_thd
    max_diff_hd = _max_diff_hd_default if max_diff_hd is None else max_diff_hd

    real_data = npu_result.detach().cpu().flatten()
    expect_data = (
        expect.detach().cpu().flatten()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect).flatten()
    )

    if real_data.numel() == 0:
        logger.info(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0

    if real_data.numel() != expect_data.numel():
        logger.info(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal.",
            real_data.numel(),
            expect_data.numel(),
        )
        return "Failed", 0.0, 0

    start = 0
    end = real_data.numel() - 1
    if end < start:
        end = start
    split_count = end - start + 1 if end != start else 1
    max_error = 0.0

    overflows_count = int(
        torch.isinf(expect_data).sum().item() + torch.isnan(expect_data).sum().item()
    )
    if overflows_count > 0:
        logger.info(
            "Overflow,size:%s,benchmark_output:%s, %s",
            overflows_count,
            expect_data[torch.isinf(expect_data)][0:10],
            expect_data[torch.isnan(expect_data)][0:10],
        )

    logger.info("split_count:%s; max_diff_hd:%s;", float(split_count), max_diff_hd)

    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    # 优先级：显式传入 rtol/atol > bf16 dtype 默认 > fp16 默认
    if rtol is None or atol is None:
        if npu_result.dtype == torch.bfloat16:
            _rtol_default = 0.0078125
            _atol_default = 0.0001
        rtol = _rtol_default if rtol is None else rtol
        atol = _atol_default if atol is None else atol

    diff_result = torch.isclose(
        real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    if expect_data.dtype == torch.bool:
        expect_data = expect_data.to(torch.int8)
        real_data = real_data.to(torch.int8)
        real_float = real_data.to(torch.float32)
        expect_float = expect_data.to(torch.float32)

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx]
    err_diff = torch.where(
        torch.isnan(err_diff) | torch.isinf(err_diff),
        torch.tensor(float("inf"), dtype=torch.float32),
        err_diff,
    )

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0

    display_output_torch(real_data, expect_data, start, end)

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s",
        rtol,
        atol,
        pct_thd,
        fulfill_percent,
        result,
    )

    if err_diff.numel() > 0:
        logger.info(
            "Max-RelativeError is: %s. Threshold is: %s.", max_error, max_diff_hd
        )

    if result == "Failed":
        display_error_output_torch(real_data, expect_data, err_idx, err_diff)

    return result, fulfill_percent, max_error


def _resolve_csv_tolerance(idx):
    """从 golden 模块读 csv 注入的 precision_tolerances / absolute_precision。

    ttk 框架不把 testcase 对象直接传给 custom compare, golden 插件在调用前把
    csv 的 precision_tolerances / absolute_precision 暂存到对应 golden 模块上。
    按 quant_mode 分流: mxfp8 → mxfp8_golden_mod, gqa_fp8 → fp8_golden_mod。
    本函数优先检查 fp8_golden_mod (quant_mode=6), 再检查 mxfp8_golden_mod,
    取首个非 None 的属性 (两个模块不会同时有值, 因一个 case 只走一条路径)。
    返回 (rtol, atol, diff_thd, pct_thd, max_diff_hd); csv 省略时返回 (None,)*5
    由 check_result 用默认值 (rtol/atol 按 dtype 分支, diff_thd=0.005, pct_thd=0.005,
    max_diff_hd=10)。

    precision_tolerances 格式: tuple of per-output tuple, 每条目可为 2-tuple 或 5-tuple
      2-tuple (向后兼容): (rtol, atol)                       如 ((0.0078125, 0.0001),)
      5-tuple (新):        (rtol, atol, diff_thd, pct_thd, max_diff_hd)
                          如 ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    5-tuple 的后三位 None → check_result 用默认值 (0.005/0.005/10)
    """
    # 优先检查 fp8_golden_mod (quant_mode=6 路径), 再检查 hif8_golden_mod, 再检查 mxfp8_golden_mod
    pt = getattr(fp8_golden_mod, "_csv_precision_tolerances", None)
    ap = getattr(fp8_golden_mod, "_csv_absolute_precision", None)
    if pt is None and ap is None:
        pt = getattr(hif8_golden_mod, "_csv_precision_tolerances", None)
        ap = getattr(hif8_golden_mod, "_csv_absolute_precision", None)
    if pt is None and ap is None:
        pt = getattr(mxfp8_golden_mod, "_csv_precision_tolerances", None)
        ap = getattr(mxfp8_golden_mod, "_csv_absolute_precision", None)
    rtol = None
    atol = None
    diff_thd = None
    pct_thd = None
    max_diff_hd = None
    if pt:
        # 优先取 idx 对应条目; idx 超出范围时回退到最后一个可用条目 (通常 idx=0)
        pair = pt[idx] if idx < len(pt) else pt[-1]
        if pair is not None and len(pair) >= 2:
            rtol = float(pair[0])
            atol = float(pair[1])
            # 5-tuple 的后三位 (向后兼容: 2-tuple 缺失 → None → check_result 用默认值)
            if len(pair) >= 3 and pair[2] is not None:
                diff_thd = float(pair[2])
            if len(pair) >= 4 and pair[3] is not None:
                pct_thd = float(pair[3])
            if len(pair) >= 5 and pair[4] is not None:
                max_diff_hd = float(pair[4])
    if atol is None and ap is not None:
        if isinstance(ap, (tuple, list)):
            val = ap[idx] if idx < len(ap) else ap[-1]
            if val is not None:
                atol = float(val)
        else:
            atol = float(ap)
    # replay 模式: pt 为 None (golden 未注入) → 用 bf16 默认, 与 CSV atten 容差一致
    if rtol is None and atol is None:
        rtol, atol = 0.0078125, 0.0001
    return rtol, atol, diff_thd, pct_thd, max_diff_hd


def _to_torch(x):
    """numpy array → torch tensor; torch tensor → 原样返回; None → None
    处理 bfloat16: numpy 原生不支持 bfloat16(ml_dtypes), torch.from_numpy 会失败,
    需要先 view uint8 再 reinterpret 为 bfloat16。
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        # bfloat16 在 numpy 里是 ml_dtypes.bfloat16, torch.from_numpy 不支持
        if x.dtype.name == "bfloat16":
            # 通过 uint8 view + torch view 还原 bfloat16
            return torch.from_numpy(x.view(np.uint8)).view(torch.bfloat16)
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def compare(*outputs, **kwargs):
    """对比 NPU 输出与 golden 输出。

    TTK 传参方式: [npu_out(, npu_lse), golden_out(, golden_lse)]
    当 lse 未启用时, npu_lse 可能是空数组(size=0), golden 不含 lse。
    所以输出数可能不等,需按实际非空输出配对。
    """
    # 过滤掉 None,但保留空数组用于后续检查
    non_none = [o for o in outputs if o is not None]

    if len(non_none) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"not enough valid outputs: {len(non_none)}",
        }

    # 检查是否都是空 tensor (actual_seq_q=[0] 时会出现)
    all_empty = all(
        (hasattr(o, "size") and o.size == 0) or (hasattr(o, "numel") and o.numel() == 0)
        for o in non_none
    )
    if all_empty:
        logger.info("[COMPARE] 所有输出都是空 tensor,视为 PASS")
        return {
            "pass": True,
            "precision": "100.0%",
            "error_info": None,
            "metrics": {"result": "Pass", "PctRlt": "100.0%", "MaxRE": 0.0},
        }

    # 过滤掉空数组,只保留有效输出
    valid = []
    for o in non_none:
        if hasattr(o, "size") and o.size == 0:
            continue
        if hasattr(o, "numel") and o.numel() == 0:
            continue
        valid.append(o)

    if len(valid) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"not enough valid outputs after filtering: {len(valid)}",
        }

    # 前半 NPU,后半 golden
    half = len(valid) // 2
    npu_outputs = valid[:half]
    golden_outputs = valid[half:]

    results = []

    for idx, (npu_out, golden_out) in enumerate(zip(npu_outputs, golden_outputs)):
        label = "atten" if idx == 0 else f"out{idx}"
        npu_torch = _to_torch(npu_out)
        golden_torch = _to_torch(golden_out)
        rtol, atol, diff_thd, pct_thd, max_diff_hd = _resolve_csv_tolerance(idx)
        logger.info(
            "[COMPARE] %s: csv rtol=%s atol=%s diff_thd=%s pct_thd=%s max_diff_hd=%s",
            label,
            rtol,
            atol,
            diff_thd,
            pct_thd,
            max_diff_hd,
        )
        npu_shape = tuple(npu_torch.shape) if hasattr(npu_torch, "shape") else "N/A"
        npu_dtype = str(npu_torch.dtype) if hasattr(npu_torch, "dtype") else "N/A"
        golden_shape = (
            tuple(golden_torch.shape) if hasattr(golden_torch, "shape") else "N/A"
        )
        golden_dtype = (
            str(golden_torch.dtype) if hasattr(golden_torch, "dtype") else "N/A"
        )
        logger.info(
            "[COMPARE] %s: npu_shape=%s npu_dtype=%s | golden_shape=%s golden_dtype=%s",
            label,
            npu_shape,
            npu_dtype,
            golden_shape,
            golden_dtype,
        )
        result, fulfill_percent, max_error = check_result(
            golden_torch,
            npu_torch,
            rtol=rtol,
            atol=atol,
            diff_thd=diff_thd,
            pct_thd=pct_thd,
            max_diff_hd=max_diff_hd,
        )
        is_pass = result == "Pass"
        results.append(
            {
                "pass": is_pass,
                "precision": fulfill_percent,
                "error_info": None
                if is_pass
                else f"{label}: result={result}, MaxRE={max_error}",
                "metrics": {
                    "label": label,
                    "result": result,
                    "PctRlt": fulfill_percent,
                    "MaxRE": max_error,
                },
            }
        )

    if len(results) == 1:
        return results[0]
    return results
