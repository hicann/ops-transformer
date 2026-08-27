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

import logging
import math
import os
import torch

logger = logging.getLogger(__name__)

_EPS = 1e-10
_DEFAULT_RTOL = 0.005
_DEFAULT_ATOL = 0.000025
_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_TABLE_WIDTH = 90
_BASE_HEADER = "{:>10} {:>18} {:>18} {:>18} {:>14}"
_BASE_ROW = "{:>10} {:>18} {:>18} {:>18} {:>14}"
_FP32_HEADER = "{:>10} {:>18} {:>18} {:>18} {:>18} {:>14}"
_FP32_ROW = "{:>10} {:>18} {:>18} {:>18} {:>18} {:>14}"


def cal_relative_diff_torch(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + _EPS)


def _format_float(value):
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return str(float(value))
    return f"{float(value):.7e}"


def _display_offsets(total_count):
    if total_count <= 20:
        return list(range(total_count))
    return list(range(10)) + [None] + list(range(total_count - 10, total_count))


def get_compare_tolerance(npu_result):
    """返回与 check_result 完全一致的逐元素 rtol/atol。"""
    if isinstance(npu_result, torch.Tensor) and npu_result.dtype == torch.bfloat16:
        return _BF16_RTOL, _BF16_ATOL
    return _DEFAULT_RTOL, _DEFAULT_ATOL


def display_output_torch(real_data, expect_data, start, end, expect_fp32_data=None):
    has_fp32 = expect_fp32_data is not None

    logger.info("-" * (_TABLE_WIDTH + (22 if has_fp32 else 0)))
    if has_fp32:
        logger.info(
            _FP32_HEADER.format(
                "Loop", "ExpFP32Out", "ExpectOut", "RealOut", "AbsDiff", "RateDiff"
            )
        )
    else:
        logger.info(
            _BASE_HEADER.format("Loop", "ExpectOut", "RealOut", "AbsDiff", "RateDiff")
        )
    logger.info("-" * (_TABLE_WIDTH + (22 if has_fp32 else 0)))

    total_count = int(end - start + 1)

    def display_row(offset):
        j = start + offset
        real_value = float(real_data[j])
        expect_value = float(expect_data[j])
        diff_rate = cal_relative_diff_torch(real_value, expect_value)
        idx = j + 1

        if not torch.isfinite(expect_data[j]).item():
            diff_abs = "inf" if torch.isinf(expect_data[j]).item() else "nan"
        else:
            diff_abs = abs(expect_value - real_value)

        if has_fp32:
            logger.info(
                _FP32_ROW.format(
                    idx,
                    _format_float(float(expect_fp32_data[j])),
                    _format_float(expect_value),
                    _format_float(real_value),
                    _format_float(diff_abs),
                    _format_float(diff_rate),
                )
            )
        else:
            logger.info(
                _BASE_ROW.format(
                    idx,
                    _format_float(expect_value),
                    _format_float(real_value),
                    _format_float(diff_abs),
                    _format_float(diff_rate),
                )
            )

    for offset in _display_offsets(total_count):
        if offset is None:
            logger.info(
                _FP32_ROW.format("...", "...", "...", "...", "...", "...")
                if has_fp32
                else _BASE_ROW.format("...", "...", "...", "...", "...")
            )
            continue
        display_row(offset)


def display_error_output_torch(real_data, expect_data, err_idx, relative_diff):
    logger.info("Error Line" + "-" * (_TABLE_WIDTH - len("Error Line")))
    logger.info(
        _BASE_HEADER.format("Loop", "ExpectOut", "RealOut", "AbsDiff", "RateDiff")
    )
    logger.info("-" * _TABLE_WIDTH)

    len_err = int(err_idx.numel())
    for offset in _display_offsets(len_err):
        if offset is None:
            logger.info(_BASE_ROW.format("...", "...", "...", "...", "..."))
            continue
        i = int(err_idx[offset].item())
        expect_value = float(expect_data[i])
        real_value = float(real_data[i])
        logger.info(
            _BASE_ROW.format(
                i,
                _format_float(expect_value),
                _format_float(real_value),
                _format_float(abs(expect_value - real_value)),
                _format_float(float(relative_diff[offset].item())),
            )
        )

    logger.info("Max-RE line:" + "-" * (_TABLE_WIDTH - len("Max-RE line:")))
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
            _BASE_ROW.format(
                m_idx,
                _format_float(expect_value),
                _format_float(real_value),
                _format_float(abs(expect_value - real_value)),
                _format_float(max_error),
            )
        )
    logger.info("-" * _TABLE_WIDTH)


def check_result(expect, npu_result, debug=False):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol, atol = get_compare_tolerance(npu_result)

    real_data = npu_result.detach().cpu().flatten()
    expect_data = (
        expect.detach().cpu().flatten()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect).flatten()
    )

    if real_data.numel() != expect_data.numel():
        logger.info(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal.",
            real_data.numel(),
            expect_data.numel(),
        )
        return "Failed", 0.0, 0

    if real_data.numel() == 0:
        logger.info(
            'The npu_output and benchmark are both [], the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0

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

    if expect_data.dtype == torch.bool:
        expect_data = expect_data.to(torch.int8)
        real_data = real_data.to(torch.int8)

    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    diff_result = torch.isclose(
        real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx]
    finite_err_diff = err_diff[torch.isfinite(err_diff)]

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0

    display_output_torch(real_data, expect_data, start, end)

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    if finite_err_diff.numel() > 0:
        max_error = float(torch.max(finite_err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info("-" * _TABLE_WIDTH)
    logger.info(
        "{:>10} {:>12} {:>12} {:>14} {:>10}".format(
            "Rtol", "Atol", "PctThd", "PctRlt", "Result"
        )
    )
    logger.info("-" * _TABLE_WIDTH)
    logger.info(
        "{:>10.4f} {:>12.6f} {:>11.2f}% {:>13.6f}% {:>10}".format(
            rtol, atol, pct_thd, fulfill_percent, result
        )
    )

    if finite_err_diff.numel() > 0:
        logger.info(
            "Max-RelativeError is: %s. Threshold is: %s.", max_error, max_diff_hd
        )

    if err_diff.numel() > 0 and (result == "Failed" or debug):
        display_error_output_torch(real_data, expect_data, err_idx, err_diff)

    return result, fulfill_percent, max_error


def save_precision_map(
    expect,
    npu_result,
    output_path,
    tensor_name,
    compare_result=None,
    max_display_rows=2048,
    max_display_cols=2048,
):
    """生成白色通过、蓝色失败的逐元素精度分布图；大 Tensor 按区域聚合显示。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError("--debug precision PNG requires matplotlib") from error

    expect_tensor = (
        expect.detach().cpu()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect)
    )
    npu_tensor = npu_result.detach().cpu()
    expect_shape = tuple(expect_tensor.shape)
    npu_shape = tuple(npu_tensor.shape)
    rtol, atol = get_compare_tolerance(npu_result)
    expect_flat = expect_tensor.flatten()
    npu_flat = npu_tensor.flatten()
    element_count_mismatch = expect_flat.numel() != npu_flat.numel()

    if element_count_mismatch:
        display_mask = torch.ones((1, 1), dtype=torch.uint8)
        logical_rows = logical_cols = 1
        fail_count = max(expect_flat.numel(), npu_flat.numel())
        total_count = fail_count
    else:
        if expect_flat.dtype == torch.bool:
            expect_flat = expect_flat.to(torch.int8)
            npu_flat = npu_flat.to(torch.int8)
        total_count = int(expect_flat.numel())
        logical_cols = (
            int(expect_shape[-1])
            if expect_shape and expect_shape[-1] > 0
            else max(total_count, 1)
        )
        logical_rows = max(1, math.ceil(total_count / logical_cols))
        display_rows = min(logical_rows, max_display_rows)
        display_cols = min(logical_cols, max_display_cols)
        display_mask = torch.zeros((display_rows, display_cols), dtype=torch.uint8)
        fail_count = 0

        # 按最终图片的一行分块比较，避免 MIX 大用例额外构造完整 FP32 失败矩阵。
        for display_row in range(display_rows):
            source_row_begin = display_row * logical_rows // display_rows
            source_row_end = (display_row + 1) * logical_rows // display_rows
            flat_begin = source_row_begin * logical_cols
            flat_end = min(source_row_end * logical_cols, total_count)
            if flat_begin >= flat_end:
                continue

            close_mask = torch.isclose(
                npu_flat[flat_begin:flat_end].to(torch.float32),
                expect_flat[flat_begin:flat_end].to(torch.float32),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            fail_chunk = ~close_mask
            fail_count += int(fail_chunk.sum().item())

            source_row_count = source_row_end - source_row_begin
            padded_count = source_row_count * logical_cols
            if fail_chunk.numel() < padded_count:
                fail_chunk = torch.nn.functional.pad(
                    fail_chunk, (0, padded_count - fail_chunk.numel()), value=False
                )
            fail_columns = fail_chunk.reshape(source_row_count, logical_cols).any(dim=0)
            if display_cols != logical_cols:
                fail_columns = torch.nn.functional.adaptive_max_pool1d(
                    fail_columns.to(torch.float32).reshape(1, 1, logical_cols),
                    display_cols,
                )[0, 0].to(torch.bool)
            display_mask[display_row] = fail_columns.to(torch.uint8)

    display_rows, display_cols = display_mask.shape
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    figure_width = min(18.0, max(8.0, display_cols / max(display_rows, 1) * 10.0))
    figure_height = min(
        14.0, max(5.0, display_rows / max(display_cols, 1) * figure_width)
    )
    fig, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.imshow(
        display_mask.numpy(),
        cmap=ListedColormap(["#FFFFFF", "#2563EB"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
        extent=(0, logical_cols, logical_rows, 0),
    )
    axis.set_xlabel(f"logical column (size={logical_cols})")
    axis.set_ylabel(f"logical row (size={logical_rows})")

    status = compare_result[0] if compare_result is not None else "N/A"
    pass_count = max(total_count - fail_count, 0)
    title = (
        f"{tensor_name} precision map | result={status}\n"
        f"golden_shape={expect_shape}, operator_shape={npu_shape}\n"
        f"logical={logical_rows}x{logical_cols}, display={display_rows}x{display_cols}, "
        f"elements={total_count}, pass={pass_count}, fail={fail_count}, rtol={rtol:g}, atol={atol:g}"
    )
    if element_count_mismatch:
        title += "\nelement-count mismatch"
    elif expect_shape != npu_shape:
        title += "\nshape differs; flattened elements use the normal precision criteria"
    elif display_rows != logical_rows or display_cols != logical_cols:
        title += "\naggregated view: a pixel is blue when its source region contains at least one failed element"
    axis.set_title(title, fontsize=10)
    axis.legend(
        handles=[
            Patch(facecolor="#FFFFFF", edgecolor="#808080", label="PASS"),
            Patch(facecolor="#2563EB", label="FAIL"),
        ],
        loc="upper right",
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("[DEBUG] precision map saved: %s", output_path)
    return {
        "golden_shape": expect_shape,
        "operator_shape": npu_shape,
        "logical_size": (logical_rows, logical_cols),
        "display_size": (display_rows, display_cols),
        "total": total_count,
        "fail": fail_count,
        "rtol": rtol,
        "atol": atol,
    }
