# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import datetime
import os
import sys

import torch

from qli_test_utils.result_compare import compare_qli_outputs
from qli_test_utils.result_compare import VALUE_ATOL, VALUE_RTOL


def _percent(report):
    rows = set(report["sparse"].get("mismatched_rows", []))
    rows.update(report["candidate"].get("mismatched_rows", []))
    return 100.0 if not rows else 0.0


def _row_count(result):
    indices = result["sparse_indices"]
    return int(indices.numel() // indices.shape[-1])


def _print_report(report, npu_result):
    sparse = report["sparse"]
    candidate = report["candidate"]
    print(f"total_line is {_row_count(npu_result)}")
    differing = sorted(set(sparse.get("mismatched_rows", [])))
    differing.extend(
        row for row in candidate.get("mismatched_rows", []) if row not in differing
    )
    if differing:
        print(f"需要进行第二步比较的batch有{len(differing)}")
    else:
        print("有效值集合相同，无需进行比较")
    if report["pass"]:
        print("[success]TopK精度通过, idx不同的地方的value误差在阈值之内")
    else:
        print("[fail]TopK精度失败")
    print(f"npu_pass is {report['pass']}")


def _print_log(message):
    print(
        "[%s] [INFO]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            message,
        )
    )


def _print_index_precision(cpu_result, npu_result, report):
    expected = cpu_result["sparse_indices"].detach().cpu()
    actual = npu_result["sparse_indices"].detach().cpu()
    rtol = VALUE_RTOL
    atol = VALUE_ATOL
    expected = expected.float().reshape(-1)
    actual = actual.float().reshape(-1)
    close = actual == expected
    fulfill = 100.0 if close.numel() == 0 else float(close.float().mean()) * 100.0
    result = bool(report["pass"])
    bad = torch.nonzero(~close, as_tuple=False).flatten()
    source_indices = bad.tolist() if bad.numel() else list(range(expected.numel()))
    if bad.numel():
        _print_log(
            "Error Line-----------------------------------------------------------------------------"
        )
    else:
        _print_log(
            "Output Sample--------------------------------------------------------------------------"
        )
    _print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    display_entries = [(index, False) for index in source_indices[:9]]
    if len(source_indices) > 90:
        display_entries.append((None, True))
        display_entries.extend((index, False) for index in source_indices[90:99])
    for index, is_ellipsis in display_entries:
        if is_ellipsis:
            _print_log("...      \t ...       \t ...       \t ...       \t ...")
            continue
        expect = float(expected[index])
        real = float(actual[index])
        diff = abs(real - expect)
        rate = diff / (abs(expect) + 1.0e-9)
        _print_log(
            f"{index:08d} \t {expect:.7f} \t {real:.7f} \t {diff:.7f} \t {rate:.7f}"
        )
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    _print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    _print_log(
        "---------------------------------------------------------------------------------------"
    )
    _print_log(
        f"{rtol:.4f}    \t {atol:.6f} \t "
        f"95.00%   \t "
        f"{fulfill:.6f}%   \t {'Pass' if result else 'Failed'}"
    )
    if expected.numel():
        relative = torch.abs(actual - expected) / (torch.abs(expected) + 1.0e-9)
        max_relative = float(relative.max())
    else:
        max_relative = 0.0
    _print_log(f"Max-RelativeError is: {max_relative}. Threshold is: {rtol}.")


def check_result(
    cpu_result,
    npu_result,
    topk_value=None,
    output_idx_offset=None,
    params=None,
    cpu_topk_value=None,
    npu_topk_value=None,
):
    del topk_value, output_idx_offset, params, cpu_topk_value, npu_topk_value
    report = compare_qli_outputs(cpu_result, npu_result)
    _print_report(report, npu_result)
    _print_index_precision(cpu_result, npu_result, report)
    return ("Pass" if report["pass"] else "Failed", _percent(report))


def check_result_return_value(
    cpu_topk_value,
    npu_topk_value,
    params,
    cpu_result,
    npu_result,
    topk_value=None,
    output_idx_offset=None,
):
    del cpu_topk_value, npu_topk_value, params, topk_value, output_idx_offset
    report = compare_qli_outputs(cpu_result, npu_result)
    return ("Pass" if report["pass"] else "Failed", _percent(report))
