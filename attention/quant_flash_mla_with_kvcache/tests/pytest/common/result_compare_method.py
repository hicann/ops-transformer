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
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    force=True,
)
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
        if count <= 200:
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
        elif count == 201:
            logger.info(
                "%08s \t %07s \t %07s \t %07s \t %07s",
                "...",
                "(truncated, total errors: {})".format(len_err),
                "...",
                "...",
                "...",
            )
        else:
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


def check_result(expect, npu_result):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025

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

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001

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
        logger.info("Total error count: %d / %d", int(err_idx.numel()), split_count)
        logger.info(
            "Real NaN count: %d, Inf count: %d",
            int(torch.isnan(real_float).sum().item()),
            int(torch.isinf(real_float).sum().item()),
        )
        logger.info(
            "Expect NaN count: %d, Inf count: %d",
            int(torch.isnan(expect_float).sum().item()),
            int(torch.isinf(expect_float).sum().item()),
        )

    return result, fulfill_percent, max_error


def check_result_device(expect, actual, chunk_size=4 * 1024 * 1024):
    """Compare in bounded chunks on actual.device; transfer only scalar statistics.

    Preserve QMLA's existing BF16/FP32 tolerances, 99.5% pass rate and maximum
    error threshold. NaNs and mismatched infinities always fail.
    """
    if expect.shape != actual.shape:
        logger.error(
            "Shape mismatch: expected=%s actual=%s", expect.shape, actual.shape
        )
        return "Failed", 0.0, float("inf")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rtol, atol = (
        (0.0078125, 0.0001) if actual.dtype == torch.bfloat16 else (0.005, 0.000025)
    )
    expected = expect.detach().reshape(-1)
    observed = actual.detach().reshape(-1)
    bad_count, max_error, invalid = 0, 0.0, False
    for start in range(0, observed.numel(), chunk_size):
        real = observed[start : start + chunk_size].float()
        ref = expected[start : start + chunk_size].to(
            device=actual.device, dtype=torch.float32
        )
        finite = torch.isfinite(real) & torch.isfinite(ref)
        invalid |= bool(
            (
                torch.isnan(real)
                | torch.isnan(ref)
                | ((torch.isinf(real) | torch.isinf(ref)) & (real != ref))
            )
            .any()
            .item()
        )
        close = torch.isclose(real, ref, rtol=rtol, atol=atol, equal_nan=False)
        bad = ~close
        bad_count += int(bad.sum().item())
        denom = (
            torch.maximum(real.abs(), ref.abs()).clamp_min((1.0 / (1 << 14)) / 0.005)
            + _EPS
        )
        errors = torch.where(bad & finite, (real - ref).abs() / denom, 0.0)
        max_error = max(max_error, float(errors.max().item()))
    count = observed.numel()
    percent = 100.0 * (count - bad_count) / count if count else 100.0
    status = "Pass" if percent >= 99.5 and max_error < 10 and not invalid else "Failed"
    logger.info(
        "[Compare] device=%s count=%d bad=%d pass=%.6f%% max_error=%g invalid=%s result=%s",
        actual.device,
        count,
        bad_count,
        percent,
        max_error,
        invalid,
        status,
    )
    return status, percent, max_error
