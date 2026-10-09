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

import numpy as np
import torch
import datetime
import os
import sys

RATIO_THRESHOLD = 0.005
FAIL_RATIO_LIMIT = 0.005


def get_tolerance(output_dtype):
    # bf16: atol=0.0001, rtol=0.0078125;  fp16(默认): atol=0.000025, rtol=0.005
    if output_dtype == torch.bfloat16:
        return 0.0001, 0.0078125
    return 0.000025, 0.005


def check_result_(
    test_name,
    expect,
    result,
    except_label="CPU",
    comp_label="NPU",
    verbose_diff=False,
    atol=0.000025,
    rtol=0.005,
):
    SEP = "─" * 64
    print(f"\n┌{SEP}┐")
    print(f"│  精度报告: {test_name}  ({except_label} vs {comp_label})")
    print(f"├{SEP}┤")
    if expect.shape != result.shape:
        print(
            f"│  [ERROR] shape不匹配: expect={tuple(expect.shape)}  {comp_label}={tuple(result.shape)}"
        )
        print(f"└{SEP}┘")
        return False

    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()

    if real_data.size != data_compe.size:
        print(
            f"│  [ERROR] size不匹配: expect={data_compe.size}  {comp_label}={real_data.size}"
        )
        print(f"└{SEP}┘")
        return False

    if real_data.size == 0:
        print("│  [INFO] 输出为空, 视为Pass")
        print(f"└{SEP}┘")
        return {
            "passed": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_rel": 0.0,
            "mean_rel": 0.0,
            "fail_cnt": 0,
            "total": 0,
            "fail_ratio": 0.0,
            "fulfill_percent": 100.0,
        }

    diff_thd = 0.005
    max_diff_hd = 10.0

    if str(real_data.dtype) == "bfloat16":
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    elif str(real_data.dtype) == "float8_e4m3fn":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    elif str(real_data.dtype) == "float8_e5m2":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        nan_pos_inf = np.isposinf(real_data)
        real_data[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(real_data)
        real_data[nan_neg_inf] = -57344
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        nan_pos_inf = np.isposinf(data_compe)
        data_compe[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(data_compe)
        data_compe[nan_neg_inf] = -57344
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )

    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)

    diff_abs = np.abs(data_compe - real_data)
    max_abs = float(diff_abs.max()) if diff_abs.size > 0 else 0.0
    mean_abs = float(diff_abs.mean()) if diff_abs.size > 0 else 0.0

    b1 = np.maximum(np.abs(real_data), np.abs(data_compe))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    rel_err = diff_abs / (b + eps)
    err_diff = rel_err[err_idx]
    max_rel = float(max(err_diff)) if len(err_diff) > 0 else 0.0
    mean_rel = float(rel_err.mean()) if rel_err.size > 0 else 0.0

    total = int(real_data.size)
    fail_cnt = int(err_idx.size)
    fulfill_percent = float(total - fail_cnt) / float(total) * 100.0
    fail_ratio = fail_cnt / total if total > 0 else 0.0

    pct_thd = (1 - diff_thd) * 100.0
    passed = fulfill_percent >= pct_thd
    if len(err_diff) > 0 and float(max(err_diff)) >= max_diff_hd:
        passed = False

    print(f"│  Shape       : {tuple(expect.shape)}")
    print(f"│  MaxAbsErr   : {max_abs:.8f}")
    print(f"│  MeanAbsErr  : {mean_abs:.8f}")
    print(f"│  MaxRelErr   : {max_rel:.8f}")
    print(f"│  MeanRelErr  : {mean_rel:.8f}")
    print(f"│  FailElems   : {fail_cnt} / {total}  ({fail_ratio * 100:.4f}%)")
    print(f"│  PctRlt      : {fulfill_percent:.6f}%")
    print(
        f"│  Threshold   : atol={atol}  rtol={rtol * 100:.2f}%  pctThd≥{pct_thd:.2f}%  maxRelErr<{max_diff_hd}"
    )
    print(f"│  结论        : {'✓ PASS' if passed else '✗ FAIL'}")
    if fail_cnt > 0:
        print(f"├{SEP}┤")
        if verbose_diff:
            print(f"│  全部 {len(err_idx)} 个超阈値元素 (rtol={rtol * 100:.2f}%):")
            print(
                f"│  {'idx':>10}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>12}"
            )
            for i in err_idx:
                print(
                    f"│  {int(i):>10}  {float(data_compe[i]):>+14.8f}  {float(real_data[i]):>+14.8f}"
                    f"  {float(diff_abs[i]):>12.8f}  {float(rel_err[i]):>12.6f}"
                )
        else:
            idxs = err_idx[:10]
            print(f"│  前{len(idxs)}个不通过元素:")
            print(
                f"│  {'idx':>8}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>10}"
            )
            for i in idxs:
                print(
                    f"│  {int(i):>8}  {float(data_compe[i]):>+14.8f}  {float(real_data[i]):>+14.8f}"
                    f"  {float(diff_abs[i]):>12.8f}  {float(rel_err[i]):>10.6f}"
                )
    print(f"└{SEP}┘")
    stats = {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "fail_cnt": fail_cnt,
        "total": total,
        "fail_ratio": fail_ratio,
        "fulfill_percent": fulfill_percent,
    }
    return stats


def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_fp32_data[j],
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )
            else:
                print_log(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f"
                    % (
                        start + idx + 1,
                        expect_data[j],
                        real_data[j],
                        diff_abs,
                        diff_rate,
                    )
                )

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i + 1,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx + 1,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy()
    data_compe = expect.cpu().numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return 100.0, "Pass"
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.size, data_compe.size)
        )
        return 0.0, result
    overflows_count = (
        data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    )

    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                data_compe[np.isinf(data_compe)][0:10],
                data_compe[np.isnan(data_compe)][0:10],
            )
        )

    if data_type == "bfloat16":
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.0078125
        atol = 0.0001
        max_error_idx = 10000000
    else:
        diff_thd = 0.005
        max_diff_hd = 10.0
        rtol = 0.005
        atol = 0.000025
        max_error_idx = 10000000

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    has_nan_inf = False
    if (
        "nan" in str(real_data)
        or "inf" in str(real_data)
        or "nan" in str(data_compe)
        or "inf" in str(data_compe)
    ):
        has_nan_inf = True

    if str(real_data.dtype) == "bfloat16":
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    elif str(real_data.dtype) == "float8_e4m3fn":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    elif str(real_data.dtype) == "float8_e5m2":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        nan_pos_inf = np.isposinf(real_data)
        real_data[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(real_data)
        real_data[nan_neg_inf] = -57344

        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        nan_pos_inf = np.isposinf(data_compe)
        data_compe[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(data_compe)
        data_compe[nan_neg_inf] = -57344

        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return fulfill_percent, result


def compare_results(cpu_result, npu_result, case_name):
    atol, rtol = get_tolerance(npu_result["attn_out"].dtype)
    print(
        f"cpu: attn_out shape {cpu_result['attn_out'].shape}, dtype {cpu_result['attn_out'].dtype}"
    )
    print(
        f"cpu: softmax_lse shape {cpu_result['softmax_lse'].shape}, dtype {cpu_result['softmax_lse'].dtype}"
    )
    print(
        f"npu: attn_out shape {npu_result['attn_out'].shape}, dtype {npu_result['attn_out'].dtype}"
    )
    print(
        f"npu: softmax_lse shape {npu_result['softmax_lse'].shape}, dtype {npu_result['softmax_lse'].dtype}"
    )
    cpu_out = cpu_result["attn_out"].to(torch.float32)
    npu_out = npu_result["attn_out"].to(torch.float32)

    attn_stats = check_result_(
        case_name,
        cpu_out,
        npu_out,
        except_label="CPU",
        comp_label="NPU",
        atol=atol,
        rtol=rtol,
    )

    lse_stats = {
        "passed": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "fail_cnt": 0,
        "total": 0,
        "fail_ratio": 0.0,
    }
    if (
        cpu_result.get("softmax_lse") is not None
        and npu_result.get("softmax_lse").numel() != 0
    ):
        cpu_lse = cpu_result["softmax_lse"].to(torch.float32)
        npu_lse = npu_result["softmax_lse"].to(torch.float32)
        lse_stats = check_result_(
            f"{case_name}_lse",
            cpu_lse,
            npu_lse,
            except_label="CPU",
            comp_label="NPU",
            atol=atol,
            rtol=rtol,
        )
        if cpu_lse.shape == npu_lse.shape and cpu_lse.numel() > 0:
            lse_stats = check_result_(
                f"{case_name}_lse",
                cpu_lse,
                npu_lse,
                except_label="CPU",
                comp_label="NPU",
            )

    return {"attn_out": attn_stats, "softmax_lse": lse_stats}
