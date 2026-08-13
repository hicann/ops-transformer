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

"""Precision compare helpers for QuantCompressor TestSpec adapters.

Comparison logic is consistent with quant_compressor_golden.py check_result.
Compares 5 tensors: cmp_kv, kv_state_update, score_state_update,
kv_state_origin, score_state_origin.
"""

import numpy as np
import datetime
import os
import sys
import torch

_BF16_RTOL = 0.0078125
_BF16_ATOL = 0.0001
_FP16_RTOL = 0.005
_FP16_ATOL = 0.000025
_DIFF_THD = 0.005
_MAX_DIFF_HD = 10.0
_PCT_THD = 0.005
_MAX_ERROR_IDX = 10000000

_OUTPUT_NAMES = [
    "cmp_kv",
    "kv_state_update",
    "score_state_update",
    "kv_state_origin",
    "score_state_origin",
]

_BATCH_CONSISTENCY_CACHE = {}


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        value = value.numpy()
    return np.asarray(value)


def _get_thresholds(dtype_str):
    if dtype_str == "bfloat16":
        return _BF16_RTOL, _BF16_ATOL, _PCT_THD
    return _FP16_RTOL, _FP16_ATOL, _PCT_THD


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
        for index in range(split_count + 1):
            display_inner(index)
    else:
        for index in range(10):
            display_inner(index)
        print_log(".....   \t   .....   \t   .....   \t   .....    \t   .....")
        for index in range(split_count - 10 + 1, split_count + 1):
            display_inner(index)


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
    for index in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    index,
                    expect_data[index],
                    real_data[index],
                    abs(np.float64(expect_data[index]) - np.float64(real_data[index])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_6 = "......"
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_6, dot_6, dot_6, dot_6, dot_6)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_index in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_index,
                    expect_data[m_index],
                    real_data[m_index],
                    abs(
                        np.float64(expect_data[m_index])
                        - np.float64(real_data[m_index])
                    ),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


def _tensor_compare(npu_out, golden_out, name):
    npu = as_numpy(npu_out)
    golden = as_numpy(golden_out)

    real_data = npu.flatten().astype(np.float32)
    data_compe = golden.flatten().astype(np.float32)

    if real_data.size == 0 and real_data.size == data_compe.size:
        return {
            "pass": True,
            "precision": 100.0,
            "metrics": {
                "standard": "check_result",
                "name": name,
                "pct_rlt": 100.0,
                "max_rel_err": 0.0,
                "err_count": 0,
            },
            "error_info": None,
        }

    if real_data.size != data_compe.size:
        return {
            "pass": False,
            "precision": 0.0,
            "metrics": {
                "standard": "check_result",
                "name": name,
                "reason": "size mismatch",
                "npu_size": int(real_data.size),
                "golden_size": int(data_compe.size),
            },
            "error_info": f"{name} size mismatch: npu={real_data.size}, golden={data_compe.size}",
        }

    dtype_str = str(golden.dtype).split(".")[-1]
    rtol, atol, pct_thd = _get_thresholds(dtype_str)
    if (
        name == _OUTPUT_NAMES[3]
        or name == _OUTPUT_NAMES[4]
        or name == "batch_consisteny"
    ):
        rtol = 0
        atol = 0
        pct_thd = 0

    diff_result = np.isclose(
        real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = np.where(~diff_result)[0]

    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    diff_abs = np.abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), np.abs(data_compe))
    b2 = float((1.0 / (1 << 14)) / _DIFF_THD)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]
    max_rel_err = (
        float(np.max(err_diff[0:_MAX_ERROR_IDX])) if len(err_diff) > 0 else 0.0
    )

    print_log(
        f"---------------------------------------check {name}------------------------------------------------"
    )
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), _MAX_DIFF_HD))

    start = 0
    end = real_data.size - 1
    display_output_np_isclose(real_data, data_compe, start, end)
    is_pass = fulfill_percent >= (1 - pct_thd) * 100.0
    result = "pass" if is_pass else "Failed"

    if len(err_diff) > 0:
        max_error = max(err_diff[0:_MAX_ERROR_IDX])
        if max_error >= _MAX_DIFF_HD:
            result = "Failed"
            is_pass = False

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, (1 - pct_thd) * 100.0, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, _MAX_DIFF_HD)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:_MAX_ERROR_IDX])

    error_info = None
    if not is_pass:
        error_info = (
            f"{name} compare failed: fulfill_percent={fulfill_percent:.4f}%, "
            f"max_rel_err={max_rel_err:.4f}, err_count={int(err_idx.size)}"
        )

    return {
        "pass": is_pass,
        "precision": fulfill_percent,
        "diff_indices": err_idx[:1000].tolist() if len(err_idx) > 0 else [],
        "error_info": error_info,
        "metrics": {
            "standard": "check_result",
            "name": name,
            "rtol": rtol,
            "atol": atol,
            "pct_thd": (1 - pct_thd) * 100.0,
            "pct_rlt": fulfill_percent,
            "max_rel_err": max_rel_err,
            "max_diff_hd": _MAX_DIFF_HD,
            "err_count": int(err_idx.size),
        },
    }


def _batch_consistency_check(npu_cmp_kv, kwargs):
    """Compare NPU output slices across cases sharing the same batch consistency group.

    Two comparison modes:
    - External: different testcases, each with 1 slice. The first testcase
      registers a base; subsequent testcases with the same seed+axis compare.
    - Internal: a single testcase with 2+ slices. The first slice registers a
      base; subsequent slices in the same call compare.

    Cache key = "{seed}_{axis}" (extracted from kl, ignoring start/stop/step)
    so that different slice positions with the same seed are grouped together.

    In internal mode, cache entries are cleared before and after processing to
    prevent pollution of external comparison groups that share the same seed+axis.
    """
    batch_consistency_id = kwargs.get("batch_consistency_id")
    batch_axis = kwargs.get("batch_axis")
    batch_slice_info = kwargs.get("batch_slice_info")

    start_pos_list = kwargs.get("start_pos_list")
    cu_seqlens_list = kwargs.get("cu_seqlens_list")
    seqused_list = kwargs.get("seqused_list")
    cmp_ratio = kwargs.get("cmp_ratio")
    is_th = kwargs.get("is_th")

    if batch_consistency_id is None:
        return

    if npu_cmp_kv is None:
        return

    print("=========compare batch consistency============")
    npu_np = as_numpy(npu_cmp_kv)
    result = []

    # Pass 1: collect all valid (sl, kl, dl, axis_idx, slices_ref) tuples.
    # dl = "{seed}_{axis}" — the cache key that groups slices by seed+axis
    # regardless of start/stop position.
    valid_slices = []
    for axis_pos, slices, cache_key_idx in zip(
        batch_axis, batch_slice_info, batch_consistency_id
    ):
        if axis_pos is None or slices is None or cache_key_idx is None:
            continue
        for axis_idx, slices_idx, key_idx in zip(axis_pos, slices, cache_key_idx):
            if axis_idx is None or slices_idx is None or key_idx is None:
                continue
            for sl, kl in zip(slices_idx, key_idx):
                if sl is None or kl is None:
                    continue
                dl = "_".join(kl.split("_")[:2])
                valid_slices.append((sl, kl, dl, axis_idx, slices))

    if not valid_slices:
        return

    # Internal mode: 2+ slices in the same testcase call. Clear cache for
    # affected dl keys before processing so external bases don't pollute.
    is_internal = len(valid_slices) >= 2
    if is_internal:
        for _, _, dl, _, _ in valid_slices:
            if dl in _BATCH_CONSISTENCY_CACHE:
                del _BATCH_CONSISTENCY_CACHE[dl]

    # Pass 2: extract slice_output for each slice and compare via cache.
    slice_idx = 0
    for sl, kl, dl, axis_idx, slices_ref in valid_slices:
        start = sl[0]
        stop = sl[1]
        length = stop - start
        if is_th:
            b_idx = 0
            tc_idx = 0
            batch_size = len(cu_seqlens_list) - 1
            for b_idx in range(batch_size):
                if start >= cu_seqlens_list[b_idx + 1] - cu_seqlens_list[0]:
                    tc_idx += seqused_list[b_idx] // cmp_ratio
                    continue
                else:
                    start = start - cu_seqlens_list[b_idx]
                    stop = start + length
                    break
            headSize = cmp_ratio - (start + start_pos_list[b_idx]) % cmp_ratio
            compare_len = (stop - start - headSize % cmp_ratio) // cmp_ratio
            cache_len = cmp_ratio - start_pos_list[b_idx] % cmp_ratio
            start_idx = (
                (start - cache_len) // cmp_ratio + 1
                if headSize == cmp_ratio
                else (start + headSize + cmp_ratio - 1) // cmp_ratio
            )
            start_idx += tc_idx
            slice_output = npu_np[start_idx : (start_idx + compare_len), :]
        else:
            if axis_idx == 1:
                if slices_ref[0] is None:
                    bidx = 0
                else:
                    bidx = slices_ref[0][slice_idx][0]
                    slice_idx += 1
                headSize = cmp_ratio - (start + start_pos_list[0]) % cmp_ratio
                compare_len = (stop - start - headSize % cmp_ratio) // cmp_ratio
                cache_len = cmp_ratio - start_pos_list[0] % cmp_ratio
                start_idx = (
                    (start - cache_len) // cmp_ratio + 1
                    if headSize == cmp_ratio
                    else (start + headSize + cmp_ratio - 1) // cmp_ratio
                )
                slice_output = npu_np[bidx, start_idx : (start_idx + compare_len), :]
            else:
                slice_output = npu_np[start:stop, 1:, :]

        cache_key = dl
        if cache_key not in _BATCH_CONSISTENCY_CACHE:
            _BATCH_CONSISTENCY_CACHE[cache_key] = {
                "base": slice_output.copy(),
                "comparisons": [],
            }
            print(f"[batch_consistency]: base for id={kl} (group={dl})")
        else:
            cached = _BATCH_CONSISTENCY_CACHE[cache_key]
            compare_result = _tensor_compare(
                cached["base"], slice_output, "batch_consisteny"
            )
            result.append(compare_result)
            status = "Pass" if compare_result["pass"] else "Failed"
            cached["comparisons"].append(
                {
                    "status": status,
                    "batch_consistency_id": kl,
                }
            )
            print(f"[batch_consistency]  {status}(id={kl}, group={dl})")

    # Internal mode: clear cache after processing so this internal group's
    # base doesn't pollute subsequent external groups with the same seed+axis.
    if is_internal:
        for _, _, dl, _, _ in valid_slices:
            if dl in _BATCH_CONSISTENCY_CACHE:
                del _BATCH_CONSISTENCY_CACHE[dl]

    return result


def compare(*outputs, **kwargs):
    if len(outputs) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }

    GOLDEN_OUTPUT_COUNT = 2
    golden_outputs = list(outputs[-GOLDEN_OUTPUT_COUNT:])
    npu_outputs = list(outputs[:-GOLDEN_OUTPUT_COUNT])
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)

    # Fallback: single NPU output (cmp_kv only) — compare cmp_kv, skip state_cache
    if len(npu_outputs) == 1:
        results = [
            _tensor_compare(
                npu_outputs[0][cmp_kv_mask], golden_outputs[0][cmp_kv_mask], "cmp_kv"
            )
        ]
        for i in range(1, len(golden_outputs)):
            name = _OUTPUT_NAMES[i] if i < len(_OUTPUT_NAMES) else f"output_{i}"
            results.append(
                {
                    "pass": True,
                    "precision": "N/A",
                    "error_info": f"{name}: NPU state_cache not captured",
                }
            )
        result_consistency = _batch_consistency_check(npu_outputs[0], kwargs)
        if result_consistency is not None:
            results.append(result_consistency)
        return results

    npu_cmp_kv = npu_outputs[0]
    npu_state_cache = npu_outputs[1]
    cpu_cmp_kv = golden_outputs[0]
    cpu_state_cache = golden_outputs[1]

    update_kv = kwargs.get("update_kv", None)
    update_score = kwargs.get("update_score", None)

    npu_sub_outputs = [npu_cmp_kv[cmp_kv_mask]]
    golden_sub_outputs = [cpu_cmp_kv[cmp_kv_mask]]

    if update_kv is not None and update_score is not None:
        npu_sub_outputs.append(
            npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][update_kv]
        )
        npu_sub_outputs.append(
            npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][update_score]
        )
        npu_sub_outputs.append(
            npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][~update_kv]
        )
        npu_sub_outputs.append(
            npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][~update_score]
        )

        golden_sub_outputs.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][update_kv]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][update_score]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][~update_kv]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][~update_score]
        )
    else:
        npu_sub_outputs.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2])
        npu_sub_outputs.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :])
        npu_sub_outputs.append(None)
        npu_sub_outputs.append(None)

        golden_sub_outputs.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :]
        )
        golden_sub_outputs.append(None)
        golden_sub_outputs.append(None)

    results = []
    for idx in range(len(golden_sub_outputs)):
        name = _OUTPUT_NAMES[idx]
        npu_out = npu_sub_outputs[idx]
        golden_out = golden_sub_outputs[idx]
        if npu_out is None and golden_out is None:
            results.append(
                {
                    "pass": True,
                    "precision": "N/A",
                    "error_info": f"{name} is None",
                }
            )
        else:
            results.append(_tensor_compare(npu_out, golden_out, name))

    result_consistency = _batch_consistency_check(npu_outputs[0], kwargs)
    if result_consistency is not None:
        results.append(result_consistency)

    return results


def _to_torch(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        return val
    if isinstance(val, np.ndarray):
        if val.dtype == np.dtype("hifloat8"):
            return torch.from_numpy(val.view(np.uint8))
        if val.dtype.itemsize == 2 and str(val.dtype) == "bfloat16":
            return torch.from_numpy(val.view(np.uint16)).view(torch.bfloat16)
        return torch.from_numpy(val)
    return val


def compare_aclnn(*outputs, **kwargs):
    outputs = tuple(_to_torch(o) for o in outputs)
    if len(outputs) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }

    GOLDEN_OUTPUT_COUNT = 2
    golden_outputs = list(outputs[-GOLDEN_OUTPUT_COUNT:])
    npu_outputs = list(outputs[:-GOLDEN_OUTPUT_COUNT])
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)

    # Fallback: single NPU output (cmp_kv only) — compare cmp_kv, skip state_cache
    if len(npu_outputs) == 1:
        results = [
            _tensor_compare(
                npu_outputs[0][cmp_kv_mask].to(torch.float32),
                golden_outputs[1][cmp_kv_mask].to(torch.float32),
                "cmp_kv",
            )
        ]
        for i in range(1, len(golden_outputs)):
            name = _OUTPUT_NAMES[i] if i < len(_OUTPUT_NAMES) else f"output_{i}"
            results.append(
                {
                    "pass": True,
                    "precision": "N/A",
                    "error_info": f"{name}: NPU state_cache not captured",
                }
            )
        result_consistency = _batch_consistency_check(npu_outputs[0], kwargs)
        if result_consistency is not None:
            results.append(result_consistency)
        return results

    npu_state_cache = npu_outputs[0].to(torch.float32)
    npu_cmp_kv = npu_outputs[1].to(torch.float32)
    # aclnn golden return order aligns with output_tensor_indexes=(3,12):
    #   golden[0] = state_cache (idx 3), golden[1] = cmp_kv (idx 12)
    cpu_state_cache = golden_outputs[0].to(torch.float32)
    cpu_cmp_kv = golden_outputs[1].to(torch.float32)

    update_kv = kwargs.get("update_kv", None)
    update_score = kwargs.get("update_score", None)

    npu_sub_outputs = [npu_cmp_kv[cmp_kv_mask]]
    golden_sub_outputs = [cpu_cmp_kv[cmp_kv_mask]]

    if update_kv is not None and update_score is not None:
        npu_sub_outputs.append(
            npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][update_kv]
        )
        npu_sub_outputs.append(
            npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][update_score]
        )
        npu_sub_outputs.append(
            npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][~update_kv]
        )
        npu_sub_outputs.append(
            npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][~update_score]
        )

        golden_sub_outputs.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][update_kv]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][update_score]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][~update_kv]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][~update_score]
        )
    else:
        npu_sub_outputs.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2])
        npu_sub_outputs.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :])
        npu_sub_outputs.append(None)
        npu_sub_outputs.append(None)

        golden_sub_outputs.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2]
        )
        golden_sub_outputs.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :]
        )
        golden_sub_outputs.append(None)
        golden_sub_outputs.append(None)

    results = []
    for idx in range(len(golden_sub_outputs)):
        name = _OUTPUT_NAMES[idx]
        npu_out = npu_sub_outputs[idx]
        golden_out = golden_sub_outputs[idx]
        if npu_out is None and golden_out is None:
            results.append(
                {
                    "pass": True,
                    "precision": "N/A",
                    "error_info": f"{name} is None",
                }
            )
        else:
            results.append(_tensor_compare(npu_out, golden_out, name))

    # aclnn NPU output order is (state_cache, cmp_kv); batch consistency
    # compares cmp_kv across cases — must pass cmp_kv (npu_outputs[1]),
    # not state_cache (npu_outputs[0]).
    result_consistency = _batch_consistency_check(npu_cmp_kv, kwargs)
    if result_consistency is not None:
        results.append(result_consistency)

    return results
