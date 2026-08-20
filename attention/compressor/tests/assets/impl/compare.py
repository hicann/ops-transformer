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

import gc
import importlib.util
from pathlib import Path

import numpy as np
import torch

try:
    from ttk.utilities.container_utils import get_global_storage
except Exception:
    get_global_storage = None

try:
    from ttk.core_modules.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.comparison.resolve import resolve_tolerance

    _TTK_CROSS_CHECK_AVAILABLE = True
except Exception:
    _TTK_CROSS_CHECK_AVAILABLE = False

_PYTEST_GOLDEN_MODULE = None

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
    "softmax_score",
    "kv",
]

_BATCH_CONSISTENCY_CACHE = {}


def _load_pytest_golden_module():
    global _PYTEST_GOLDEN_MODULE
    if _PYTEST_GOLDEN_MODULE is not None:
        return _PYTEST_GOLDEN_MODULE
    golden_path = Path(__file__).resolve().parent / "golden.py"
    spec = importlib.util.spec_from_file_location(
        f"compressor_assets_golden_ref_{abs(hash(golden_path))}", golden_path
    )
    golden_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden_mod)
    _PYTEST_GOLDEN_MODULE = golden_mod.load_pytest_golden_module()
    return _PYTEST_GOLDEN_MODULE


_pytest_golden = _load_pytest_golden_module()
display_output_np_isclose = _pytest_golden.display_output_np_isclose
display_error_output = _pytest_golden.display_error_output
cal_relative_diff_np_isclose = _pytest_golden.cal_relative_diff_np_isclose
print_log = _pytest_golden.print_log


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().type(torch.float32).numpy()
    return np.asarray(value)


def _to_torch(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    arr = np.asarray(value)
    dtype_str = str(arr.dtype)
    if dtype_str == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    if arr.dtype == np.float16:
        return torch.from_numpy(arr)
    if arr.dtype == np.float32:
        return torch.from_numpy(arr)
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    if arr.dtype == np.uint8:
        return torch.from_numpy(arr)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _get_thresholds(dtype_str):
    if dtype_str == "bfloat16":
        return _BF16_RTOL, _BF16_ATOL, _PCT_THD
    return _FP16_RTOL, _FP16_ATOL, _PCT_THD


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
        or name == "batch_consistency"
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
    err_count = int(err_idx.size)
    diff_indices = err_idx[:1000].tolist() if len(err_idx) > 0 else []
    del real_data, data_compe, diff_result, err_idx, diff_abs
    gc.collect()
    return {
        "pass": is_pass,
        "precision": fulfill_percent,
        "diff_indices": diff_indices,
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
            "err_count": err_count,
        },
    }


def _batch_consistency_check(npu_cmp_kv, kwargs):
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
    cache_key = batch_consistency_id
    result = []
    slice_idx = 0
    for axis_pos, slices, cache_key_idx in zip(batch_axis, batch_slice_info, cache_key):
        if axis_pos is None or slices is None or cache_key_idx is None:
            continue
        for axis_idx, slices_idx, key_idx in zip(axis_pos, slices, cache_key_idx):
            if axis_idx is None or slices_idx is None or key_idx is None:
                continue
            for sl, kl in zip(slices_idx, key_idx):
                if sl is None or kl is None:
                    continue
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
                        if slices[0] is None:
                            bidx = 0
                        else:
                            bidx = slices[0][slice_idx][0]
                            slice_idx += 1
                        headSize = (
                            cmp_ratio - (start + start_pos_list[bidx]) % cmp_ratio
                        )
                        compare_len = (stop - start - headSize % cmp_ratio) // cmp_ratio
                        cache_len = cmp_ratio - start_pos_list[bidx] % cmp_ratio
                        start_idx = (
                            (start - cache_len) // cmp_ratio + 1
                            if headSize == cmp_ratio
                            else (start + headSize + cmp_ratio - 1) // cmp_ratio
                        )
                        slice_output = npu_np[
                            bidx, start_idx : (start_idx + compare_len), :
                        ]
                    else:
                        slice_output = npu_np[start:stop, 1:, :]
                dl = "_".join(kl.split("_")[:2])
                if dl not in _BATCH_CONSISTENCY_CACHE:
                    _BATCH_CONSISTENCY_CACHE[dl] = {
                        "base": slice_output.copy(),
                        "comparisons": [],
                    }
                    print(f"[batch_consistency]: base for id={kl}")
                else:
                    cached = _BATCH_CONSISTENCY_CACHE[dl]
                    compare_result = _tensor_compare(
                        cached["base"], slice_output, "batch_consistency"
                    )
                    result.append(compare_result)
                    status = "Pass" if compare_result["pass"] else "Failed"
                    cached["comparisons"].append(
                        {
                            "status": status,
                            "batch_consistency_id": kl,
                        }
                    )
                    print(f"[batch_consistency]  {status}(id={kl})")

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
        gc.collect()
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
    del npu_cmp_kv, npu_state_cache, cpu_cmp_kv, cpu_state_cache
    del npu_sub_outputs, golden_sub_outputs
    gc.collect()
    return results


def compare_aclnn(*outputs, **kwargs):
    if len(outputs) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }

    GOLDEN_OUTPUT_COUNT = 4
    golden_outputs = [_to_torch(g) for g in outputs[-GOLDEN_OUTPUT_COUNT:]]
    npu_outputs = [_to_torch(o) for o in outputs[:-GOLDEN_OUTPUT_COUNT]]
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)
    mid_result_mask = kwargs.get("mid_result_mask", None)
    gradEnabled = kwargs.get("gradEnabled", None)
    # Fallback: single NPU output (cmp_kv only) — compare cmp_kv, skip state_cache
    if len(npu_outputs) == 1:
        results = [
            _tensor_compare(
                npu_outputs[0][cmp_kv_mask].to(torch.float32),
                golden_outputs[0][cmp_kv_mask].to(torch.float32),
                "cmp_kv",
            )
        ]
        gc.collect()
        for i in range(1, len(golden_outputs)):
            name = _OUTPUT_NAMES[i] if i < len(_OUTPUT_NAMES) else f"output_{i}"
            results.append(
                {
                    "pass": True,
                    "precision": "N/A",
                    "error_info": f"{name}: NPU state_cache not captured",
                }
            )
        result_consistency = _batch_consistency_check(
            npu_outputs[0].to(torch.float32), kwargs
        )
        if result_consistency is not None:
            results.append(result_consistency)
        return results

    npu_cmp_kv = npu_outputs[0].to(torch.float32)
    npu_state_cache = npu_outputs[3].to(torch.float32)
    cpu_cmp_kv = golden_outputs[0].to(torch.float32)
    cpu_state_cache = golden_outputs[3].to(torch.float32)

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

    if gradEnabled:
        npu_softmax_out = npu_outputs[1][mid_result_mask].to(torch.float32)
        npu_kv_out = npu_outputs[2][mid_result_mask].to(torch.float32)
        cpu_softmax_out = golden_outputs[1][mid_result_mask].to(torch.float32)
        cpu_kv_out = golden_outputs[2][mid_result_mask].to(torch.float32)
        npu_sub_outputs.append(npu_softmax_out)
        npu_sub_outputs.append(npu_kv_out)
        golden_sub_outputs.append(cpu_softmax_out)
        golden_sub_outputs.append(cpu_kv_out)

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

    result_consistency = _batch_consistency_check(
        npu_outputs[0].to(torch.float32), kwargs
    )
    if result_consistency is not None:
        results.append(result_consistency)
    del npu_cmp_kv, npu_state_cache, cpu_cmp_kv, cpu_state_cache
    del npu_sub_outputs, golden_sub_outputs
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# ttk built-in cross_check (three-way comparison) entry points
# ---------------------------------------------------------------------------


def _get_compare_method():
    if get_global_storage is not None:
        try:
            return getattr(get_global_storage(), "compare_method", None)
        except Exception:
            return None
    return None


def _resolve_cross_check_params(spec_tolerance, dtype_str):
    standards = resolve_tolerance(
        spec_tolerance, None, None, [dtype_str], "cross_check"
    )
    return standards[0].params


def _ttk_cross_check_single(npu_out, golden_out, bench_out, idx, dtype_str, params):
    c = CrossCheckComparison(
        npu_out, bench_out, idx, dtype_str, params, third_party=golden_out
    )
    precision_str, log, is_pass, metrics = c.compare()
    return {
        "pass": is_pass,
        "precision": precision_str,
        "metrics": metrics,
        "error_info": None if is_pass else log,
    }


def _split_state_cache_sub_outputs(
    npu_state_cache, cpu_state_cache, bench_state, update_kv, update_score
):
    npu_sub = []
    golden_sub = []
    bench_sub = []

    if update_kv is not None and update_score is not None:
        npu_sub.append(
            npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][update_kv]
        )
        npu_sub.append(
            npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][update_score]
        )
        npu_sub.append(
            npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][~update_kv]
        )
        npu_sub.append(
            npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][~update_score]
        )

        golden_sub.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][update_kv]
        )
        golden_sub.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][update_score]
        )
        golden_sub.append(
            cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][~update_kv]
        )
        golden_sub.append(
            cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][~update_score]
        )

        if bench_state is not None:
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2][update_kv])
            bench_sub.append(
                bench_state[:, :, bench_state.shape[2] // 2 :][update_score]
            )
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2][~update_kv])
            bench_sub.append(
                bench_state[:, :, bench_state.shape[2] // 2 :][~update_score]
            )
        else:
            bench_sub.extend([None, None, None, None])
    else:
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :])
        npu_sub.append(None)
        npu_sub.append(None)

        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :])
        golden_sub.append(None)
        golden_sub.append(None)

        if bench_state is not None:
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :])
            bench_sub.append(None)
            bench_sub.append(None)
        else:
            bench_sub.extend([None, None, None, None])

    return npu_sub, golden_sub, bench_sub


def _resolve_raw_dtype(kwargs, default="bfloat16"):
    raw_dtype_str = kwargs.get("data_type")
    if raw_dtype_str is None:
        raw_dtype_str = default
    return raw_dtype_str.split(".")[-1].rstrip("'>\" ")


def _run_cross_check_loop(npu_sub, golden_sub, bench_sub, names, params):
    """Iterate over sub-outputs and run ttk cross_check on each.

    Shared by e2e and aclnn cross_check runners.
    """
    results = []
    for idx in range(len(golden_sub)):
        name = names[idx] if idx < len(names) else f"output_{idx}"
        npu_out = npu_sub[idx]
        golden_out = golden_sub[idx]
        bench_out = bench_sub[idx] if idx < len(bench_sub) else None
        if npu_out is None and golden_out is None:
            results.append(
                {
                    "pass": True,
                    "precision": "N/A",
                    "error_info": f"{name} is None",
                }
            )
        else:
            results.append(
                _ttk_cross_check_single(
                    npu_out, golden_out, bench_out, idx, "float32", params
                )
            )
    return results


def _split_aclnn_outputs(outputs, kwargs, bench_outputs):
    GOLDEN_OUTPUT_COUNT = 4
    golden_outputs = [_to_torch(g) for g in outputs[-GOLDEN_OUTPUT_COUNT:]]
    npu_outputs = [_to_torch(o) for o in outputs[:-GOLDEN_OUTPUT_COUNT]]
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)
    mid_result_mask = kwargs.get("mid_result_mask", None)
    gradEnabled = kwargs.get("gradEnabled", None)
    update_kv = kwargs.get("update_kv", None)
    update_score = kwargs.get("update_score", None)

    npu_sub = []
    golden_sub = []
    bench_sub = []

    if len(npu_outputs) == 1:
        npu_sub.append(npu_outputs[0][cmp_kv_mask])
        golden_sub.append(golden_outputs[0][cmp_kv_mask])
        bench = bench_outputs[0] if bench_outputs else None
        bench_sub.append(bench[cmp_kv_mask] if bench is not None else None)
        for i in range(1, len(golden_outputs)):
            npu_sub.append(None)
            golden_sub.append(None)
            bench_sub.append(None)
        return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES

    npu_cmp_kv = npu_outputs[0]
    npu_state_cache = npu_outputs[3]
    cpu_cmp_kv = golden_outputs[0]
    cpu_state_cache = golden_outputs[3]

    npu_sub.append(npu_cmp_kv[cmp_kv_mask])
    golden_sub.append(cpu_cmp_kv[cmp_kv_mask])
    bench_sub.append(bench_outputs[0][cmp_kv_mask] if bench_outputs else None)

    bench_state = None
    if bench_outputs and len(bench_outputs) > 1:
        bench_state = _to_torch(bench_outputs[1])
    sc_npu, sc_golden, sc_bench = _split_state_cache_sub_outputs(
        npu_state_cache, cpu_state_cache, bench_state, update_kv, update_score
    )
    npu_sub.extend(sc_npu)
    golden_sub.extend(sc_golden)
    bench_sub.extend(sc_bench)

    if gradEnabled:
        npu_sub.append(npu_outputs[1][mid_result_mask])
        npu_sub.append(npu_outputs[2][mid_result_mask])
        golden_sub.append(golden_outputs[1][mid_result_mask])
        golden_sub.append(golden_outputs[2][mid_result_mask])
        bench_sub.append(None)
        bench_sub.append(None)

    return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES


def _run_ttk_cross_check_aclnn(outputs, kwargs, bench_outputs, spec_tolerance):
    npu_sub, golden_sub, bench_sub, names = _split_aclnn_outputs(
        outputs, kwargs, bench_outputs
    )
    raw_dtype_str = _resolve_raw_dtype(kwargs)
    params = _resolve_cross_check_params(spec_tolerance, raw_dtype_str)
    results = _run_cross_check_loop(npu_sub, golden_sub, bench_sub, names, params)
    return results


def _split_e2e_outputs(outputs, kwargs, bench_outputs):
    GOLDEN_OUTPUT_COUNT = 2
    golden_outputs = [_to_torch(g) for g in outputs[-GOLDEN_OUTPUT_COUNT:]]
    npu_outputs = [_to_torch(o) for o in outputs[:-GOLDEN_OUTPUT_COUNT]]
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)
    update_kv = kwargs.get("update_kv", None)
    update_score = kwargs.get("update_score", None)

    npu_sub = []
    golden_sub = []
    bench_sub = []

    if len(npu_outputs) == 1:
        npu_sub.append(npu_outputs[0][cmp_kv_mask].to(torch.float32))
        golden_sub.append(golden_outputs[0][cmp_kv_mask].to(torch.float32))
        bench = bench_outputs[0] if bench_outputs else None
        bench_sub.append(
            bench[cmp_kv_mask].to(torch.float32) if bench is not None else None
        )
        for i in range(1, len(golden_outputs)):
            npu_sub.append(None)
            golden_sub.append(None)
            bench_sub.append(None)
        return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES

    npu_cmp_kv = npu_outputs[0]
    npu_state_cache = npu_outputs[1]
    cpu_cmp_kv = golden_outputs[0]
    cpu_state_cache = golden_outputs[1]

    npu_sub.append(npu_cmp_kv[cmp_kv_mask])
    golden_sub.append(cpu_cmp_kv[cmp_kv_mask])
    bench_sub.append(bench_outputs[0][cmp_kv_mask] if bench_outputs else None)

    bench_state = None
    if bench_outputs and len(bench_outputs) > 1:
        bench_state = _to_torch(bench_outputs[1])
    sc_npu, sc_golden, sc_bench = _split_state_cache_sub_outputs(
        npu_state_cache, cpu_state_cache, bench_state, update_kv, update_score
    )
    npu_sub.extend(sc_npu)
    golden_sub.extend(sc_golden)
    bench_sub.extend(sc_bench)

    return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES


def _run_ttk_cross_check_e2e(outputs, kwargs, bench_outputs, spec_tolerance):
    npu_sub, golden_sub, bench_sub, names = _split_e2e_outputs(
        outputs, kwargs, bench_outputs
    )
    raw_dtype_str = _resolve_raw_dtype(kwargs)
    params = _resolve_cross_check_params(spec_tolerance, raw_dtype_str)
    results = _run_cross_check_loop(npu_sub, golden_sub, bench_sub, names, params)
    return results


def is_cross_check_available():
    return _TTK_CROSS_CHECK_AVAILABLE
