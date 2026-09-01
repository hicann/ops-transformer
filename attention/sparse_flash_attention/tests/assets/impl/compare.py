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

"""TTK result adapter for the SparseFlashAttention pytest compare."""

import importlib.util
import logging
import sys
import threading
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


class PytestResultComparator:
    """Load and invoke the canonical pytest comparison without changing TTK logging."""

    def __init__(self):
        self.module = None
        self.lock = threading.Lock()

    def load_module(self):
        if self.module is not None:
            return self.module
        with self.lock:
            if self.module is not None:
                return self.module
            pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
            module_path = pytest_dir / "result_compare_method.py"
            module_name = "sfa_ttk_pytest_compare"
            inserted = str(pytest_dir) not in sys.path
            original_basic_config = logging.basicConfig
            if inserted:
                sys.path.insert(0, str(pytest_dir))
            try:
                logging.basicConfig = lambda *args, **kwargs: None
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.module = module
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise RuntimeError(
                    "Failed to load SparseFlashAttention pytest compare; "
                    f"module={module_path.resolve()}; "
                    f"original error: {type(exc).__name__}: {exc}"
                ) from exc
            finally:
                logging.basicConfig = original_basic_config
                if inserted and str(pytest_dir) in sys.path:
                    sys.path.remove(str(pytest_dir))
        return self.module

    @staticmethod
    def to_torch(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        array = np.array(value, copy=True, order="C")
        dtype_name = str(array.dtype)
        custom_dtypes = {
            "bfloat16": (np.uint16, torch.bfloat16),
            "float8_e4m3fn": (np.uint8, torch.float8_e4m3fn),
            "float8_e5m2": (np.uint8, torch.float8_e5m2),
        }
        if dtype_name in custom_dtypes:
            storage_dtype, torch_dtype = custom_dtypes[dtype_name]
            storage = np.ascontiguousarray(array).view(storage_dtype)
            return torch.from_numpy(storage).view(torch_dtype).reshape(array.shape)
        try:
            return torch.from_numpy(array)
        except TypeError:
            return torch.from_numpy(array.astype(np.float32))

    @staticmethod
    def normalize_result(result, output_index):
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise ValueError(
                f"pytest check_result output[{output_index}] returned invalid result: {result!r}"
            )
        status, precision = result[:2]
        passed = str(status).strip().lower() == "pass"
        return {
            "pass": passed,
            "precision": float(precision),
            "error_info": None
            if passed
            else (f"pytest check_result output[{output_index}] returned {status!r}"),
        }

    @staticmethod
    def is_empty_tensor(value):
        if torch.is_tensor(value):
            return value.numel() == 0
        return isinstance(value, np.ndarray) and value.size == 0

    def compare(self, *outputs):
        if len(outputs) < 2 or len(outputs) % 2 != 0:
            return {
                "pass": False,
                "precision": "invalid",
                "error_info": "compare expects NPU outputs followed by golden outputs",
            }
        module = self.load_module()
        half = len(outputs) // 2
        results = []
        for output_index, (npu_output, golden) in enumerate(
            zip(outputs[:half], outputs[half:])
        ):
            if golden is None:
                results.append({"pass": True, "precision": "SUPPRESSED"})
                continue
            if npu_output is None:
                results.append(
                    {
                        "pass": False,
                        "precision": "NO_OUTPUT",
                        "error_info": f"NPU output[{output_index}] is None",
                    }
                )
                continue
            if self.is_empty_tensor(golden) and self.is_empty_tensor(npu_output):
                continue

            result = module.check_result(
                self.to_torch(golden), self.to_torch(npu_output)
            )
            results.append(self.normalize_result(result, output_index))
        return results


COMPARATOR = PytestResultComparator()


def compare(*outputs):
    """Compare outputs with the operator's canonical pytest policy."""
    return COMPARATOR.compare(*outputs)


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


def is_cross_check_available():
    return _TTK_CROSS_CHECK_AVAILABLE


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


def _dtype_key(value):
    if torch.is_tensor(value):
        return str(value.dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def run_ttk_cross_check(*outputs, bench_outputs=None, spec_tolerance=None):
    """NPU outputs followed by golden outputs, each compared against the bench."""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    results = []
    for output_index, (npu_output, golden) in enumerate(
        zip(outputs[:half], outputs[half:])
    ):
        bench = None
        if bench_outputs is not None and output_index < len(bench_outputs):
            bench = bench_outputs[output_index]
        if golden is None:
            results.append({"pass": True, "precision": "SUPPRESSED"})
            continue
        if npu_output is None:
            results.append(
                {
                    "pass": False,
                    "precision": "NO_OUTPUT",
                    "error_info": f"NPU output[{output_index}] is None",
                }
            )
            continue
        if COMPARATOR.is_empty_tensor(golden) and COMPARATOR.is_empty_tensor(
            npu_output
        ):
            continue
        if bench is None or COMPARATOR.is_empty_tensor(bench):
            results.append(
                {
                    "pass": False,
                    "precision": "NO_BENCH",
                    "error_info": f"cross_check bench output[{output_index}] is missing",
                }
            )
            continue
        dtype_str = _dtype_key(golden)
        params = _resolve_cross_check_params(spec_tolerance, dtype_str)
        results.append(
            _ttk_cross_check_single(
                COMPARATOR.to_torch(npu_output),
                COMPARATOR.to_torch(golden),
                COMPARATOR.to_torch(bench),
                output_index,
                dtype_str,
                params,
            )
        )
    return results


def dispatch(*outputs, bench_outputs=None, spec_tolerance=None):
    """Route to ttk cross_check when enabled, otherwise the pytest policy."""
    if (
        _get_compare_method() == "cross_check"
        and bench_outputs is not None
        and _TTK_CROSS_CHECK_AVAILABLE
    ):
        return run_ttk_cross_check(
            *outputs, bench_outputs=bench_outputs, spec_tolerance=spec_tolerance
        )
    return COMPARATOR.compare(*outputs)
