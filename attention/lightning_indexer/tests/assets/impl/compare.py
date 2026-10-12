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

"""TTK result adapter for the LightningIndexer pytest TopK comparison."""

import importlib.util
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import torch


class PytestTopKComparator:
    """Run the pytest TopK compare with replay-safe data supplied by the TestSpec."""

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
            module_name = "li_ttk_pytest_compare"
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
                    "Failed to load LightningIndexer pytest compare; "
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
        custom_dtypes = {
            "bfloat16": (np.uint16, torch.bfloat16),
            "float8_e4m3fn": (np.uint8, torch.float8_e4m3fn),
            "float8_e5m2": (np.uint8, torch.float8_e5m2),
        }
        dtype_name = str(array.dtype)
        if dtype_name in custom_dtypes:
            storage_dtype, torch_dtype = custom_dtypes[dtype_name]
            storage = np.ascontiguousarray(array).view(storage_dtype)
            return torch.from_numpy(storage).view(torch_dtype).reshape(array.shape)
        try:
            return torch.from_numpy(array)
        except TypeError:
            return torch.from_numpy(array.astype(np.float32))

    @staticmethod
    def result_dict(result):
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise ValueError(f"pytest check_result returned invalid result: {result!r}")
        status, precision = result[:2]
        passed = str(status).strip().lower() == "pass"
        return {
            "pass": passed,
            "precision": float(precision),
            "error_info": None
            if passed
            else (f"pytest LightningIndexer check_result returned {status!r}"),
        }

    def compare(self, *outputs, compare_data=None):
        if compare_data is None:
            raise ValueError("LightningIndexer pytest compare data is unavailable")
        if len(outputs) < 2 or len(outputs) % 2 != 0:
            return {
                "pass": False,
                "precision": "invalid",
                "error_info": "compare expects NPU outputs followed by golden outputs",
            }
        params = compare_data.get("params")
        topk_value = compare_data.get("topk_value")
        if params is None or topk_value is None:
            raise ValueError(
                "LightningIndexer pytest compare data lacks params or topk_value"
            )
        half = len(outputs) // 2
        npu_outputs = outputs[:half]
        golden_outputs = outputs[half:]
        return_value = bool(params[-1])
        if return_value and len(npu_outputs) < 2:
            return {
                "pass": False,
                "precision": "missing_output",
                "error_info": "return_value is enabled but the NPU sparse-value output is missing",
            }
        sparse_value = npu_outputs[1] if len(npu_outputs) > 1 else torch.empty(0)
        npu_indices, _ = torch.sort(self.to_torch(npu_outputs[0]))
        result = self.load_module().check_result(
            self.to_torch(golden_outputs[0]),
            npu_indices,
            self.to_torch(topk_value),
            self.to_torch(sparse_value),
            params,
        )
        return self.result_dict(result)


COMPARATOR = PytestTopKComparator()


def compare(*outputs, compare_data=None):
    """Compare TopK outputs with the canonical pytest policy."""
    return COMPARATOR.compare(*outputs, compare_data=compare_data)
