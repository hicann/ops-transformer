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

"""TTK result adapter for the QuantLightningIndexer pytest TopK comparison."""

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
            module_name = "qli_ttk_pytest_compare"
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
                    "Failed to load QuantLightningIndexer pytest compare; "
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
        return torch.from_numpy(np.array(value, copy=True, order="C"))

    @staticmethod
    def result_dict(result):
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise ValueError(f"pytest check_result returned invalid result: {result!r}")
        status, precision = result[:2]
        passed = str(status).strip().lower() == "pass"
        return {
            "pass": passed,
            "precision": float(precision),
            "error_info": None if passed else (
                f"pytest QuantLightningIndexer check_result returned {status!r}"
            ),
        }

    def compare(self, *outputs, compare_data=None):
        if compare_data is None:
            raise ValueError("QuantLightningIndexer pytest compare data is unavailable")
        if len(outputs) != 2:
            return {
                "pass": False,
                "precision": "invalid",
                "error_info": "compare expects one NPU output followed by one golden output",
            }
        params = compare_data.get("params")
        topk_value = compare_data.get("topk_value")
        if params is None or topk_value is None:
            raise ValueError("QuantLightningIndexer pytest compare data lacks params or topk_value")
        npu_output, golden = outputs
        result = self.load_module().check_result(
            self.to_torch(golden),
            self.to_torch(npu_output),
            self.to_torch(topk_value),
            params,
        )
        return self.result_dict(result)


COMPARATOR = PytestTopKComparator()


def compare(*outputs, compare_data=None):
    """Compare TopK outputs with the canonical pytest policy."""
    return COMPARATOR.compare(*outputs, compare_data=compare_data)
