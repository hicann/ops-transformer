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

"""TTK result adapter for the QuantSparseFlashMla pytest compare."""

import importlib.util
import logging
import sys
import threading
from numbers import Integral
from pathlib import Path

import numpy as np
import torch


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
            module_name = "qsmla_ttk_pytest_compare"
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
                    "Failed to load QuantSparseFlashMla pytest compare; "
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
            "error_info": None if passed else (
                f"pytest check_result output[{output_index}] returned {status!r}"
            ),
        }

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
                zip(outputs[:half], outputs[half:])):
            if golden is None:
                results.append({"pass": True, "precision": "SUPPRESSED"})
                continue
            if npu_output is None:
                results.append({
                    "pass": False,
                    "precision": "NO_OUTPUT",
                    "error_info": f"NPU output[{output_index}] is None",
                })
                continue
            result = module.check_result(self.to_torch(golden), self.to_torch(npu_output))
            results.append(self.normalize_result(result, output_index))
        return results


COMPARATOR = PytestResultComparator()


class QuantSparseFlashMlaBatchComparator:
    """Validate same-case batch relations; cross-case checks use dumped output bins."""

    OPERATOR_NAME = "QSMLA"

    def config_failure(self, message):
        return {
            "pass": False,
            "precision": "batch_config=FAIL",
            "error_info": message,
        }

    @staticmethod
    def storage_bytes(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            return (
                tuple(tensor.shape),
                str(tensor.dtype),
                tensor.view(torch.uint8).numpy().tobytes(),
            )
        array = np.ascontiguousarray(np.asarray(value))
        return tuple(array.shape), array.dtype.str, array.view(np.uint8).tobytes()

    def parse_relations(self, batch_consistency_id, batch_axis,
                        batch_slice_info, batch_seed):
        fields = (batch_consistency_id, batch_axis, batch_slice_info, batch_seed)
        if all(field is None for field in fields):
            return None, None
        if any(field is None for field in fields):
            return None, self.config_failure("incomplete batch consistency metadata")

        try:
            if not (len(batch_axis) == len(batch_slice_info) == len(batch_seed)):
                raise ValueError("batch metadata top-level counts differ")
            if not batch_axis or tuple(batch_axis[0]) != (0,):
                raise ValueError(f"{self.OPERATOR_NAME} batch compare requires q axis 0")
            if batch_slice_info[0] is None or batch_seed[0] is None:
                raise ValueError(
                    f"{self.OPERATOR_NAME} batch compare requires q slices and seeds"
                )
            if any(value is not None for value in batch_slice_info[1:]):
                raise ValueError(
                    f"{self.OPERATOR_NAME} batch compare supports q relations only"
                )
            if any(value is not None for value in batch_seed[1:]):
                raise ValueError(
                    f"{self.OPERATOR_NAME} batch compare supports q seeds only"
                )
            if len(batch_slice_info[0]) != 1 or len(batch_seed[0]) != 1:
                raise ValueError(
                    f"{self.OPERATOR_NAME} q metadata must contain exactly one axis group"
                )

            slices = batch_slice_info[0][0]
            seeds = batch_seed[0][0]
            if not slices or len(slices) != len(seeds):
                raise ValueError(
                    f"{self.OPERATOR_NAME} q slice and seed counts differ or are empty"
                )

            relations = []
            expected_ids = []
            for slice_value, seed in zip(slices, seeds):
                if not isinstance(slice_value, (tuple, list)) or len(slice_value) != 3:
                    raise ValueError(f"invalid q slice {slice_value!r}")
                if not all(isinstance(value, Integral) for value in slice_value):
                    raise ValueError(f"q slice must contain integers: {slice_value!r}")
                if not isinstance(seed, Integral):
                    raise ValueError(f"q seed must be an integer: {seed!r}")
                start, stop, step = (int(value) for value in slice_value)
                seed = int(seed)
                if step != 1 or start < 0 or start >= stop:
                    raise ValueError(
                        "q slice must be non-empty, non-negative and contiguous: "
                        f"{slice_value!r}"
                    )
                relation = (0, 0, seed, stop - start)
                relations.append(((start, stop, step), relation))
                expected_ids.append(f"{seed}_0_{start}_{stop}_{step}")
        except (IndexError, TypeError, ValueError) as error:
            return None, self.config_failure(str(error))

        expected_batch_id = ((tuple(expected_ids),),)
        if batch_consistency_id != expected_batch_id:
            return None, self.config_failure(
                f"{self.OPERATOR_NAME} batch_consistency_id does not match batch "
                f"slice metadata: expected={expected_batch_id!r}, "
                f"actual={batch_consistency_id!r}"
            )
        return relations, None

    def compare_same_case(self, npu_output, batch_consistency_id,
                          batch_axis, batch_slice_info, batch_seed):
        relations, error = self.parse_relations(
            batch_consistency_id, batch_axis, batch_slice_info, batch_seed
        )
        if relations is None and error is None:
            return None
        if error is not None:
            return error
        if npu_output is None:
            return self.config_failure(f"{self.OPERATOR_NAME} batch output is None")

        npu = npu_output.detach().cpu() if torch.is_tensor(npu_output) else np.asarray(npu_output)
        if npu.ndim == 0:
            return self.config_failure(
                f"{self.OPERATOR_NAME} batch output must have a batch axis"
            )
        current_groups = {}
        for slice_value, relation in relations:
            if slice_value[1] > npu.shape[0]:
                return self.config_failure(
                    f"{self.OPERATOR_NAME} q slice {slice_value!r} exceeds output "
                    f"batch size {npu.shape[0]}"
                )
            selector = (slice(*slice_value),) + (slice(None),) * (npu.ndim - 1)
            current_groups.setdefault(relation, []).append(
                self.storage_bytes(npu[selector])
            )

        compared_groups = 0
        for relation, values in current_groups.items():
            if len(values) < 2:
                continue
            compared_groups += 1
            if any(values[0] != value for value in values[1:]):
                return {
                    "pass": False,
                    "precision": "batch_intra=FAIL",
                    "error_info": (
                        f"{self.OPERATOR_NAME} intra-case relation {relation} differs"
                    ),
                }

        if compared_groups == 0:
            return {"pass": True, "precision": "batch_intra=NOT_APPLICABLE"}
        return {"pass": True, "precision": "batch_intra=PASS"}


BATCH_COMPARATOR = QuantSparseFlashMlaBatchComparator()


def compare(*outputs, batch_consistency_id=None, batch_axis=None,
            batch_slice_info=None, batch_seed=None):
    """Run pytest precision comparison before exact same-case batch checks."""
    results = COMPARATOR.compare(*outputs)
    if not isinstance(results, list) or not all(result["pass"] for result in results):
        return results

    batch_result = BATCH_COMPARATOR.compare_same_case(
        outputs[0], batch_consistency_id, batch_axis, batch_slice_info, batch_seed
    )
    if batch_result is not None:
        results.append(batch_result)
    return results
