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

"""TTK result adapter for the SparseFlashMla pytest compare."""

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
            module_name = "smla_ttk_pytest_compare"
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
                    "Failed to load SparseFlashMla pytest compare; "
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
            result = module.check_result(
                self.to_torch(golden), self.to_torch(npu_output)
            )
            results.append(self.normalize_result(result, output_index))
        return results


COMPARATOR = PytestResultComparator()


class SparseFlashMlaBatchComparator:
    """Validate same-case batch relations; cross-case checks use dumped output bins."""

    OPERATOR_NAME = "SMLA"

    def config_failure(self, message):
        return {
            "pass": False,
            "precision": "batch_config=FAIL",
            "error_info": message,
        }

    @staticmethod
    def relation_slices_overlap(first, second):
        """Return whether two relation samples select the same q output region."""
        first_slices = first[0]
        first_axes = first[2]
        second_slices = second[0]
        second_axes = second[2]
        if first_axes != second_axes:
            return False
        first_batch = first_slices[0]
        second_batch = second_slices[0]
        if first_batch[1] <= second_batch[0] or second_batch[1] <= first_batch[0]:
            return False
        if first_axes == (0,):
            return True
        first_sequence = first_slices[1]
        second_sequence = second_slices[1]
        return not (
            first_sequence[1] <= second_sequence[0]
            or second_sequence[1] <= first_sequence[0]
        )

    def validate_disjoint_relations(self, relations):
        """Reject duplicate or overlapping samples that would self-compare."""
        for index, relation in enumerate(relations):
            for candidate in relations[index + 1 :]:
                if self.relation_slices_overlap(relation, candidate):
                    raise ValueError(
                        f"{self.OPERATOR_NAME} relation samples must not overlap"
                    )

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

    def validate_batch_consistency_id(self, batch_consistency_id, axes, axis_seeds):
        """Validate the stable seed/axis portion of framework relation IDs."""
        if (
            not isinstance(batch_consistency_id, (tuple, list))
            or len(batch_consistency_id) != 1
        ):
            raise ValueError("batch_consistency_id must contain one q relation group")
        id_axes = batch_consistency_id[0]
        if not isinstance(id_axes, (tuple, list)) or len(id_axes) != len(axes):
            raise ValueError("batch_consistency_id axis groups do not match q axes")
        for axis, ids, seeds in zip(axes, id_axes, axis_seeds):
            if not isinstance(ids, (tuple, list)) or len(ids) != len(seeds):
                raise ValueError(
                    "batch_consistency_id sample count does not match q seeds"
                )
            for relation_id, seed in zip(ids, seeds):
                parts = str(relation_id).split("_", 2)
                if len(parts) < 2:
                    raise ValueError(
                        f"invalid batch_consistency_id relation: {relation_id!r}"
                    )
                try:
                    id_seed, id_axis = (int(parts[0]), int(parts[1]))
                except ValueError as error:
                    raise ValueError(
                        f"invalid batch_consistency_id relation: {relation_id!r}"
                    ) from error
                if id_seed != int(seed) or id_axis != int(axis):
                    raise ValueError(
                        "batch_consistency_id seed/axis does not match q relation"
                    )

    def parse_relations(
        self, batch_consistency_id, batch_axis, batch_slice_info, batch_seed
    ):
        fields = (batch_consistency_id, batch_axis, batch_slice_info, batch_seed)
        if all(field is None for field in fields):
            return None, None
        if any(field is None for field in fields):
            return None, self.config_failure("incomplete batch consistency metadata")

        try:
            if not (len(batch_axis) == len(batch_slice_info) == len(batch_seed)):
                raise ValueError("batch metadata top-level counts differ")
            if not batch_axis or tuple(batch_axis[0]) not in ((0,), (0, 1)):
                raise ValueError(
                    f"{self.OPERATOR_NAME} batch compare requires logical q axes "
                    "(0,) or (0, 1)"
                )
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
            axes = tuple(batch_axis[0])
            if len(batch_slice_info[0]) != len(axes) or len(batch_seed[0]) != len(axes):
                raise ValueError(
                    f"{self.OPERATOR_NAME} q slice/seed groups must match logical axes"
                )

            axis_slices = batch_slice_info[0]
            axis_seeds = batch_seed[0]
            sample_count = len(axis_slices[0])
            if not sample_count or any(
                len(values) != sample_count for values in (*axis_slices, *axis_seeds)
            ):
                raise ValueError(
                    f"{self.OPERATOR_NAME} q axis sample counts differ or are empty"
                )

            relations = []
            for sample_index in range(sample_count):
                parsed_slices = []
                seed = None
                for axis_group, axis in enumerate(axes):
                    slice_value = axis_slices[axis_group][sample_index]
                    seed_value = axis_seeds[axis_group][sample_index]
                    if (
                        not isinstance(slice_value, (tuple, list))
                        or len(slice_value) != 3
                    ):
                        raise ValueError(f"invalid q axis {axis} slice {slice_value!r}")
                    if not all(isinstance(value, Integral) for value in slice_value):
                        raise ValueError(
                            f"q slice must contain integers: {slice_value!r}"
                        )
                    if not isinstance(seed_value, Integral):
                        raise ValueError(f"q seed must be an integer: {seed_value!r}")
                    start, stop, step = (int(value) for value in slice_value)
                    seed_value = int(seed_value)
                    if step != 1 or start < 0 or start >= stop:
                        raise ValueError(
                            "q slice must be non-empty, non-negative and contiguous: "
                            f"{slice_value!r}"
                        )
                    if seed is not None and seed_value != seed:
                        raise ValueError(
                            "logical B and S slices must use the same seed"
                        )
                    seed = seed_value
                    parsed_slices.append((start, stop, step))
                if axes == (0, 1) and parsed_slices[0][1] - parsed_slices[0][0] != 1:
                    raise ValueError("logical (B,S) relation requires one B per sample")
                relations.append((tuple(parsed_slices), seed, axes))
            self.validate_disjoint_relations(relations)
            self.validate_batch_consistency_id(batch_consistency_id, axes, axis_seeds)
        except (IndexError, TypeError, ValueError) as error:
            return None, self.config_failure(str(error))
        return relations, None

    def output_selector(self, npu, relation, compare_context):
        slices, _seed, axes = relation
        batch_slice = slices[0]
        sequence_slice = slices[1] if axes == (0, 1) else None
        attributes = (
            dict(compare_context.attributes) if compare_context is not None else {}
        )
        layout_q = attributes.get("layout_q", "BSND")
        batch_start, batch_stop, _ = batch_slice
        if layout_q == "BSND":
            if batch_stop > npu.shape[0]:
                raise ValueError(
                    f"logical B slice {batch_slice!r} exceeds output B={npu.shape[0]}"
                )
            selector = [slice(*batch_slice)]
            if sequence_slice is not None:
                if npu.ndim < 2 or sequence_slice[1] > npu.shape[1]:
                    raise ValueError(
                        f"logical S slice {sequence_slice!r} exceeds BSND output"
                    )
                selector.append(slice(*sequence_slice))
        elif layout_q == "TND":
            prefix = attributes.get("cu_seqlens_q_values")
            if prefix is None or len(prefix) < 2 or prefix[0] != 0:
                raise ValueError("TND batch compare requires cu_seqlens_q_values")
            if batch_stop >= len(prefix) or prefix[-1] != npu.shape[0]:
                raise ValueError("cu_seqlens_q_values does not match TND output")
            if sequence_slice is None:
                token_start, token_stop = prefix[batch_start], prefix[batch_stop]
            else:
                token_start = prefix[batch_start] + sequence_slice[0]
                token_stop = prefix[batch_start] + sequence_slice[1]
                if token_stop > prefix[batch_start + 1]:
                    raise ValueError("logical S slice exceeds its TND batch interval")
            selector = [slice(token_start, token_stop, 1)]
        else:
            raise ValueError(f"unsupported layout_q={layout_q!r}")
        selector.extend([slice(None)] * (npu.ndim - len(selector)))
        return tuple(selector)

    def compare_same_case(
        self,
        npu_output,
        batch_consistency_id,
        batch_axis,
        batch_slice_info,
        batch_seed,
        compare_context=None,
    ):
        relations, error = self.parse_relations(
            batch_consistency_id, batch_axis, batch_slice_info, batch_seed
        )
        if relations is None and error is None:
            return None
        if error is not None:
            return error
        if npu_output is None:
            return self.config_failure(f"{self.OPERATOR_NAME} batch output is None")

        npu = (
            npu_output.detach().cpu()
            if torch.is_tensor(npu_output)
            else np.asarray(npu_output)
        )
        if npu.ndim == 0:
            return self.config_failure(
                f"{self.OPERATOR_NAME} batch output must have a batch axis"
            )
        current_groups = {}
        for relation in relations:
            try:
                value = self.storage_bytes(
                    npu[self.output_selector(npu, relation, compare_context)]
                )
            except (IndexError, TypeError, ValueError) as error:
                return self.config_failure(str(error))
            _slices, seed, axes = relation
            relation_key = (0, axes, seed, value[0])
            current_groups.setdefault(relation_key, []).append(value)

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


BATCH_COMPARATOR = SparseFlashMlaBatchComparator()


def compare(
    *outputs,
    batch_consistency_id=None,
    batch_axis=None,
    batch_slice_info=None,
    batch_seed=None,
    compare_context=None,
):
    """Run pytest precision comparison before exact same-case batch checks."""
    results = COMPARATOR.compare(*outputs)
    if not isinstance(results, list) or not all(result["pass"] for result in results):
        return results

    batch_result = BATCH_COMPARATOR.compare_same_case(
        outputs[0],
        batch_consistency_id,
        batch_axis,
        batch_slice_info,
        batch_seed,
        compare_context,
    )
    if batch_result is not None:
        results.append(batch_result)
    return results
