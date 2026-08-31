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

"""Pytest-backed CPU golden adapter for KvQuantSparseFlashAttention."""

import copy
import importlib.util
import sys
import threading
from importlib import metadata
from pathlib import Path

import torch


class CaseDataStore:
    """Keep one pytest-generated result for each TTK testcase."""

    def __init__(self):
        self.cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError(
                "KvQuantSparseFlashAttention requires testcase_name for pytest-backed golden"
            )
        self.cases = {str(testcase_name): data}

    def get(self, testcase_name):
        data = self.cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "KvQuantSparseFlashAttention golden requires customize_inputs to run before golden generation"
            )
        return data


CASE_DATA = CaseDataStore()


class KvQuantSparseFlashAttentionPytestAdapter:
    """Load and invoke the operator's canonical pytest parameter conversion."""

    modules = None
    modules_lock = threading.Lock()

    LOCAL_MODULE_NAMES = (
        "check_valid_param",
        "generate_tensor_data",
        "kv_quant_sparse_flash_attention_golden",
        "result_compare_method",
    )

    REQUIRED_PARAM_NAMES = (
        "Testcase_Prefix",
        "layout_query",
        "layout_kv",
        "q_type",
        "kv_type",
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "scale_value",
        "key_quant_mode",
        "value_quant_mode",
        "sparse_block_size",
        "tile_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
        "quant_scale_repo_mode",
        "actual_seq_q",
        "actual_seq_kv",
    )

    @staticmethod
    def package_version(*distribution_names):
        for name in distribution_names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return "not-installed"

    @classmethod
    def load_error(cls, module_path, exc):
        versions = (
            f"python={sys.version.split()[0]}, "
            f"tensorflow={cls.package_version('tensorflow', 'tensorflow-cpu')}, "
            f"numpy={cls.package_version('numpy')}"
        )
        return RuntimeError(
            "Failed to load KvQuantSparseFlashAttention pytest parameter conversion; "
            f"module={module_path}; {versions}; original error: "
            f"{type(exc).__name__}: {exc}"
        )

    @classmethod
    def load_modules(cls):
        if cls.modules is not None:
            return cls.modules
        with cls.modules_lock:
            if cls.modules is not None:
                return cls.modules

            pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
            module_path = pytest_dir / "utils.py"
            module_name = "qsfa_ttk_pytest_utils"
            saved_modules = {
                name: sys.modules.pop(name, None) for name in cls.LOCAL_MODULE_NAMES
            }
            sys.path.insert(0, str(pytest_dir))
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                pytest_golden = module.kv_quant_sparse_flash_attention_golden
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise cls.load_error(module_path, exc) from exc
            finally:
                sys.path.remove(str(pytest_dir))
                for name in cls.LOCAL_MODULE_NAMES:
                    sys.modules.pop(name, None)
                for name, saved in saved_modules.items():
                    if saved is not None:
                        sys.modules[name] = saved
            cls.modules = (module, pytest_golden)
            return cls.modules

    @staticmethod
    def query_dtype(value):
        if isinstance(value, torch.dtype):
            return value
        normalized = str(value).strip().lower().removeprefix("torch.")
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported QSFA pytest query dtype: {value!r}")
        return mapping[normalized]

    @classmethod
    def convert_params(cls, attributes, testcase_name):
        if not isinstance(attributes, dict):
            raise ValueError("QSFA input attributes must be a dictionary")
        missing = [
            f"pytest_{name}"
            for name in cls.REQUIRED_PARAM_NAMES
            if f"pytest_{name}" not in attributes
        ]
        if missing:
            raise ValueError(f"QSFA CSV is missing explicit pytest fields: {missing}")
        pytest_params = {
            name: copy.deepcopy(attributes.get(f"pytest_{name}"))
            for name in cls.REQUIRED_PARAM_NAMES
        }
        for name in ("T1", "T2", "block_size", "block_num"):
            key = f"pytest_{name}"
            if key in attributes:
                pytest_params[name] = copy.deepcopy(attributes[key])
        for name in (
            "range_query",
            "range_key",
            "range_query_rope",
            "range_key_rope",
            "range_dequant_scale",
            "Testcase_Number",
        ):
            key = f"pytest_{name}"
            if key in attributes:
                pytest_params[name] = copy.deepcopy(attributes[key])
        if (
            pytest_params["actual_seq_q"] is None
            or pytest_params["actual_seq_kv"] is None
        ):
            raise ValueError(
                "QSFA CSV must provide actual_seq_q and actual_seq_kv; "
                "random sequence generation is not allowed during TTK conversion"
            )

        combination = pytest_params
        combination["q_type"] = cls.query_dtype(combination["q_type"])
        pytest_utils, pytest_golden = cls.load_modules()
        params = pytest_utils.convert_param_combination_to_cs_format(combination)
        params["case_name"] = testcase_name or params["case_name"]
        return params, pytest_golden


def cpu_kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    testcase_name=None,
    **kwargs,
):
    """Return the pytest CPU golden that was generated for this exact case."""
    del (
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        kwargs,
    )
    return CASE_DATA.get(testcase_name)["golden"]


__golden__ = {
    "e2e": {
        "torch_npu.npu_kv_quant_sparse_flash_attention": "cpu_kv_quant_sparse_flash_attention",
        "qsfa_ttk_ops.kv_quant_sparse_flash_attention_ttk": "cpu_kv_quant_sparse_flash_attention",
    }
}
