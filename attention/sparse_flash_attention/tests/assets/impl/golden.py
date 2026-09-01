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

"""Pytest-backed CPU golden adapter for SparseFlashAttention."""

import copy
import importlib.util
import sys
import threading
from importlib import metadata
from pathlib import Path

import torch

try:
    from ttk.utilities.container_utils import get_global_storage
except Exception:
    get_global_storage = None


def _get_compare_method():
    if get_global_storage is not None:
        try:
            return getattr(get_global_storage(), "compare_method", None)
        except Exception:
            return None
    return None


def compare_method_is_cross_check():
    return _get_compare_method() == "cross_check"


class CaseDataStore:
    """Keep the pytest-generated result paired with its TTK testcase."""

    def __init__(self):
        self.cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError(
                "SparseFlashAttention requires testcase_name for pytest-backed golden"
            )
        self.cases = {str(testcase_name): data}

    def get(self, testcase_name):
        data = self.cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "SparseFlashAttention golden requires customize_inputs to run "
                "before golden generation"
            )
        return data


def peek_bench(testcase_name):
    """取当前用例的三方标杆输出(cross_check 用)；未生成或未开启三方时返回 None。"""
    if testcase_name is None:
        return None
    entry = CASE_DATA.cases.get(str(testcase_name))
    if not entry:
        return None
    return entry.get("bench")


CASE_DATA = CaseDataStore()


class SparseFlashAttentionPytestAdapter:
    """Load pytest and restore its source parameter combination from CSV."""

    modules = None
    modules_lock = threading.Lock()

    LOCAL_MODULE_NAMES = (
        "check_valid_param",
        "generate_tensor_data",
        "result_compare_method",
        "sparse_flash_attention_golden",
    )

    PARAMETER_NAMES = (
        "Testcase_Prefix",
        "layout_query",
        "layout_kv",
        "q_type",
        "kv_type",
        "B",
        "T1",
        "T2",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "scale_value",
        "sparse_block_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
        "return_softmax_lse",
        "use_sinks",
        "block_size",
        "block_num",
        "actual_seq_q",
        "actual_seq_kv",
        "range_query",
        "range_key",
        "range_query_rope",
        "range_key_rope",
        "range_sinks",
    )

    REQUIRED_PARAMETER_NAMES = (
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
        "sparse_block_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
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
    def load_error(cls, module_path, dependency_path, exc):
        versions = (
            f"python={sys.version.split()[0]}, "
            f"tensorflow={cls.package_version('tensorflow', 'tensorflow-cpu')}, "
            f"numpy={cls.package_version('numpy')}"
        )
        return RuntimeError(
            "Failed to load SparseFlashAttention pytest parameter conversion; "
            f"module={module_path}; dependency={dependency_path}; {versions}; "
            f"original error: {type(exc).__name__}: {exc}"
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
            dependency_path = pytest_dir / "generate_tensor_data.py"
            module_name = "sfa_ttk_pytest_utils"
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
                pytest_golden = module.sparse_flash_attention_golden
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise cls.load_error(module_path, dependency_path, exc) from exc
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
    def torch_dtype(value, field):
        if isinstance(value, torch.dtype):
            return value
        normalized = str(value).strip().lower().removeprefix("torch.")
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if normalized not in mapping:
            raise ValueError(
                f"Unsupported SparseFlashAttention pytest {field}: {value!r}"
            )
        return mapping[normalized]

    @classmethod
    def convert_params(cls, attributes, testcase_name):
        if not isinstance(attributes, dict):
            raise ValueError(
                "SparseFlashAttention input attributes must be a dictionary"
            )
        missing = [
            f"pytest_{name}"
            for name in cls.REQUIRED_PARAMETER_NAMES
            if f"pytest_{name}" not in attributes
        ]
        if missing:
            raise ValueError(
                f"SparseFlashAttention CSV is missing explicit pytest fields: {missing}"
            )
        combination = {
            name: copy.deepcopy(attributes[f"pytest_{name}"])
            for name in cls.PARAMETER_NAMES
            if f"pytest_{name}" in attributes
        }
        combination["q_type"] = cls.torch_dtype(combination["q_type"], "q_type")
        combination["kv_type"] = cls.torch_dtype(combination["kv_type"], "kv_type")
        pytest_utils, pytest_golden = cls.load_modules()
        params = pytest_utils.convert_param_combination_to_cs_format(combination)
        params["case_name"] = testcase_name or params["case_name"]
        return params, pytest_golden


def normalize_pytest_outputs(outputs, query, return_softmax_lse):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError(
            "SparseFlashAttention pytest compute_cpu returned an invalid output tuple"
        )
    # 对齐算子Cast(CAST_ROUND)语义:仅"有限值"超出目标dtype表示范围时钳到上下界，±inf/NaN 原样保留
    attn_fp32 = outputs[0].detach().cpu().float()
    attn_out = torch.where(
        torch.isfinite(attn_fp32),
        attn_fp32.clamp(-torch.finfo(query.dtype).max, torch.finfo(query.dtype).max),
        attn_fp32,
    ).to(dtype=query.dtype)
    if not bool(return_softmax_lse):
        empty = torch.zeros(0, dtype=torch.float32)
        return attn_out, empty, empty
    return tuple(output.detach().cpu() for output in outputs)


def cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, *, testcase_name=None, **kwargs
):
    """Return the result generated by pytest for this exact TTK testcase."""
    del query, key, value, sparse_indices, scale_value, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


def cpu_sparse_flash_attention_aclnn(*args, testcase_name=None, **kwargs):
    """ACLNN 直调版 golden：按 aclnn C 签名顺序接收全部位置参数。"""
    del args, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


__golden__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "cpu_sparse_flash_attention",
    },
    "aclnn": {
        "aclnnSparseFlashAttention": "cpu_sparse_flash_attention_aclnn",
        "aclnnSparseFlashAttentionV2": "cpu_sparse_flash_attention_aclnn",
    },
}
