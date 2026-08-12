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

"""TTK adapter for the canonical StemIndexer pytest precision comparison."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


PYTEST_COMPARE_MODULE = None
INPUT_TENSOR_COUNT = 7


def load_pytest_compare_module():
    """Load the pytest comparator so TTK and pytest share one precision policy."""
    global PYTEST_COMPARE_MODULE
    if PYTEST_COMPARE_MODULE is not None:
        return PYTEST_COMPARE_MODULE

    pytest_dir = Path(__file__).resolve().parents[3] / "pytest"
    module_path = pytest_dir / "result_compare_method.py"
    module_name = "stem_indexer_ttk_pytest_compare"
    inserted = str(pytest_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        PYTEST_COMPARE_MODULE = module
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted:
            sys.path.remove(str(pytest_dir))
    return PYTEST_COMPARE_MODULE


def to_torch(value):
    """Convert a TTK tensor representation to a detached CPU Torch tensor."""
    if torch.is_tensor(value):
        return value.detach().cpu()

    array = np.array(value, copy=True, order="C")
    custom_dtypes = {
        "bfloat16": (np.uint16, torch.bfloat16),
    }
    dtype_name = str(array.dtype)
    if dtype_name in custom_dtypes:
        storage_dtype, torch_dtype = custom_dtypes[dtype_name]
        storage = np.ascontiguousarray(array).view(storage_dtype)
        return torch.from_numpy(storage).view(torch_dtype).reshape(array.shape)
    return torch.from_numpy(array)


def get_length_list(attributes, name, tensor):
    """Prefer the logical sequence lengths recorded in the case attributes."""
    values = attributes.get(f"{name}_list")
    if values is None:
        values = to_torch(tensor).reshape(-1).tolist()
    return [int(value) for value in values]


def build_compare_data(compare_context):
    """Rebuild the case and inputs expected by the canonical pytest comparator."""
    input_tensors = tuple(compare_context.input_tensors or ())
    if len(input_tensors) != INPUT_TENSOR_COUNT:
        raise ValueError(
            f"StemIndexer compare expects {INPUT_TENSOR_COUNT} input tensors, "
            f"got {len(input_tensors)}"
        )

    qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens, metadata = (
        input_tensors
    )
    attributes = dict(compare_context.attributes or {})
    qflat = to_torch(qflat)
    kflat = to_torch(kflat)
    vbias = to_torch(vbias)
    q_seq_lens = to_torch(q_seq_lens).to(torch.int32)
    kv_seq_lens = to_torch(kv_seq_lens).to(torch.int32)
    num_prompt_tokens = to_torch(num_prompt_tokens).to(torch.int32)

    q_seq_lens_list = get_length_list(attributes, "q_seq_lens", q_seq_lens)
    kv_seq_lens_list = get_length_list(attributes, "kv_seq_lens", kv_seq_lens)
    num_prompt_tokens_list = get_length_list(
        attributes, "num_prompt_tokens", num_prompt_tokens
    )
    case = {
        "case_id": str(compare_context.testcase_name),
        "batch_size": int(attributes.get("batch_size", qflat.shape[0])),
        "q_heads": int(attributes.get("q_heads", qflat.shape[1])),
        "kv_heads": int(attributes.get("kv_heads", kflat.shape[1])),
        "q_seq_lens": q_seq_lens_list,
        "kv_seq_lens": kv_seq_lens_list,
        "num_prompt_tokens": num_prompt_tokens_list,
        "causal": bool(attributes.get("causal", True)),
        "alpha": float(attributes.get("alpha", 1.0)),
        "stem_block_size": int(attributes.get("stem_block_size", 128)),
        "stem_stride": int(attributes.get("stem_stride", 16)),
        "initial_blocks": int(attributes.get("initial_blocks", 4)),
        "window_size": int(attributes.get("window_size", 4)),
        "topk_score_precision": int(attributes.get("topk_score_precision", 1)),
        "special_setting": attributes.get("special_setting", ""),
    }
    inputs = {
        "qflat": qflat,
        "kflat": kflat,
        "vbias": vbias,
        "q_seq_lens": q_seq_lens,
        "kv_seq_lens": kv_seq_lens,
        "num_prompt_tokens": num_prompt_tokens,
        "metadata": to_torch(metadata),
    }
    return case, inputs


def compare(*outputs, compare_context=None, **kwargs):
    """Compare TTK outputs using the canonical pytest precision policy."""
    del kwargs
    if len(outputs) != 4:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"compare expects 4 outputs, got {len(outputs)}",
        }
    if compare_context is None:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "StemIndexer pytest precision compare requires compare_context",
        }

    actual_indices, actual_seq_len, expected_indices, expected_seq_len = (
        to_torch(output) for output in outputs
    )
    case, inputs = build_compare_data(compare_context)
    comparator = load_pytest_compare_module()
    try:
        comparator.assert_stem_indexer_result(
            expected_indices,
            expected_seq_len,
            (actual_indices, actual_seq_len),
            case,
            inputs,
        )
    except AssertionError as error:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": str(error),
        }

    return {
        "pass": True,
        "precision": 100.0,
        "error_info": None,
    }
