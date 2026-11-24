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

"""Input conversion and mixed-precision CPU reference for SparseFlashAttention."""

import copy
import importlib.util
import logging
import sys
import threading
from importlib import metadata
from pathlib import Path

import torch

# The TTK file plugin provides a flat operator-local package; checkout imports use tests.pytest.
if __package__.endswith(".assets.impl"):
    from ...pytest.sparse_flash_attention_golden import compute_golden_row
else:
    from .sparse_flash_attention_golden import compute_golden_row


class CaseDataStore:
    def __init__(self):
        self.cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError(
                "SparseFlashAttention requires testcase_name for final inputs"
            )
        self.cases = {str(testcase_name): data}

    def get(self, testcase_name):
        data = self.cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "SparseFlashAttention Promote requires customize_inputs for this testcase"
            )
        return data


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
        # The pytest random generator accepts list ranges, not tuple ranges.
        for name, value in tuple(combination.items()):
            if name.startswith("range_") and isinstance(value, tuple):
                combination[name] = list(value)
        pytest_utils, pytest_golden = cls.load_modules()
        params = pytest_utils.convert_param_combination_to_cs_format(combination)
        params["case_name"] = testcase_name or params["case_name"]
        return params, pytest_golden


def cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, *, testcase_name=None, **kwargs
):
    arguments = dict(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        **kwargs,
    )
    if (
        arguments.pop("golden_mode", "Enable") == "Promote"
        or query.dtype == torch.float32
    ):
        inputs = CASE_DATA.get(testcase_name)["reference_inputs"]
        for name, cached in inputs.items():
            if torch.is_tensor(cached):
                current = arguments.get(name)
                if not torch.is_tensor(current) or current.shape != cached.shape:
                    raise ValueError(f"cached {name} shape differs from this testcase")
        return reference_from_inputs(
            **inputs, promote=True, testcase_name=testcase_name
        )
    return mixed_reference(**arguments)


def aclnn_arguments(args, kwargs, with_sinks):
    names = [
        "query",
        "key",
        "value",
        "sparse_indices",
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
        "query_rope",
        "key_rope",
    ]
    if with_sinks:
        names.append("sinks")
    names += [
        "scale_value",
        "sparse_block_size",
        "layout_query",
        "layout_kv",
        "sparse_mode",
        "pre_tokens",
        "next_tokens",
        "attention_mode",
        "return_softmax_lse",
        "attention_out",
        "softmax_max",
        "softmax_sum",
    ]
    if len(args) != len(names):
        raise ValueError(
            f"ACLNN SparseFlashAttention expects {len(names)} positional arguments"
        )
    arguments = dict(zip(names, args))
    for name in ("attention_out", "softmax_max", "softmax_sum"):
        arguments.pop(name)
    arguments.update(kwargs)
    return arguments


def cpu_sparse_flash_attention_aclnn(*args, **kwargs):
    return cpu_sparse_flash_attention(**aclnn_arguments(args, kwargs, False))


def cpu_sparse_flash_attention_aclnn_v2(*args, **kwargs):
    return cpu_sparse_flash_attention(**aclnn_arguments(args, kwargs, True))


def integer_vector(value, name, count=None):
    if value is None:
        return None
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    values = value.detach().cpu().tolist()
    if count is not None and len(values) != count:
        raise ValueError(f"{name} must contain {count} entries")
    return values


def sequence_layout(tensor, layout, lengths, name, batch=None):
    values = integer_vector(lengths, name)
    if layout == "BSND":
        if tensor.ndim != 4 or (batch is not None and tensor.shape[0] != batch):
            raise ValueError(f"{name} BSND shape mismatch")
        values = [tensor.shape[1]] * tensor.shape[0] if values is None else values
        if len(values) != tensor.shape[0] or any(
            x < 0 or x > tensor.shape[1] for x in values
        ):
            raise ValueError(f"{name} lengths exceed BSND capacity")
        return values, [b * tensor.shape[1] for b in range(tensor.shape[0])]
    if layout != "TND" or tensor.ndim != 3 or values is None:
        raise ValueError(f"{name} TND requires cumulative lengths")
    if (
        not values
        or values[-1] != tensor.shape[0]
        or (batch is not None and len(values) != batch)
    ):
        raise ValueError(f"{name} cumulative lengths must span the tensor")
    starts = [0] + values[:-1]
    sizes = [end - start for start, end in zip(starts, values)]
    if any(x < 0 for x in sizes):
        raise ValueError(f"{name} cumulative lengths must be nondecreasing")
    return sizes, starts


def prepare_reference(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    attention_mode=0,
    pre_tokens=(1 << 63) - 1,
    next_tokens=(1 << 63) - 1,
    **kwargs,
):
    """Allocate output/stat buffers and expose the visible Q/K/V for each query and KV head.

    Both precision paths consume these rows. Sparse selection and layout addressing
    are shared, while the caller chooses how to compute scores and probabilities.
    """
    if attention_mode != 2 or sparse_mode not in (0, 3):
        raise ValueError(
            "SparseFlashAttention reference requires attention_mode=2, sparse_mode=0/3"
        )
    if pre_tokens != (1 << 63) - 1 or next_tokens != (1 << 63) - 1:
        raise ValueError(
            "SparseFlashAttention supports only the default pre_tokens/next_tokens"
        )
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("SparseFlashAttention requires FP16/BF16 final inputs")
    if (query_rope is None) != (key_rope is None):
        raise ValueError("query_rope/key_rope must be present or absent together")
    if key.dtype != query.dtype or (
        value is not None and (value.shape != key.shape or value.dtype != key.dtype)
    ):
        raise ValueError("query/key/value dtype or shape mismatch")
    if not isinstance(sparse_block_size, int) or sparse_block_size <= 0:
        raise ValueError("sparse_block_size must be a positive integer")
    q_lengths, starts = sequence_layout(
        query, layout_query, actual_seq_lengths_query, "query"
    )
    batch = len(q_lengths)
    if layout_kv == "PA_BSND":
        kv_lengths = integer_vector(actual_seq_lengths_kv, "KV lengths", batch)
        if key.ndim != 4 or kv_lengths is None or block_table is None:
            raise ValueError(
                "PA_BSND requires a rank-4 cache, block table and KV lengths"
            )
        if (
            block_table.ndim != 2
            or block_table.shape[0] != batch
            or block_table.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("block_table shape/dtype mismatch")
        if any(x < 0 or x > block_table.shape[1] * key.shape[1] for x in kv_lengths):
            raise ValueError("block_table columns do not cover KV lengths")
        kv_starts = [0] * batch
    else:
        kv_lengths, kv_starts = sequence_layout(
            key, layout_kv, actual_seq_lengths_kv, "KV", batch
        )
    heads, kv_heads, dim = query.shape[-2], key.shape[-2], query.shape[-1]
    if key.shape[-1] != dim or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("query/key head geometry mismatch")
    if sparse_indices.dtype not in (
        torch.int32,
        torch.int64,
    ) or sparse_indices.shape != (
        *query.shape[:-2],
        kv_heads,
        sparse_indices.shape[-1],
    ):
        raise ValueError("sparse_indices shape/dtype mismatch")
    if query_rope is not None:
        if (
            query_rope.shape[:-1] != query.shape[:-1]
            or key_rope.shape[:-1] != key.shape[:-1]
            or query_rope.shape[-1] != key_rope.shape[-1]
        ):
            raise ValueError("RoPE shape mismatch")
        if query_rope.dtype != query.dtype or key_rope.dtype != key.dtype:
            raise ValueError("RoPE dtype mismatch")
    q = query.detach().cpu().float()
    k = key.detach().cpu().float()
    v = (key if value is None else value).detach().cpu().float()
    if query_rope is not None:
        q = torch.cat((q, query_rope.detach().cpu().float()), dim=-1)
        k = torch.cat((k, key_rope.detach().cpu().float()), dim=-1)
    q = q.reshape(-1, heads, q.shape[-1])
    k = k.reshape(-1, kv_heads, k.shape[-1])
    v = v.reshape(-1, kv_heads, dim)
    indices = (
        sparse_indices.detach().cpu().reshape(-1, kv_heads, sparse_indices.shape[-1])
    )
    sink_values = None if sinks is None else sinks.detach().cpu().float()
    if sink_values is not None and sink_values.shape != (heads,):
        raise ValueError("sinks must have one value per query head")
    output = torch.zeros(query.shape, dtype=torch.float32)
    group = heads // kv_heads
    shape = (
        (batch, kv_heads, query.shape[1], group)
        if layout_query == "BSND"
        else (kv_heads, query.shape[0], group)
    )
    statistics = (torch.zeros(shape), torch.zeros(shape))

    def rows():
        for b, q_length in enumerate(q_lengths):
            for row in range(q_length):
                visible = (
                    kv_lengths[b]
                    if sparse_mode == 0
                    else max(0, kv_lengths[b] - q_length + row + 1)
                )
                flat_row = starts[b] + row
                for head in range(kv_heads):
                    positions = []
                    count = min(
                        indices.shape[-1],
                        (visible + sparse_block_size - 1) // sparse_block_size,
                    )
                    for block in indices[flat_row, head, :count].tolist():
                        if block == -1:
                            break
                        if block < 0 or block * sparse_block_size >= kv_lengths[b]:
                            raise ValueError("sparse index out of range")
                        positions.extend(
                            range(
                                block * sparse_block_size,
                                min((block + 1) * sparse_block_size, visible),
                            )
                        )
                    logical = torch.tensor(positions, dtype=torch.long)
                    if layout_kv == "PA_BSND":
                        pages = (
                            block_table[b, logical // key.shape[1]]
                            .detach()
                            .cpu()
                            .long()
                        )
                        if torch.any((pages < 0) | (pages >= key.shape[0])):
                            raise ValueError(
                                "block_table contains an invalid physical page"
                            )
                        physical = pages * key.shape[1] + logical % key.shape[1]
                    else:
                        physical = kv_starts[b] + logical
                    out_index = (
                        (b, row, slice(head * group, (head + 1) * group))
                        if layout_query == "BSND"
                        else (flat_row, slice(head * group, (head + 1) * group))
                    )
                    stat_index = (
                        (b, head, row) if layout_query == "BSND" else (head, flat_row)
                    )
                    sink = (
                        None
                        if sink_values is None
                        else sink_values[head * group : (head + 1) * group]
                    )
                    yield (
                        q[flat_row, head * group : (head + 1) * group],
                        k[physical, head],
                        v[physical, head],
                        sink,
                        out_index,
                        stat_index,
                    )

    return output, statistics, rows, query.dtype, scale_value


def reference_from_inputs(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    promote=False,
    testcase_name=None,
    **kwargs,
):
    """Address final API tensors once and pass each visible row to the operator's pytest golden."""
    if promote:
        if query.device.type != "cpu":
            raise ValueError("the independent FP32 golden requires CPU inputs")
        logging.info(
            "[%s] SFA FP32 golden: preparing final inputs for pytest compute_golden_row, layout=%s/%s",
            testcase_name,
            kwargs.get("layout_query", "BSND"),
            kwargs.get("layout_kv", "BSND"),
        )
    output, statistics, rows, dtype, scale = prepare_reference(
        query, key, value, sparse_indices, scale_value, **kwargs
    )
    if promote:
        logging.info(
            "[%s] SFA pytest FP32: QK -> softmax -> PV, scale=%s",
            testcase_name,
            scale,
        )
    q_dtype = str(dtype).removeprefix("torch.")
    for q, k, v, sink, out_index, stat_index in rows():
        if not k.shape[0]:
            if sink is not None:
                statistics[0][stat_index], statistics[1][stat_index] = sink, 1.0
            continue
        result, maximum, total = compute_golden_row(
            q, k, v, scale, sink, q_dtype, promote=promote
        )
        output[out_index] = result
        statistics[0][stat_index], statistics[1][stat_index] = (
            maximum[:, 0],
            total[:, 0],
        )
    if promote:
        logging.info(
            "[%s] SFA FP32 golden ready: attention=%s %s",
            testcase_name,
            tuple(output.shape),
            output.dtype,
        )
    if not promote:
        finite = torch.isfinite(output)
        output = torch.where(
            finite,
            output.clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max),
            output,
        ).to(dtype)
    if not kwargs.get("return_softmax_lse", False):
        return output, None, None
    return output, *statistics


def mixed_reference(*args, **kwargs):
    """Use the ordinary precision contract for the existing two-party and provider callers."""
    return reference_from_inputs(*args, promote=False, **kwargs)
