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

"""CPU references and canonical pytest input generation for QSFA V2."""

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
    from ...pytest.kv_quant_sparse_flash_attention_v2_golden import (
        compute_golden_row,
        padding_statistics,
    )
else:
    from .kv_quant_sparse_flash_attention_v2_golden import (
        compute_golden_row,
        padding_statistics,
    )


class CaseDataStore:
    """Retain the exact final inputs for the current testcase only."""

    def __init__(self):
        self.cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError(
                "KvQuantSparseFlashAttentionV2 requires testcase_name for the final input snapshot"
            )
        self.cases = {str(testcase_name): data}

    def get(self, testcase_name):
        data = self.cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "KvQuantSparseFlashAttentionV2 golden requires customize_inputs to run before golden generation"
            )
        return data


CASE_DATA = CaseDataStore()


class KvQuantSparseFlashAttentionV2PytestAdapter:
    """Load and invoke the operator's canonical pytest parameter conversion."""

    modules = None
    modules_lock = threading.Lock()

    LOCAL_MODULE_NAMES = (
        "check_valid_param",
        "generate_tensor_data",
        "kv_quant_sparse_flash_attention_v2_golden",
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
            f"torch={cls.package_version('torch')}, "
            f"numpy={cls.package_version('numpy')}"
        )
        return RuntimeError(
            "Failed to load KvQuantSparseFlashAttentionV2 pytest parameter conversion; "
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
            module_name = "qsfa_v2_ttk_pytest_utils"
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
                pytest_golden = module.kv_quant_sparse_flash_attention_v2_golden
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
            raise ValueError(f"Unsupported QSFAV2 pytest query dtype: {value!r}")
        return mapping[normalized]

    @classmethod
    def convert_params(cls, attributes, testcase_name):
        if not isinstance(attributes, dict):
            raise ValueError("QSFAV2 input attributes must be a dictionary")
        missing = [
            f"pytest_{name}"
            for name in cls.REQUIRED_PARAM_NAMES
            if f"pytest_{name}" not in attributes
        ]
        if missing:
            raise ValueError(f"QSFAV2 CSV is missing explicit pytest fields: {missing}")
        pytest_params = {
            name: copy.deepcopy(attributes.get(f"pytest_{name}"))
            for name in cls.REQUIRED_PARAM_NAMES
        }
        for name in (
            "T1",
            "T2",
            "block_size",
            "block_num",
            "enable_sinks",
            "return_softmax_lse",
        ):
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
            "range_sinks",
        ):
            key = f"pytest_{name}"
            if key in attributes:
                pytest_params[name] = copy.deepcopy(attributes[key])
        if (
            pytest_params["actual_seq_q"] is None
            or pytest_params["actual_seq_kv"] is None
        ):
            raise ValueError(
                "QSFAV2 CSV must provide actual_seq_q and actual_seq_kv; "
                "random sequence generation is not allowed during TTK conversion"
            )

        combination = pytest_params
        combination["q_type"] = cls.query_dtype(combination["q_type"])
        # The pytest random generator accepts list ranges, not tuple ranges.
        for name, value in tuple(combination.items()):
            if name.startswith("range_") and isinstance(value, tuple):
                combination[name] = list(value)
        pytest_utils, pytest_golden = cls.load_modules()
        params = pytest_utils.convert_param_combination_to_cs_format(combination)
        params["case_name"] = testcase_name or params["case_name"]
        return params, pytest_golden


def cpu_kv_quant_sparse_flash_attention_v2(
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
    arguments = dict(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        key_quant_mode=key_quant_mode,
        value_quant_mode=value_quant_mode,
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


def integer_vector(tensor, name, count=None):
    if tensor is None:
        return None
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    values = tensor.detach().cpu().tolist()
    if count is not None and len(values) != count:
        raise ValueError(f"{name} must contain {count} values")
    return values


def hifloat8_values(raw):
    code = torch.arange(256, dtype=torch.int64)
    dot = (code & 120) >> 3
    exponent = (code & 7) - 23
    mantissa = torch.ones(256)
    for lower, bits, shift, negative, bias in (
        (1, 7, 0, 0, 0),
        (2, 7, 0, 0, 0),
        (4, 7, 3, 2, 2),
        (8, 3, 2, 4, 4),
        (12, 1, 1, 8, 8),
    ):
        mask = dot >= lower
        mantissa = torch.where(mask, 1.0 + (code & bits).float() / (bits + 1), mantissa)
        if lower == 1:
            current = torch.zeros_like(code)
        elif lower == 2:
            current = torch.where((code & 8) != 0, -1, 1)
        else:
            encoded = (
                code & (30 if lower == 12 else 28 if lower == 8 else 24)
            ) >> shift
            current = torch.where(encoded >= negative, -encoded, encoded + bias)
        exponent = torch.where(mask, current, exponent)
    values = (
        torch.where(code >= 128, -1.0, 1.0)
        * torch.pow(2.0, exponent.float())
        * mantissa
    )
    values[0], values[128], values[111], values[239] = (
        0,
        float("nan"),
        float("inf"),
        -float("inf"),
    )
    return values[raw.long()]


TQ4_CENTERS = (
    -0.12091285,
    -0.09111122,
    -0.07112455,
    -0.05513602,
    -0.04132067,
    -0.02874970,
    -0.01700489,
    -0.00568677,
    0.00547294,
    0.01680406,
    0.02857605,
    0.04108622,
    0.05492980,
    0.07101817,
    0.09115373,
    0.12037795,
)


def unpack_cache(tensor, query_dtype, dim, rope, tile, mode):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8)
    if mode == 2:
        expected = dim + rope * 2 + dim // tile * 4
        if dim % tile or raw.shape[-1] != expected:
            raise ValueError(
                f"packed per-tile KV must contain {expected} bytes per token"
            )
        latent_raw = raw[..., :dim].contiguous()
        if tensor.dtype == torch.int8:
            latent = latent_raw.view(torch.int8).float()
        elif tensor.dtype == torch.float8_e4m3fn:
            latent = latent_raw.view(torch.float8_e4m3fn).float()
        elif tensor.dtype == torch.uint8:
            latent = hifloat8_values(latent_raw)
        else:
            raise ValueError(f"unsupported packed KV dtype: {tensor.dtype}")
        scale = raw[..., dim + rope * 2 :].contiguous().view(torch.float32)
        scale = scale.repeat_interleave(tile, dim=-1)
    elif mode == 3:
        if dim % 2 or raw.shape[-1] != dim // 2 + rope * 2 + 2:
            raise ValueError("invalid TQ4 cache width")
        codes = raw[..., : dim // 2].long()
        indices = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
        latent = torch.tensor(TQ4_CENTERS, dtype=torch.float32)[indices]
        scale = raw[..., -2:].contiguous().view(torch.float16).float()
        dim //= 2
    else:
        raise ValueError(f"unsupported KV quantization mode: {mode}")
    rope_values = raw[..., dim : dim + rope * 2].contiguous().view(query_dtype).float()
    return latent, rope_values, scale


def prepare_reference(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=(1 << 63) - 1,
    next_tokens=(1 << 63) - 1,
    attention_mode=0,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    key_dtype=None,
    value_dtype=None,
    sinks=None,
    return_softmax_lse=False,
    **_context,
):
    """Decode the packed cache and expose visible rows with their output/stat indices.

    The returned rows() uses FP32 decoded operands. The ordinary reference requests
    rows(mixed=True) for its quantization-specific rounding before attention.
    """
    if query.device.type != "cpu" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QSFA CPU reference requires FP16/BF16 CPU inputs")
    if attention_mode != 2 or quant_scale_repo_mode != 1:
        raise ValueError("QSFA CPU reference requires MLA-absorb with packed scales")
    if key_dequant_scale is not None or value_dequant_scale is not None:
        raise ValueError("packed KV cannot use separate scale inputs")
    if pre_tokens != (1 << 63) - 1 or next_tokens != (1 << 63) - 1:
        raise ValueError("QSFA does not support finite pre_tokens/next_tokens")
    if sparse_mode not in (0, 3) or sparse_block_size < 1 or tile_size < 1:
        raise ValueError("unsupported sparse mode, block size or tile size")
    if layout_query not in ("BSND", "TND") or layout_kv not in (
        "BSND",
        "TND",
        "PA_BSND",
    ):
        raise ValueError("unsupported QSFA layout")
    dim = query.shape[-1] - rope_head_dim
    if dim <= 0 or rope_head_dim < 0:
        raise ValueError("invalid query/RoPE dimension")
    q_lengths = integer_vector(actual_seq_lengths_query, "actual_seq_lengths_query")
    if layout_query == "TND":
        if q_lengths is None or query.ndim != 3:
            raise ValueError("TND query requires cumulative sequence lengths")
        ends = q_lengths
        if not ends or ends[-1] != query.shape[0]:
            raise ValueError("query cumulative lengths must span the tensor")
        starts = [0] + ends[:-1]
        q_lengths = [end - start for start, end in zip(starts, ends)]
    else:
        if query.ndim != 4:
            raise ValueError("BSND query must have rank 4")
        q_lengths = (
            [query.shape[1]] * query.shape[0] if q_lengths is None else q_lengths
        )
        if len(q_lengths) != query.shape[0] or any(
            length > query.shape[1] for length in q_lengths
        ):
            raise ValueError("query lengths do not match BSND capacity")
        starts = [batch * query.shape[1] for batch in range(query.shape[0])]
    if any(length < 0 for length in q_lengths):
        raise ValueError("query lengths must be nonnegative")
    batch = len(q_lengths)
    kv_lengths = integer_vector(actual_seq_lengths_kv, "actual_seq_lengths_kv", batch)
    if layout_kv == "BSND":
        if key.ndim != 4 or key.shape[0] != batch:
            raise ValueError("KV BSND batch mismatch")
        kv_lengths = [key.shape[1]] * batch if kv_lengths is None else kv_lengths
        kv_starts = [b * key.shape[1] for b in range(batch)]
        if any(length < 0 or length > key.shape[1] for length in kv_lengths):
            raise ValueError("KV length exceeds BSND capacity")
    elif layout_kv == "TND":
        if kv_lengths is None or key.ndim != 3 or kv_lengths[-1] != key.shape[0]:
            raise ValueError("TND KV requires cumulative lengths spanning its tensor")
        kv_starts = [0] + kv_lengths[:-1]
        kv_lengths = [end - start for start, end in zip(kv_starts, kv_lengths)]
    else:
        if key.ndim != 4 or kv_lengths is None or block_table is None:
            raise ValueError("PA_BSND requires block_table and KV lengths")
        if block_table.ndim != 2 or block_table.shape[0] != batch:
            raise ValueError("block_table batch mismatch")
        if any(length > block_table.shape[1] * key.shape[1] for length in kv_lengths):
            raise ValueError("block_table has too few columns for KV lengths")
        kv_starts = [0] * batch
    if any(length < 0 for length in kv_lengths):
        raise ValueError("KV lengths must be nonnegative")
    if value.shape != key.shape or value.dtype != key.dtype:
        raise ValueError("key/value shapes or dtypes differ")
    if key_quant_mode != value_quant_mode or not torch.equal(
        key.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)
    ):
        raise ValueError(
            "MLA-absorb requires key and value to contain the same packed KV"
        )
    heads, kv_heads = query.shape[-2], key.shape[-2]
    if heads % kv_heads or sparse_indices.shape != (
        *query.shape[:-2],
        kv_heads,
        sparse_indices.shape[-1],
    ):
        raise ValueError("GQA heads or sparse_indices shape mismatch")
    group = heads // kv_heads
    key_parts = unpack_cache(
        key, query.dtype, dim, rope_head_dim, tile_size, key_quant_mode
    )
    q = query.reshape(-1, heads, query.shape[-1]).float()
    indices = sparse_indices.reshape(-1, kv_heads, sparse_indices.shape[-1]).cpu()
    output = torch.zeros((*query.shape[:-1], dim), dtype=torch.float32)
    if layout_query == "BSND":
        statistic_shape = (batch, kv_heads, query.shape[1], group)
    else:
        statistic_shape = (kv_heads, query.shape[0], group)
    statistics = padding_statistics(statistic_shape, output.device)
    sink_values = None if sinks is None else sinks.detach().cpu().float()
    if sink_values is not None and sink_values.shape != (heads,):
        raise ValueError("sinks must match query heads")

    def rows(mixed=False):
        latent, rope, scale = key_parts
        if key_quant_mode == 3 and mixed:
            latent = latent.to(query.dtype).double()
            rope, scale = rope.double(), scale.double()
        k = torch.cat(
            (latent * scale, rope * scale if key_quant_mode == 3 else rope), dim=-1
        )
        v = latent * scale
        if mixed and key_quant_mode != 3:
            k = k.to(query.dtype).float()
            v = v.to(query.dtype).float()
        k = k.reshape(-1, kv_heads, query.shape[-1])
        v = v.reshape(-1, kv_heads, dim)
        for b, length in enumerate(q_lengths):
            for row in range(length):
                flat_row = starts[b] + row
                visible = (
                    kv_lengths[b]
                    if sparse_mode == 0
                    else max(0, kv_lengths[b] - length + row + 1)
                )
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
                        pages = block_table[b, logical // key.shape[1]].cpu().long()
                        if torch.any((pages < 0) | (pages >= key.shape[0])):
                            raise ValueError(
                                "block_table contains an invalid physical page"
                            )
                        physical = pages * key.shape[1] + logical % key.shape[1]
                    else:
                        physical = logical + kv_starts[b]
                    head_slice = slice(head * group, (head + 1) * group)
                    out_index = (
                        (b, row, head_slice)
                        if layout_query == "BSND"
                        else (flat_row, head_slice)
                    )
                    stat_index = (
                        (b, head, row) if layout_query == "BSND" else (head, flat_row)
                    )
                    sink = None if sink_values is None else sink_values[head_slice]
                    yield (
                        q[flat_row, head_slice],
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
    key_quant_mode,
    value_quant_mode,
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
            "[%s] QSFAV2 FP32 golden: preparing final inputs for pytest compute_golden_row, layout=%s/%s",
            testcase_name,
            kwargs.get("layout_query", "BSND"),
            kwargs.get("layout_kv", "BSND"),
        )
    if (key_quant_mode, value_quant_mode) != (2, 2):
        raise ValueError("QSFA V2 reference supports per-tile quantization only")
    if kwargs.get("sinks") is None:
        kwargs["sinks"] = query.new_zeros(query.shape[-2], dtype=torch.float32)
    output, statistics, rows, dtype, scale = prepare_reference(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        **kwargs,
    )
    if promote:
        logging.info(
            "[%s] QSFAV2 pytest FP32: QK -> softmax -> PV, scale=%s, kv_mode=%s/%s",
            testcase_name,
            scale,
            key_quant_mode,
            value_quant_mode,
        )
    rope = kwargs.get("rope_head_dim", 64)
    for q, k, v, sink, out_index, stat_index in rows(mixed=not promote):
        if not k.shape[0]:
            statistics[0][stat_index], statistics[1][stat_index] = sink, 1.0
            continue
        result, maximum, total = compute_golden_row(
            q, k, v, scale, sink, dtype, rope, promote=promote
        )
        output[out_index] = result
        statistics[0][stat_index], statistics[1][stat_index] = (
            maximum[:, 0],
            total[:, 0],
        )
    if promote:
        logging.info(
            "[%s] QSFAV2 FP32 golden ready: attention=%s %s",
            testcase_name,
            tuple(output.shape),
            output.dtype,
        )
    else:
        finite = torch.isfinite(output)
        output = torch.where(
            finite,
            output.clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max),
            output,
        ).to(dtype)
    if not kwargs.get("return_softmax_lse", False):
        return (
            output,
            output.new_empty(0, dtype=torch.float32),
            output.new_empty(0, dtype=torch.float32),
        )
    return output, *statistics


def mixed_reference(*args, **kwargs):
    """Use the ordinary precision contract for the existing two-party and provider callers."""
    return reference_from_inputs(*args, promote=False, **kwargs)
