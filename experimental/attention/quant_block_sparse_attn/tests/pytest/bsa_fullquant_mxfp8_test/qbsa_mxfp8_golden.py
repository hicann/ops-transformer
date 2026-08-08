#!/usr/bin/python3
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
"""
MXFP8 Flash Attention Golden

功能：参考式生成输入 → CPU golden → NPU 调用 → layout 对齐 → 精度对比
支持：Q/QDescale 公共输入为 TND 或 NTD；K/V 生成后在本文件内部按 BNSD 保存，NPU 侧只支持 PA KV cache
支持：PA 场景、GQA、causal/dense sparse mode、QDescale 固定 TND D-group 打包
数据：随机 FP8 Q/K/V + MXFP8 D-group descale，生成结构对齐 quant_block_sparse_attn_golden.py
输出：逐元素表格 + 统计汇总 (PctRlt 通过率，双千分之五标准)

"""

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

try:
    from . import result_compare_method
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QBSA_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
TORCH_OPS_EXTENSION_DIR = os.path.join(QBSA_ROOT, "torch_ops_extension")
_CUSTOM_OPS_IMPORTED = False
_DEFAULT_CASE_FILE = "qbsa_mxfp8_test_cases"
_DEFAULT_CASE_CSV = "qbsa_mxfp8_test_cases.csv"
_CASE_LOG_WIDTH = 100
_DTYPE_MAP = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e8m0fnu": torch.float8_e8m0fnu,
}

_CASE_BOOL_FIELDS = {
    "enable",
    "is_contiguous",
    "return_softmax_lse",
    "empty_actual_seq",
}
_CASE_INT_FIELDS = {
    "B",
    "N1",
    "N2",
    "D",
    "S1",
    "S2",
    "block_size",
    "mask_mode",
    "quant_group_size",
    "seed",
    "device_id",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "sparse_count",
    "quant_mode",
    "s2_base_size",
}
_CASE_FLOAT_FIELDS = {
    "p_scale_value",
    "softmax_scale",
}
_CASE_RANGE_FIELDS = {"data_range_q", "data_range_k", "data_range_v"}
_CASE_INT_LIST_FIELDS = {"actual_seq_q", "actual_seq_kv"}
_CASE_REQUIRED_RUNTIME_FIELDS = {
    "B",
    "N1",
    "N2",
    "D",
    "S1",
    "S2",
    "actual_seq_q",
    "actual_seq_kv",
    "s2_base_size",
    "block_size",
    "mask_mode",
    "fp8_dtype",
    "scale_dtype",
    "quant_group_size",
    "layout_q",
    "layout_kv",
    "layout_sparse_indices",
    "layout_out",
    "kv_cache_layout",
    "is_contiguous",
    "return_softmax_lse",
    "seed",
    "data_range_q",
    "data_range_k",
    "data_range_v",
    "device_id",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "sparse_count",
    "quant_mode",
    "sparse_pattern",
    "block_table_pattern",
}

# 其余列作为用例设计参考信息原样保留，不参与 pytest 计算。CSV 按列名解析，允许继续携带扩展字段。


def _ensure_custom_ops_registered():
    """Import custom_ops before calling torch.ops.custom QBSA APIs."""
    global _CUSTOM_OPS_IMPORTED
    if _CUSTOM_OPS_IMPORTED:
        return

    try:
        import custom_ops  # noqa: F401
    except ImportError as install_import_error:
        sys.modules.pop("custom_ops", None)
        if TORCH_OPS_EXTENSION_DIR not in sys.path:
            sys.path.insert(0, TORCH_OPS_EXTENSION_DIR)
        try:
            import custom_ops  # noqa: F401
        except ImportError as source_import_error:
            raise RuntimeError(
                "import custom_ops failed. Please build/install QBSA PTA extension first, for example: "
                "`cd experimental/attention/quant_block_sparse_attn/torch_ops_extension && bash build_and_install.sh`. "
                f"Installed-package import error: {install_import_error}. "
                f"Source-tree import error: {source_import_error}."
            ) from source_import_error

    if not hasattr(torch.ops, "custom") or not hasattr(
        torch.ops.custom, "npu_quant_block_sparse_attn"
    ):
        raise RuntimeError(
            "torch.ops.custom.npu_quant_block_sparse_attn is not registered after importing custom_ops"
        )
    if not hasattr(torch.ops.custom, "npu_quant_block_sparse_attn_metadata"):
        raise RuntimeError(
            "torch.ops.custom.npu_quant_block_sparse_attn_metadata is not registered after importing custom_ops"
        )
    _CUSTOM_OPS_IMPORTED = True


def _load_case_file_test_cases(case_file):
    case_file = case_file.strip()
    case_file_no_ext = os.path.splitext(os.path.basename(case_file))[0]
    case_path = case_file
    if not case_path.endswith(".py"):
        case_path = f"{case_path}.py"
    if not os.path.isabs(case_path):
        case_path = os.path.join(CURRENT_DIR, case_path)
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"case file not found: {case_path}")

    module_name = f"_qbsa_mxfp8_cases_{case_file_no_ext}"
    spec = importlib.util.spec_from_file_location(module_name, case_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"load case file failed: {case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "TestCases"):
        raise AttributeError(f"{case_path} must define TestCases")
    return module.TestCases


def _iter_test_cases(test_cases):
    if isinstance(test_cases, dict):
        for case_name, case in test_cases.items():
            item = dict(case)
            item.setdefault("name", case_name)
            yield item
        return
    for case in test_cases:
        yield dict(case)


def _parse_bool(value, field_name):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field_name} must be true or false, got: {value}")


def _parse_case_field(field_name, value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if field_name in _CASE_BOOL_FIELDS:
        return _parse_bool(value, field_name)
    if field_name in _CASE_INT_FIELDS:
        return int(value)
    if field_name in _CASE_RANGE_FIELDS:
        parsed = json.loads(value) if isinstance(value, str) and value.startswith("[") else value
        if isinstance(parsed, (list, tuple)):
            if len(parsed) != 2:
                raise ValueError(f"{field_name} must contain exactly [min, max], got: {value}")
            return [float(parsed[0]), float(parsed[1])]
        return float(parsed)
    if field_name in _CASE_FLOAT_FIELDS:
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return float(value)
    if field_name in _CASE_INT_LIST_FIELDS:
        parsed = value if isinstance(value, list) else json.loads(str(value))
        if not isinstance(parsed, list):
            raise ValueError(f"{field_name} must be a JSON list, got: {value}")
        return [int(item) for item in parsed]
    return value


def _resolve_case_csv_path(case_csv):
    case_path = case_csv.strip()
    if not case_path.endswith(".csv"):
        case_path = f"{case_path}.csv"
    if not os.path.isabs(case_path):
        case_path = os.path.join(CURRENT_DIR, case_path)
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"case CSV not found: {case_path}")
    return case_path


def _load_case_csv(case_csv):
    case_path = _resolve_case_csv_path(case_csv)
    with open(case_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"{case_path} must contain a name column")
        for row_number, row in enumerate(reader, start=2):
            if not any(value is not None and value.strip() for value in row.values()):
                continue
            try:
                yield {
                    field: _parse_case_field(field, value)
                    for field, value in row.items()
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid case data in {case_path}:{row_number}: {error}"
                ) from error


def _resolve_dtype(value, field_name):
    if not isinstance(value, str):
        return value
    if value not in _DTYPE_MAP:
        raise ValueError(f"unsupported {field_name}: {value}")
    return _DTYPE_MAP[value]


def _validate_case_design_fields(case):
    """只校验实际参与运行的字段；参考字段允许由外部泛化表自由扩展。"""
    if case["N2"] <= 0 or case["N1"] % case["N2"] != 0:
        raise ValueError(
            f"N1 must be divisible by N2, got N1={case['N1']}, N2={case['N2']}"
        )

    if case["s2_base_size"] != 512:
        raise ValueError(
            f"s2_base_size must be 512 for MXFP8, got {case['s2_base_size']}"
        )

    block_size = case.get("block_size")
    sparse_q_block_size = case.get("sparse_q_block_size")
    sparse_kv_block_size = case.get("sparse_kv_block_size")
    if sparse_q_block_size is not None and sparse_kv_block_size is not None:
        if sparse_q_block_size != sparse_kv_block_size:
            raise ValueError(
                "sparse_q_block_size must equal sparse_kv_block_size, "
                f"got {sparse_q_block_size}, {sparse_kv_block_size}"
            )
    if block_size is not None and sparse_kv_block_size is not None:
        if block_size % sparse_kv_block_size != 0:
            raise ValueError(
                "block_size must be a multiple of sparse_kv_block_size, "
                f"got block_size={block_size}, sparse_kv_block_size={sparse_kv_block_size}"
            )

    _get_runtime_seq_lengths(case, "actual_seq_q", "S1")
    _get_runtime_seq_lengths(case, "actual_seq_kv", "S2")


def _get_runtime_seq_lengths(case, field_name, total_field_name):
    """支持逐 batch 长度，也支持用单元素总 T 表示等长 batch。"""
    values = [int(item) for item in case[field_name]]
    batch = int(case["B"])
    if len(values) == batch:
        return values
    if len(values) == 1 and batch > 1:
        total_size = values[0]
        if total_size != int(case[total_field_name]) or total_size % batch != 0:
            raise ValueError(
                f"single-value {field_name} must equal {total_field_name} and be divisible by B, "
                f"got {values}, {total_field_name}={case[total_field_name]}, B={batch}"
            )
        return [total_size // batch] * batch
    raise ValueError(
        f"{field_name} length must be B or 1, got {len(values)} for B={batch}"
    )


def _normalize_case(case):
    case = dict(case)
    missing_fields = sorted(
        field for field in _CASE_REQUIRED_RUNTIME_FIELDS if case.get(field) is None
    )
    if missing_fields:
        raise ValueError(
            f"case {case.get('name', '<unnamed>')} missing required runtime fields: {missing_fields}"
        )
    case["fp8_dtype"] = _resolve_dtype(case["fp8_dtype"], "fp8_dtype")
    case["scale_dtype"] = _resolve_dtype(case["scale_dtype"], "scale_dtype")
    if case.get("enable") is None:
        case["enable"] = True
    case["softmax_scale"] = case.get("softmax_scale") or (
        1.0 / math.sqrt(float(case["D"]))
    )
    _validate_case_design_fields(case)
    return case


def load_test_cases(case_files=None, use_csv=False):
    case_files = case_files or (_DEFAULT_CASE_CSV if use_csv else _DEFAULT_CASE_FILE)
    all_cases = {}
    for case_file in [item.strip() for item in case_files.split(",") if item.strip()]:
        if use_csv:
            test_cases = _load_case_csv(case_file)
        else:
            test_cases = _iter_test_cases(_load_case_file_test_cases(case_file))
        for case in test_cases:
            case = _normalize_case(case)
            case_name = case.get("name")
            if not case_name:
                raise ValueError(f"case in {case_file} must set name")
            if case_name in all_cases:
                raise ValueError(f"duplicate case name: {case_name}")
            all_cases[case_name] = case
    return all_cases


def resolve_case_names(case_name_arg, all_cases):
    if not case_name_arg or case_name_arg == "all":
        return [name for name, case in all_cases.items() if case.get("enable", True)]
    result = []
    missing = []
    for case_name in [
        item.strip() for item in case_name_arg.split(",") if item.strip()
    ]:
        if case_name in all_cases:
            result.append(case_name)
        else:
            missing.append(case_name)
    if missing:
        raise ValueError(
            f"Unknown case_name: {missing}. Available cases: {sorted(all_cases)}"
        )
    return result


def _load_initial_case():
    all_cases = load_test_cases(_DEFAULT_CASE_FILE)
    enabled_cases = resolve_case_names(None, all_cases)
    if enabled_cases:
        return dict(all_cases[enabled_cases[0]])
    if all_cases:
        return dict(next(iter(all_cases.values())))
    raise ValueError(f"{_DEFAULT_CASE_FILE}.py must define at least one case")


def set_active_case(case):
    global CASE, B, N_q, N_kv, D, ACTUAL_SEQ_Q, ACTUAL_SEQ_KV, BLOCK_SIZE, MASK_MODE
    global \
        FP8_DTYPE, \
        SCALE_DTYPE, \
        QUANT_GROUP_SIZE, \
        Q_INPUT_LAYOUT, \
        KV_CACHE_LAYOUT, \
        IS_CONTIGUOUS
    global \
        ENABLE_LSE, \
        DATA_RANGE_Q, \
        DATA_RANGE_K, \
        DATA_RANGE_V, \
        DEVICE_ID, \
        SPARSE_BLOCK_SIZE
    global QUANT_MODE_MXFP8, KEY_LAYOUT, SPARSE_INDICES_LAYOUT, OUT_LAYOUT

    CASE = dict(case)
    B = CASE["B"]
    N_q = CASE["N1"]
    N_kv = CASE["N2"]
    D = CASE["D"]
    ACTUAL_SEQ_Q = CASE["actual_seq_q"]
    ACTUAL_SEQ_KV = CASE["actual_seq_kv"]
    BLOCK_SIZE = CASE["block_size"]
    MASK_MODE = CASE["mask_mode"]
    FP8_DTYPE = CASE["fp8_dtype"]
    SCALE_DTYPE = CASE["scale_dtype"]
    QUANT_GROUP_SIZE = CASE["quant_group_size"]
    Q_INPUT_LAYOUT = CASE["layout_q"]
    KV_CACHE_LAYOUT = CASE["kv_cache_layout"]
    IS_CONTIGUOUS = CASE["is_contiguous"]
    ENABLE_LSE = CASE["return_softmax_lse"]
    DATA_RANGE_Q = CASE["data_range_q"]
    DATA_RANGE_K = CASE["data_range_k"]
    DATA_RANGE_V = CASE["data_range_v"]
    DEVICE_ID = CASE["device_id"]
    SPARSE_BLOCK_SIZE = CASE["sparse_q_block_size"]
    QUANT_MODE_MXFP8 = CASE["quant_mode"]
    KEY_LAYOUT = CASE["layout_kv"]
    SPARSE_INDICES_LAYOUT = CASE["layout_sparse_indices"]
    OUT_LAYOUT = CASE["layout_out"]


CASE = _load_initial_case()
set_active_case(CASE)

# ==============================================================================
# 固定运行常量
# ==============================================================================
# Q/K/V 使用 fp8_e4m3fn；Q/K/V descale 与 P scale 使用 fp8_e8m0。
# e8m0fnu 最小正数: 2^(-127)，用于替换 descale 中的 0 和非有限值
# e8m0fnu 没有 0 值语义，0 的 biased exponent 会被 NPU 解释为 NaN
E8M0_MIN_POSITIVE = 2 ** (-127)
_EMAX_MAP = {
    torch.float8_e4m3fn: 8,
}

# ==============================================================================
# 参考式数据生成 helper
# 与 quant_block_sparse_attn_golden.py 保持同一组织方式：
#   case -> rng/generator -> query/key/value/descale/block_table/sparse_indices
# 当前文件的 MXFP8 full-quant 差异：
#   - Q/QDescale 直接生成 TND 或 NTD，不经过 BNSD 中间格式
#   - K/V 先按参考文件的 BSND 语义生成，再适配为本文件 CPU/PA 路径使用的 BNSD
#   - Q/K/V descale 由对应 FP8 数据按 MXFP8 group 规则生成
# ==============================================================================


def _prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _normalize_data_range(data_range):
    """将标量或 [最小值, 最大值] 转换为统一的范围描述。"""
    if isinstance(data_range, (list, tuple)):
        if len(data_range) != 2:
            raise ValueError(
                f"data_range must be a scalar or [min, max], got: {data_range}"
            )
        low, high = float(data_range[0]), float(data_range[1])
        explicit_bounds = True
    else:
        radius = float(data_range)
        if radius < 0:
            raise ValueError(f"scalar data_range must be non-negative, got: {radius}")
        low, high = -radius, radius
        explicit_bounds = False

    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"data_range bounds must be finite, got: [{low}, {high}]")
    if low > high:
        raise ValueError(f"data_range min must not exceed max, got: [{low}, {high}]")
    return low, high, explicit_bounds


def _rand_float(shape, generator, data_range=1.0):
    low, high, explicit_bounds = _normalize_data_range(data_range)
    if low == high:
        return torch.full(shape, low, dtype=torch.float32)

    result = (
        torch.rand(shape, dtype=torch.float32, generator=generator) * (high - low)
        + low
    )
    if explicit_bounds and result.numel() > 0:
        # 显式 [a, b] 表示闭区间，固定首尾元素以确保两个端点都被覆盖。
        flattened = result.view(-1)
        flattened[0] = low
        if flattened.numel() > 1:
            flattened[-1] = high
    return result


def _log_tensor_ranges(prefix, **tensors):
    """将低精度 Tensor 转为 FP32 后打印最大值和最小值。"""
    ranges = []
    for name, tensor in tensors.items():
        if tensor is None:
            ranges.append(f"{name}=None")
            continue
        if tensor.numel() == 0:
            ranges.append(f"{name}=empty")
            continue
        values = tensor.detach().to(dtype=torch.float32)
        ranges.append(
            f"{name}=[min={values.min().item():.8g}, max={values.max().item():.8g}]"
        )
    logger.info("[INFO] %s: %s", prefix, ", ".join(ranges))


def _rand_fp8(shape, generator, data_range=1.0):
    return _rand_float(shape, generator, data_range).to(FP8_DTYPE)


def _validate_fp8_dtype(fp8_dtype):
    if fp8_dtype not in _EMAX_MAP:
        raise ValueError(
            f"Unsupported FP8 dtype: {fp8_dtype}; only torch.float8_e4m3fn is supported"
        )


def quantize_mxfp8_qk(tensor_fp32, fp8_dtype, group_size=32):
    """MXFP8 配套量化 Q/K: per-32-element-group along D axis。

    正确流程: 从 fp32 算 descale → fp8 = clamp(fp32/descale)。
    保证 fp8 × descale ≈ 原始 fp32，反量化不溢出。

    Input:  (..., D) fp32
    Output: (fp8 (..., D), descale (..., D//group_size))  -- descale 与 fp8 配套
    """
    _validate_fp8_dtype(fp8_dtype)
    tensor_fp32 = tensor_fp32.to(torch.float32)

    last_dim = tensor_fp32.shape[-1]
    num_groups = math.ceil(last_dim / group_size)
    pad_size = num_groups * group_size - last_dim
    if pad_size > 0:
        tensor_fp32 = torch.nn.functional.pad(tensor_fp32, (0, pad_size))

    grouped = tensor_fp32.reshape(*tensor_fp32.shape[:-1], num_groups, group_size)
    all_zero_mask = torch.all(grouped == 0, dim=-1)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    # Choose the smallest power-of-two descale that keeps the whole group in
    # the finite E4M3 range.  floor(log2(max)) - emax only accounts for the
    # exponent bits and can still overflow values between 256 and 448.
    fp8_max = torch.finfo(fp8_dtype).max
    shared_exp = torch.ceil(torch.log2(max_vals / fp8_max))
    descale = torch.where(all_zero_mask, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )

    descale_expanded = descale.repeat_interleave(group_size, dim=-1)[..., :last_dim]
    quantized = (
        (tensor_fp32[..., :last_dim] / descale_expanded)
        .clamp(-448.0, 448.0)
        .to(fp8_dtype)
    )
    return quantized, descale


def quantize_mxfp8_v(tensor_fp32, fp8_dtype, group_size=32):
    """MXFP8 配套量化 V: per-32-element-group along S axis。

    Input:  (B, S, N, D) fp32 = BSND
    Output: (fp8 (B, S, N, D), descale (B, S//group_size, N, D))  -- descale 与 fp8 配套
    """
    _validate_fp8_dtype(fp8_dtype)
    tensor_fp32 = tensor_fp32.to(torch.float32)

    batch, seq_len, num_heads, head_dim = tensor_fp32.shape
    num_groups = math.ceil(seq_len / group_size)
    pad_size = num_groups * group_size - seq_len
    if pad_size > 0:
        tensor_fp32 = torch.nn.functional.pad(tensor_fp32, (0, 0, 0, 0, 0, pad_size))

    grouped = tensor_fp32.reshape(batch, num_groups, group_size, num_heads, head_dim)
    all_zero_mask = torch.all(grouped == 0, dim=2)
    max_vals = torch.max(torch.abs(grouped), dim=2)[0].clamp(min=1e-12)
    fp8_max = torch.finfo(fp8_dtype).max
    shared_exp = torch.ceil(torch.log2(max_vals / fp8_max))
    descale = torch.where(all_zero_mask, torch.ones_like(shared_exp), 2**shared_exp).to(
        torch.float32
    )

    descale_expanded = descale.repeat_interleave(group_size, dim=1)[:, :seq_len, :, :]
    quantized = (
        (tensor_fp32[:, :seq_len, :, :] / descale_expanded)
        .clamp(-448.0, 448.0)
        .to(fp8_dtype)
    )
    return quantized, descale


def _physical_ids(start, count, pattern, rng):
    ids = list(range(start, start + count))
    if pattern == "reverse":
        ids.reverse()
    elif pattern == "random":
        rng.shuffle(ids)
    elif pattern != "sequential":
        raise ValueError(f"Unsupported block table pattern: {pattern}")
    return ids


def _make_reference_block_table(batch, seq_lens, block_size, pattern, rng):
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens] * batch
    block_nums = [math.ceil(int(seq_len) / block_size) for seq_len in seq_lens]
    max_block_num = max(block_nums) if block_nums else 0
    block_table = torch.full((batch, max_block_num), fill_value=-1, dtype=torch.int32)

    total_logical_blocks = sum(block_nums)
    if total_logical_blocks == 0:
        return block_table

    # 与 common MX 一致：物理页数由实际 KV 序列自动推导，每个逻辑页使用唯一物理页。
    physical_ids = _physical_ids(0, total_logical_blocks, pattern, rng)

    logical_offset = 0
    for batch_idx, logical_block_num in enumerate(block_nums):
        block_table[batch_idx, :logical_block_num] = torch.tensor(
            physical_ids[logical_offset : logical_offset + logical_block_num],
            dtype=torch.int32,
        )
        logical_offset += logical_block_num
    return block_table


def _allowed_blocks(
    mask_mode, qb_idx, sparse_q_block_size, sparse_kv_block_size, q_len, kv_len
):
    block_num = math.ceil(kv_len / sparse_kv_block_size)
    if block_num <= 0:
        return []
    if mask_mode == 0:
        return list(range(block_num))
    if mask_mode != 3:
        raise ValueError(f"Unsupported sparse mode: {mask_mode}")

    max_token = (qb_idx + 1) * sparse_q_block_size - 1 + kv_len - q_len
    if max_token < 0:
        return []
    max_block = min(block_num - 1, max_token // sparse_kv_block_size)
    return list(range(max_block + 1))


def _select_blocks(blocks, sparse_count, pattern, rng):
    if sparse_count <= 0 or not blocks or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal"):
        return blocks[: min(sparse_count, len(blocks))]
    if pattern == "reverse":
        return list(reversed(blocks[-min(sparse_count, len(blocks)) :]))
    if pattern == "tail":
        selected = blocks[: max(0, min(sparse_count, len(blocks)) - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:sparse_count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[: min(sparse_count, len(selected))]
    raise ValueError(f"Unsupported sparse pattern: {pattern}")


def _make_reference_sparse_indices(case, q_lengths, kv_lengths, rng):
    batch = case["B"]
    n1 = case["N1"]
    qb_max = math.ceil(max(q_lengths) / case["sparse_q_block_size"])
    kv_max = math.ceil(max(kv_lengths) / case["sparse_kv_block_size"])
    sparse_indices = torch.full(
        (batch, n1, qb_max, kv_max), fill_value=-1, dtype=torch.int32
    )
    sparse_seq_len = torch.zeros((batch, n1, qb_max), dtype=torch.int32)

    for batch_idx in range(batch):
        qb_batch = math.ceil(q_lengths[batch_idx] / case["sparse_q_block_size"])
        for qb_idx in range(qb_max):
            allowed = _allowed_blocks(
                case["mask_mode"],
                qb_idx,
                case["sparse_q_block_size"],
                case["sparse_kv_block_size"],
                q_lengths[batch_idx],
                kv_lengths[batch_idx],
            )
            for head_idx in range(n1):
                if qb_idx >= qb_batch:
                    selected = []
                else:
                    selected = _select_blocks(
                        allowed, case["sparse_count"], case["sparse_pattern"], rng
                    )
                sparse_seq_len[batch_idx, head_idx, qb_idx] = len(selected)
                if selected:
                    sparse_indices[batch_idx, head_idx, qb_idx, : len(selected)] = (
                        torch.tensor(selected, dtype=torch.int32)
                    )
    return sparse_indices, sparse_seq_len


def _make_reference_sparse_for_lengths(actual_seq_q, actual_seq_kv):
    case = dict(CASE)
    case["S1"] = max(int(item) for item in actual_seq_q)
    case["S2"] = max(int(item) for item in actual_seq_kv)
    case["sparse_count"] = math.ceil(case["S2"] / case["sparse_kv_block_size"])
    return _make_reference_sparse_indices(
        case,
        [int(item) for item in actual_seq_q],
        [int(item) for item in actual_seq_kv],
        random.Random(case["seed"]),
    )


# ==============================================================================
# Layout 转换函数 - 数据
# Q/QDescale 公共输入只接受 TND/NTD；BNSD 仅作为 K/V 和 PA KV cache 的内部布局。
# ==============================================================================


def canonical_q_input_layout(layout=None):
    layout = (layout or Q_INPUT_LAYOUT).upper()
    if layout not in ("TND", "NTD"):
        raise ValueError(f"Unsupported Q input layout: {layout}, expected TND or NTD")
    return layout


def convert_q_tnd_or_ntd_to_tnd(tensor, seq_lens, num_heads, name="Q"):
    """Q/QDescale input must be 3D TND or NTD; normalize to TND."""
    if tensor.dim() != 3:
        raise ValueError(
            f"{name} must be TND/NTD rank-3 input, got rank {tensor.dim()} and shape {tuple(tensor.shape)}"
        )

    total_s = sum(seq_lens)
    if tensor.shape[0] == total_s and tensor.shape[1] == num_heads:
        return tensor.contiguous()
    if tensor.shape[0] == num_heads and tensor.shape[1] == total_s:
        return tensor.permute(1, 0, 2).contiguous()

    raise ValueError(
        f"Unsupported {name} shape: {tuple(tensor.shape)}, expected TND=({total_s}, {num_heads}, D) "
        f"or NTD=({num_heads}, {total_s}, D)"
    )


def convert_q_tnd_or_ntd_to_layout(tensor, seq_lens, num_heads, layout=None, name="Q"):
    layout = canonical_q_input_layout(layout)
    tensor_tnd = convert_q_tnd_or_ntd_to_tnd(tensor, seq_lens, num_heads, name)
    if layout == "TND":
        return tensor_tnd
    return tensor_tnd.permute(1, 0, 2).contiguous()


# ==============================================================================
# Layout 转换函数 - Descale
# QDescale 公共输入为 TND/NTD；K/V descale 在生成后以内部 BNSD 保存并按 NPU layout 打包。
# ==============================================================================


def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 → e8m0fnu (uint8)，提取 IEEE 754 biased exponent
    e8m0fnu 格式: 只有指数位，没有尾数，表示 2^(e-127)
    biased exponent = 0xFF 时表示 NaN
    """
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0fnu 没有 0 值语义；非有限值进入 0xFF 会在 NPU 侧变 NaN。"""
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad_mask = ~torch.isfinite(result)
    zero_mask = result == 0
    bad_count = int(bad_mask.sum().item())
    zero_count = int(zero_mask.sum().item())
    if bad_count:
        logger.info(
            "[WARN] %s: replace %d non-finite scale values before e8m0 packing",
            name,
            bad_count,
        )
        result[bad_mask] = E8M0_MIN_POSITIVE
    if zero_count:
        result[zero_mask] = E8M0_MIN_POSITIVE
    return result


def fp32_to_e8m0fnu_safe(scale, name="scale"):
    scale_safe = sanitize_e8m0_scale(scale, name)
    packed = fp32_to_e8m0fnu(scale_safe)
    nan_byte_count = int((packed == 0xFF).sum().item())
    if nan_byte_count:
        raise ValueError(
            f"{name}: {nan_byte_count} values would become e8m0fnu NaN (0xFF)"
        )
    return packed.view(SCALE_DTYPE)


def pack_qk_scale_for_npu(scale_flat):
    """Pack Q/K D-group descale: (..., Dg) -> (..., Dg//2, 2).
    NPU 要求 Q/K descale 按 (偶, 奇) 对打包，每两个相邻 scale 值合并为一个 [..., 2]
    """
    orig_shape = scale_flat.shape
    last_dim = orig_shape[-1]
    new_shape = orig_shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)


def convert_q_scale_tnd_or_ntd_to_layout(scale, seq_lens):
    """QDescale public input TND/NTD -> packed NPU TND descale layout."""
    scale_tnd = convert_q_tnd_or_ntd_to_tnd(scale, seq_lens, N_q, "Q descale")
    return pack_qk_scale_for_npu(scale_tnd)


# ==============================================================================
# PA 格式转换 - mxfp8_pa_preprocessing
# 仅处理 K/V 及 K/V descale；Q/QDescale 不走 PA 预处理。
# ==============================================================================


def mxfp8_pa_preprocessing(
    tensor_bnsd,
    seq_lens,
    block_size,
    block_table,
    is_vscale=False,
    is_scale=False,
    kv_layout="BnNBsD",
    group_size=32,
    sparse_indices=None,
    sparse_seq_len=None,
):
    """
    MXFP8 PA 预处理: internal K/V BNSD -> PagedAttention KV Cache

    输入: [B, N, S, D]，仅用于 K/V 或 K/V descale
    输出 (is_scale=False): [BlockNum, N, BlockSize, D] (fp8 K/V)
    输出 (is_scale=True, is_vscale=False): [BlockNum, N, BlockSize, D//64, 2] (K descale)
    输出 (is_scale=True, is_vscale=True): [BlockNum, N, BlockSize//64, D, 2] (V descale)

    kv_layout:
      - BnNBsD: fp8=[Bn,N,Bs,D], KDescale=[Bn,N,Bs,D//64,2], VDescale=[Bn,N,Bs//64,D,2]
      - BnBsND: fp8=[Bn,Bs,N,D], KDescale=[Bn,Bs,N,D//64,2], VDescale=[Bn,Bs//64,N,D,2]
    """
    tensor_bnsd = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    B, N, S, D = tensor_bnsd.shape
    physical_block_num = (
        int(block_table.max().item()) + 1 if block_table.numel() > 0 else 0
    )

    if is_scale:
        if is_vscale:
            tensor_processed = convert_v_scale_to_pa(tensor_bnsd, seq_lens, group_size)
            v_descale_pack_ratio = group_size * 2
            pack_seq_lens = [
                math.ceil(act_s / v_descale_pack_ratio) for act_s in seq_lens
            ]
            pack_block_size = math.ceil(block_size / v_descale_pack_ratio)
            out_shape = (physical_block_num, N, pack_block_size, D, 2)
        else:
            if D % 2 != 0:
                raise ValueError("K descale Dg must be even for pair packing")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            pack_seq_lens = seq_lens
            pack_block_size = block_size
            out_shape = (physical_block_num, N, block_size, D // 2, 2)
    else:
        tensor_processed = tensor_bnsd
        pack_seq_lens = seq_lens
        pack_block_size = block_size
        out_shape = (physical_block_num, N, block_size, D)

    fill_value = E8M0_MIN_POSITIVE if is_scale else 0

    out_cache = torch.full(
        out_shape,
        fill_value,
        dtype=tensor_processed.dtype,
        device=tensor_processed.device,
    )
    block_num = [math.ceil(act_s / pack_block_size) for act_s in pack_seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        # for n_q_idx in range(N_q):
        for blk_idx in range(block_num[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * pack_block_size
            # block_offset = sparse_indices[b,n_q_idx,blk_idx] * pack_block_size
            valid_len = min(pack_block_size, pack_seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[blockid, :, :valid_len] = tensor_processed[
                b, :, block_offset : block_offset + valid_len
            ]

    if kv_layout == "BnNBsD":
        return out_cache
    elif kv_layout == "BnBsND":
        return out_cache.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def convert_v_scale_to_pa(scale_bnsd, seq_lens, group_size=32):
    """
    V descale 偶奇行交错 packing，用于 PA 预处理
    输入: [B, N, Sg, D], Sg = ceil(orgS/32)
    输出: [B, N, Sg//2, D, 2]，奇数行 pad 到偶数
    """
    scale_bnsd = (
        scale_bnsd
        if isinstance(scale_bnsd, torch.Tensor)
        else torch.as_tensor(scale_bnsd)
    )
    B, N, _, D = scale_bnsd.shape
    max_org_s = max(seq_lens)
    actual_Sg = math.ceil(max_org_s / group_size)

    transposed = scale_bnsd[:, :, :actual_Sg, :]

    if actual_Sg % 2 != 0:
        pad = torch.full(
            (B, N, 1, D),
            E8M0_MIN_POSITIVE,
            dtype=transposed.dtype,
            device=transposed.device,
        )
        transposed = torch.cat([transposed, pad], dim=2)
        actual_Sg += 1

    S_out = actual_Sg // 2
    result = torch.zeros(
        (B, N, S_out, D, 2), dtype=torch.float32, device=scale_bnsd.device
    )
    result[..., 0] = transposed[..., ::2, :]
    result[..., 1] = transposed[..., 1::2, :]
    return result


# ==============================================================================
# 数据生成
# ==============================================================================


def _generate_reference_style_mxfp8_inputs():
    """Generate test inputs with paired fp8 + descale.

    流程: fp32 随机 → 从 fp32 算 descale → fp8 = clamp(fp32/descale)
    fp8 和 descale 配套生成，保证 fp8 × descale ≈ 原始 fp32，反量化不溢出。
    """
    case = dict(CASE)
    rng = random.Random(case["seed"])
    generator = torch.Generator().manual_seed(case["seed"])

    q_lengths = _get_runtime_seq_lengths(case, "actual_seq_q", "S1")
    kv_lengths = _get_runtime_seq_lengths(case, "actual_seq_kv", "S2")
    cu_seqlens_q = _prefix(q_lengths)
    cu_seqlens_kv = _prefix(kv_lengths)
    total_q = int(cu_seqlens_q[-1].item())

    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    head_dim = case["D"]
    layout_q = case["layout_q"]

    # Q: fp32 随机 → MXFP8 配套量化 (per-32-group along D)
    if layout_q == "NTD":
        q_fp32 = _rand_float((n1, total_q, head_dim), generator, DATA_RANGE_Q)
        query, q_descale = quantize_mxfp8_qk(
            q_fp32.unsqueeze(0), FP8_DTYPE, QUANT_GROUP_SIZE
        )
        query = query.squeeze(0)
        q_descale = q_descale.squeeze(0)
    else:
        q_fp32 = _rand_float((total_q, n1, head_dim), generator, DATA_RANGE_Q)
        query, q_descale = quantize_mxfp8_qk(
            q_fp32.unsqueeze(0), FP8_DTYPE, QUANT_GROUP_SIZE
        )
        query = query.squeeze(0)
        q_descale = q_descale.squeeze(0)

    # K/V: BSND fp32 → MXFP8 配套量化 → permute 到 BNSD
    max_kv_len = max(kv_lengths)
    k_fp32 = _rand_float((batch, max_kv_len, n2, head_dim), generator, DATA_RANGE_K)
    v_fp32 = _rand_float((batch, max_kv_len, n2, head_dim), generator, DATA_RANGE_V)
    _log_tensor_ranges(
        "original FP32 range",
        q=q_fp32,
        k=k_fp32,
        v=v_fp32,
    )
    dense_key, dense_k_descale = quantize_mxfp8_qk(k_fp32, FP8_DTYPE, QUANT_GROUP_SIZE)
    dense_value, dense_v_descale = quantize_mxfp8_v(v_fp32, FP8_DTYPE, QUANT_GROUP_SIZE)

    block_table = _make_reference_block_table(
        batch, kv_lengths, BLOCK_SIZE, case["block_table_pattern"], rng
    )

    sparse_indices, sparse_seq_len = _make_reference_sparse_indices(
        case, q_lengths, kv_lengths, rng
    )
    p_scale_value = case.get("p_scale_value")
    p_scale = (
        None
        if p_scale_value is None
        else torch.tensor([float(p_scale_value)], dtype=torch.float32)
    )

    return {
        "case": case,
        "query": query,
        "key": dense_key.permute(0, 2, 1, 3).contiguous(),
        "value": dense_value.permute(0, 2, 1, 3).contiguous(),
        "q_descale": q_descale,
        "k_descale": dense_k_descale.permute(0, 2, 1, 3).contiguous(),
        "v_descale": dense_v_descale.permute(0, 2, 1, 3).contiguous(),
        "p_scale": p_scale,
        "block_table": block_table,
        "sparse_indices": sparse_indices,
        "sparse_seq_len": sparse_seq_len,
        "q_lengths": q_lengths,
        "kv_lengths": kv_lengths,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
    }


def generate_data():
    """Generate inputs through the reference-style data-generation wrapper."""
    data = _generate_reference_style_mxfp8_inputs()
    logger.info(
        "[INFO] reference-style data: q=%s, k=%s, v=%s, q_descale=%s, k_descale=%s, v_descale=%s",
        data["query"].shape,
        data["key"].shape,
        data["value"].shape,
        data["q_descale"].shape,
        data["k_descale"].shape,
        data["v_descale"].shape,
    )
    logger.info(
        "[INFO] sparse_indices=%s, sparse_seq_len=%s",
        data["sparse_indices"].shape,
        data["sparse_seq_len"].shape,
    )

    return (
        data["query"],
        data["key"],
        data["value"],
        data["q_descale"],
        data["k_descale"],
        data["v_descale"],
        data["p_scale"],
        data["block_table"],
        data["sparse_indices"],
        data["sparse_seq_len"],
        data["q_lengths"],
        data["kv_lengths"],
        data["cu_seqlens_q"],
        data["cu_seqlens_kv"],
    )


# ==============================================================================
# CPU Golden
# 参考 quant_block_sparse_attn_golden.py 的 sparse block 计算流：
#   - Q/QDescale 公共输入为 TND/NTD；K/V 与 K/V descale 使用内部 BNSD
#   - 按 sparse_indices 中记录的 KV block 顺序收集 positions，并按 256-token C1 粒度做 online 累加
#   - Q/K 使用 per-token D-group descale；V 使用 per-channel S-group descale
#   - 输出 OUT 固定 TND，MXFP8 TND LSE 固定 TN
# ==============================================================================

EMPTY_LSE = -3.4028234663852886e38
MASK_VALUE = -10000.0
LN2 = math.log(2.0)


def _align_up_to_ln2(value):
    return torch.ceil(value / LN2) * LN2


def _positions_from_sparse(
    sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len
):
    block_count = int(sparse_seq_len[batch_idx, head_idx, qb_idx].item())
    c1_s2_size = int(CASE["s2_base_size"]) // 2
    if c1_s2_size % SPARSE_BLOCK_SIZE != 0:
        raise ValueError(
            f"C1 S2 size must be divisible by sparse block size, got {c1_s2_size} and {SPARSE_BLOCK_SIZE}"
        )
    blocks_per_c1 = c1_s2_size // SPARSE_BLOCK_SIZE
    blocks_per_task = int(CASE["s2_base_size"]) // SPARSE_BLOCK_SIZE

    raw_blocks = []
    for i in range(block_count):
        idx = int(sparse_indices[batch_idx, head_idx, qb_idx, i].item())
        raw_blocks.append(idx)

    all_task_positions = []
    task_start = 0
    while task_start < block_count:
        task_end = min(task_start + blocks_per_task, block_count)
        task_raw = raw_blocks[task_start:task_end]
        task_valid = sorted(b for b in task_raw if b >= 0)

        chunk_positions = []
        cursor = 0
        while cursor < len(task_valid):
            c1_positions = []
            for offset in range(blocks_per_c1):
                if cursor + offset >= len(task_valid):
                    continue
                block_idx = task_valid[cursor + offset]
                start = block_idx * SPARSE_BLOCK_SIZE
                end = min(start + SPARSE_BLOCK_SIZE, kv_len)
                if start < kv_len:
                    c1_positions.extend(range(start, end))
            if c1_positions:
                chunk_positions.append(c1_positions)
            cursor += blocks_per_c1
        if chunk_positions:
            all_task_positions.append(chunk_positions)
        task_start += blocks_per_task
    return all_task_positions


def _expand_d_group_scale(scale, width):
    return scale.to(torch.float32).repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[
        ..., :width
    ]


def _gather_q_block_and_scale(
    q_tensor, q_scale, layout_q, cu_seqlens_q, batch_idx, q_start, q_end, head_idx
):
    if layout_q == "NTD":
        base = int(cu_seqlens_q[batch_idx].item())
        q_block = q_tensor[head_idx, base + q_start : base + q_end]
        q_scale_block = q_scale[head_idx, base + q_start : base + q_end]
    else:
        base = int(cu_seqlens_q[batch_idx].item())
        q_block = q_tensor[base + q_start : base + q_end, head_idx]
        q_scale_block = q_scale[base + q_start : base + q_end, head_idx]
    return q_block.to(torch.float32), q_scale_block.to(torch.float32)


def _valid_mask_for_positions(q_indices, positions, q_len, kv_len):
    nq = len(q_indices)
    npos = len(positions)
    if MASK_MODE == 0:
        return torch.ones((nq, npos), dtype=torch.bool)
    if MASK_MODE != 3:
        raise ValueError(f"Unsupported mask_mode: {MASK_MODE}")
    q_idx_col = torch.as_tensor(q_indices, dtype=torch.long).view(nq, 1)
    pos_tensor = torch.as_tensor(positions, dtype=torch.long).view(1, npos)
    return pos_tensor <= (q_idx_col + kv_len - q_len)


def cpu_mxfp8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    q_lengths,
    kv_lengths,
    cu_seqlens_q,
    sparse_indices,
    sparse_seq_len,
):
    """Reference-style CPU golden for MXFP8 block sparse attention."""
    layout_q = canonical_q_input_layout(Q_INPUT_LAYOUT)
    q_tensor = q_fp8
    q_scale = dequant_scale_q
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)
    k_scale = dequant_scale_k
    v_scale = dequant_scale_v

    total_q = int(cu_seqlens_q[-1].item())
    batch = len(q_lengths)
    n1 = N_q
    n2 = N_kv
    group = n1 // n2
    head_dim = D
    softmax_scale = float(CASE["softmax_scale"])
    p_scale_value = (
        1.0
        if p_scale is None
        else float(torch.as_tensor(p_scale).reshape(-1)[0].item())
    )
    ln_p_scale = 0.0 if p_scale_value == 1.0 else math.log(p_scale_value)

    attention_out = torch.zeros((total_q, n1, head_dim), dtype=torch.float32)
    softmax_lse = torch.full((total_q, n1), EMPTY_LSE, dtype=torch.float32)

    qb_max = math.ceil(max(q_lengths) / SPARSE_BLOCK_SIZE)
    neg_inf = float("-inf")

    logger.info(
        "[CPU Golden] reference sparse flow: layout_q=%s, OUT=TND, LSE=TN", layout_q
    )

    for batch_idx in range(batch):
        q_len = int(q_lengths[batch_idx])
        kv_len = int(kv_lengths[batch_idx])
        q_base = int(cu_seqlens_q[batch_idx].item())
        for head_idx in range(n1):
            n2_idx = head_idx // group
            for qb_idx in range(qb_max):
                q_start = qb_idx * SPARSE_BLOCK_SIZE
                if q_start >= q_len:
                    break
                q_end = min(q_start + SPARSE_BLOCK_SIZE, q_len)
                q_indices = list(range(q_start, q_end))
                nq = q_end - q_start

                # chunk_positions 对齐 kernel 侧一次 C1V1 的 K 读取粒度：
                #   - sparse_indices 记录的是当前 QB 可访问的 KV basic block id。
                #   - _positions_from_sparse 按 256-token C1 粒度取 KV basic block，
                #     并展开成 token 级 KV index。
                #   - blockSize=128 时每个 C1 取 2 块，blockSize=64 时每个 C1 取 4 块。
                #   - 无效 block id(-1) 或超出 kv_len 的 block 会被跳过，
                #     所以尾 chunk 可能不足 256 token。
                all_task_chunks = _positions_from_sparse(
                    sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len
                )
                if not all_task_chunks:
                    continue

                q_block, q_scale_block = _gather_q_block_and_scale(
                    q_tensor,
                    q_scale,
                    layout_q,
                    cu_seqlens_q,
                    batch_idx,
                    q_start,
                    q_end,
                    head_idx,
                )
                q_dequant = q_block * _expand_d_group_scale(q_scale_block, head_dim)

                m_run = torch.full((nq,), neg_inf, dtype=torch.float32)
                l_run = torch.zeros((nq,), dtype=torch.float32)
                acc = torch.zeros((nq, head_dim), dtype=torch.float32)
                any_valid = torch.zeros((nq,), dtype=torch.bool)

                for task_chunks in all_task_chunks:
                    positions = [pos for chunk in task_chunks for pos in chunk]
                    pos_tensor = torch.as_tensor(positions, dtype=torch.long)

                    k_mat = k_tensor[batch_idx, n2_idx, pos_tensor, :]
                    k_scale_mat = k_scale[batch_idx, n2_idx, pos_tensor, :]
                    k_dequant = k_mat * _expand_d_group_scale(k_scale_mat, head_dim)

                    valid_mask = _valid_mask_for_positions(
                        q_indices, positions, q_len, kv_len
                    )
                    any_valid |= valid_mask.any(dim=-1)
                    scores = (
                        torch.matmul(q_dequant, k_dequant.transpose(0, 1))
                        * softmax_scale
                    )
                    scores = torch.where(
                        valid_mask, scores, torch.full_like(scores, MASK_VALUE)
                    )

                    chunk_offsets = []
                    offset = 0
                    for chunk_pos_list in task_chunks:
                        next_offset = offset + len(chunk_pos_list)
                        chunk_offsets.append((offset, next_offset))
                        offset = next_offset

                    for round_idx in range(0, len(chunk_offsets), 2):
                        subloop_results = []
                        m_before_round = m_run
                        m_subloop = m_run  # 已减去 ln_p_scale 的 max
                        round_has = torch.zeros((nq,), dtype=torch.bool)
                        round_end_idx = min(round_idx + 2, len(chunk_offsets))
                        for subloop_idx in range(round_idx, round_end_idx):
                            subloop_start, subloop_end = chunk_offsets[subloop_idx]
                            s_subloop = scores[:, subloop_start:subloop_end]
                            vm_subloop = valid_mask[:, subloop_start:subloop_end]
                            subloop_has = vm_subloop.any(dim=-1)
                            round_has |= subloop_has

                            masked_scores = torch.where(
                                vm_subloop,
                                s_subloop,
                                torch.full_like(s_subloop, neg_inf),
                            )
                            # NPU 顺序: local_max → 减 ln_p_scale → 对齐 → 和上一轮合并
                            local_max = masked_scores.max(dim=-1).values
                            local_max = _align_up_to_ln2(local_max - ln_p_scale)
                            subloop_started = m_subloop != neg_inf
                            m_candidate = torch.where(
                                subloop_started,
                                torch.maximum(m_subloop, local_max),
                                local_max,
                            )
                            m_subloop = torch.where(subloop_has, m_candidate, m_subloop)

                            # NPU: FusedExpSub(x, x, max) = exp(x - max)
                            # max 已含 -ln_p_scale，所以等价于 exp(x - max + ln_p_scale)
                            safe_m_subloop = torch.where(
                                torch.isfinite(m_subloop),
                                m_subloop,
                                torch.zeros_like(m_subloop),
                            )
                            p_subloop = torch.exp(
                                s_subloop - safe_m_subloop.view(nq, 1)
                            )
                            p_subloop = torch.where(
                                vm_subloop & subloop_has.view(nq, 1),
                                p_subloop,
                                torch.zeros_like(p_subloop),
                            )
                            p_quant_subloop = p_subloop.to(FP8_DTYPE).to(torch.float32)
                            subloop_results.append(
                                (
                                    subloop_start,
                                    subloop_end,
                                    m_subloop.clone(),
                                    p_subloop,
                                    p_quant_subloop,
                                )
                            )

                        m_new = m_subloop
                        run_started = m_before_round != neg_inf
                        history_rescale = torch.where(
                            run_started & torch.isfinite(m_new),
                            torch.exp(m_before_round - m_new),
                            torch.zeros_like(m_new),
                        )
                        history_rescale = torch.where(
                            torch.isfinite(history_rescale),
                            history_rescale,
                            torch.zeros_like(history_rescale),
                        )

                        pv = torch.zeros_like(acc)
                        round_sum = torch.zeros_like(l_run)
                        for (
                            subloop_start,
                            subloop_end,
                            subloop_max,
                            p_subloop,
                            p_quant_subloop,
                        ) in subloop_results:
                            subloop_rescale = torch.where(
                                torch.isfinite(subloop_max) & torch.isfinite(m_new),
                                torch.exp(subloop_max - m_new),
                                torch.zeros_like(m_new),
                            )
                            subloop_rescale = torch.where(
                                torch.isfinite(subloop_rescale),
                                subloop_rescale,
                                torch.zeros_like(subloop_rescale),
                            )

                            subloop_pos_tensor = pos_tensor[subloop_start:subloop_end]
                            v_mat = v_tensor[batch_idx, n2_idx, subloop_pos_tensor, :]
                            v_group_idx = torch.div(
                                subloop_pos_tensor,
                                QUANT_GROUP_SIZE,
                                rounding_mode="floor",
                            )
                            v_scale_mat = v_scale[batch_idx, n2_idx, v_group_idx, :].to(
                                torch.float32
                            )
                            v_dequant = v_mat * v_scale_mat

                            pv += torch.matmul(
                                p_quant_subloop * subloop_rescale.view(nq, 1), v_dequant
                            )
                            # C2 consumes the E4M3-quantized P.  Normalize with
                            # that same P so quantization does not become a
                            # query-row-wide output scale error.
                            round_sum += p_quant_subloop.sum(dim=-1) * subloop_rescale

                        acc = acc * history_rescale.view(nq, 1) + pv
                        l_run = l_run * history_rescale + round_sum
                        m_run = torch.where(round_has, m_new, m_run)

                safe_l = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                attn = acc / safe_l.view(nq, 1)
                # m_run 已减去 ln_p_scale（对齐 NPU），LSE 需要加回 ln_p_scale 恢复原始语义。
                lse = torch.log(safe_l) + m_run + ln_p_scale

                for local_idx in range(nq):
                    if not bool(any_valid[local_idx].item()):
                        continue
                    out_idx = q_base + q_start + local_idx
                    attention_out[out_idx, head_idx] = attn[local_idx].to(
                        torch.bfloat16
                    )
                    softmax_lse[out_idx, head_idx] = lse[local_idx]

    logger.info(
        "[CPU Golden] output(TND)=%s, lse(TN)=%s",
        attention_out.shape,
        softmax_lse.shape,
    )
    return attention_out.contiguous(), softmax_lse.contiguous()


# ==============================================================================
# NPU 调用
# 只支持单算子模式。
# Q: 使用 Q_INPUT_LAYOUT；QDescale: 固定按 TND 打包 D-group pair。
# K/V 与 K/V descale: 固定走 PA 预处理 (block_table + block_size)。
# ==============================================================================


def _to_npu(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.npu() if tensor.device.type == "cpu" else tensor
    return torch.as_tensor(tensor).npu()


def _prepare_npu_metadata(
    q_lengths,
    kv_lengths,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table,
    q_n,
    kv_n,
    layout_q,
    layout_kv,
    sparse_indices,
    sparse_seq_len,
):
    """准备 metadata 算子输入，并生成主算子依赖的 metadata。"""
    if CASE.get("empty_actual_seq", False):
        seqused_q = torch.empty((0,), dtype=torch.int32)
        seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)
    else:
        seqused_q = torch.tensor(q_lengths, dtype=torch.int32)
        seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)

    cu_seqlens_q = _to_npu(cu_seqlens_q)
    cu_seqlens_kv = _to_npu(cu_seqlens_kv)
    seqused_q = seqused_q.npu()
    seqused_kv = seqused_kv.npu()
    sparse_indices = _to_npu(sparse_indices)
    sparse_seq_len = _to_npu(sparse_seq_len)
    block_table = _to_npu(block_table)

    # metadata 算子在主算子及图捕获之外执行。前一次同步保证异步 H2D 输入就绪，
    # 后一次同步保证 metadata 完成物化；二者对应不同的数据依赖，不能合并。
    torch_npu.npu.synchronize()
    metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        sparse_seq_len,
        q_n,
        kv_n,
        D,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=B,
        sparse_block_size_q=SPARSE_BLOCK_SIZE,
        sparse_block_size_k=SPARSE_BLOCK_SIZE,
        quant_mode=QUANT_MODE_MXFP8,
        mask_mode=MASK_MODE,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_indices=SPARSE_INDICES_LAYOUT,
    )
    torch_npu.npu.synchronize()
    return (
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sparse_indices,
        sparse_seq_len,
        block_table,
        metadata,
    )


def _call_npu_fa_op(
    q,
    k,
    v,
    mask,
    q_lengths,
    kv_lengths,
    cu_seqlens_q,
    cu_seqlens_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_kv,
    sparse_indices,
    sparse_seq_len,
):
    """调用 QuantBlockSparseAttn 单算子，入参对齐 custom op schema。"""
    _ensure_custom_ops_registered()
    torch_npu.npu.set_device(int(DEVICE_ID))

    (
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sparse_indices,
        sparse_seq_len,
        block_table,
        metadata,
    ) = _prepare_npu_metadata(
        q_lengths,
        kv_lengths,
        cu_seqlens_q,
        cu_seqlens_kv,
        block_table,
        q_n,
        kv_n,
        layout_q,
        layout_kv,
        sparse_indices,
        sparse_seq_len,
    )

    output = torch.ops.custom.npu_quant_block_sparse_attn(
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        mask,
        softmax_scale,
        SPARSE_BLOCK_SIZE,
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        metadata=metadata,
        layout_kv=layout_kv,
        layout_q=layout_q,
        layout_sparse_indices=SPARSE_INDICES_LAYOUT,
        layout_out=OUT_LAYOUT,
        quant_mode=QUANT_MODE_MXFP8,
        mask_mode=MASK_MODE,
        return_softmax_lse=ENABLE_LSE,
    )
    torch_npu.npu.synchronize()
    atten_out, lse_out = output
    return atten_out.detach().cpu(), lse_out.detach().cpu()


class _QuantBlockSparseAttnGraph(torch.nn.Module):
    def __init__(self, softmax_scale, layout_q, layout_kv):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.sparse_block_size = SPARSE_BLOCK_SIZE
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_sparse_indices = SPARSE_INDICES_LAYOUT
        self.layout_out = OUT_LAYOUT
        self.quant_mode = QUANT_MODE_MXFP8
        self.mask_mode = MASK_MODE
        self.return_softmax_lse = ENABLE_LSE

    def forward(
        self,
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        metadata,
    ):
        return torch.ops.custom.npu_quant_block_sparse_attn(
            q,
            k,
            v,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            mask,
            self.softmax_scale,
            self.sparse_block_size,
            self.sparse_block_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            block_table=block_table,
            metadata=metadata,
            layout_kv=self.layout_kv,
            layout_q=self.layout_q,
            layout_sparse_indices=self.layout_sparse_indices,
            layout_out=self.layout_out,
            quant_mode=self.quant_mode,
            mask_mode=self.mask_mode,
            return_softmax_lse=self.return_softmax_lse,
        )


def _mark_static_graph_tensors(*tensors):
    """Freeze QBSA input shapes for ACL graph capture/replay."""
    for tensor in tensors:
        if tensor is not None:
            torch._dynamo.mark_static(tensor)


def _call_npu_fa_op_graph(
    q,
    k,
    v,
    mask,
    q_lengths,
    kv_lengths,
    cu_seqlens_q,
    cu_seqlens_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_kv,
    sparse_indices,
    sparse_seq_len,
):
    """调用 QuantBlockSparseAttn 图模式，入参对齐 custom op schema。"""
    _ensure_custom_ops_registered()
    torch_npu.npu.set_device(int(DEVICE_ID))

    (
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sparse_indices,
        sparse_seq_len,
        block_table,
        metadata,
    ) = _prepare_npu_metadata(
        q_lengths,
        kv_lengths,
        cu_seqlens_q,
        cu_seqlens_kv,
        block_table,
        q_n,
        kv_n,
        layout_q,
        layout_kv,
        sparse_indices,
        sparse_seq_len,
    )

    # 与批量 ACL 图路径保持一致。QBSA 是 AIC/AIV 混合核，图捕获需固定输入 shape，
    # 同时禁止对 custom op 仍在使用的 Buffer 执行 reinplace。
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    _mark_static_graph_tensors(
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        metadata,
    )
    torch._dynamo.reset()
    model = torch.compile(
        _QuantBlockSparseAttnGraph(softmax_scale, layout_q, layout_kv).npu(),
        fullgraph=True,
        backend=npu_backend,
        dynamic=False,
    )
    output = model(
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        metadata,
    )
    torch_npu.npu.synchronize()
    atten_out, lse_out = output
    return atten_out.detach().cpu(), lse_out.detach().cpu()


def _build_causal_mask():
    if MASK_MODE == 0:
        return torch.empty((0, 0), dtype=torch.uint8).npu()
    return torch.triu(torch.ones(2048, 2048, dtype=torch.uint8), diagonal=0).npu()


def _calc_cube_compute_amount(q_lengths, kv_lengths, sparse_indices, sparse_seq_len):
    q_block_size = int(CASE["sparse_q_block_size"])
    kv_block_size = int(CASE["sparse_kv_block_size"])
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError(
            f"sparse block size must be positive, got q={q_block_size}, kv={kv_block_size}"
        )

    sparse_indices_cpu = torch.as_tensor(sparse_indices).cpu()
    sparse_seq_len_cpu = torch.as_tensor(sparse_seq_len).cpu()
    if sparse_indices_cpu.dim() != 4 or sparse_seq_len_cpu.dim() != 3:
        raise ValueError(
            f"invalid sparse shapes: sparse_indices={tuple(sparse_indices_cpu.shape)}, "
            f"sparse_seq_len={tuple(sparse_seq_len_cpu.shape)}"
        )

    batch = len(q_lengths)
    if len(kv_lengths) != batch:
        raise ValueError(
            f"q_lengths and kv_lengths batch mismatch: {len(q_lengths)} vs {len(kv_lengths)}"
        )
    if sparse_seq_len_cpu.shape[0] < batch or sparse_indices_cpu.shape[0] < batch:
        raise ValueError(
            f"sparse batch dimension is smaller than seqused batch: sparse_indices={tuple(sparse_indices_cpu.shape)}, "
            f"sparse_seq_len={tuple(sparse_seq_len_cpu.shape)}, batch={batch}"
        )
    if sparse_seq_len_cpu.shape[1] < N_q or sparse_indices_cpu.shape[1] < N_q:
        raise ValueError(
            f"sparse head dimension is smaller than N_q={N_q}: sparse_indices={tuple(sparse_indices_cpu.shape)}, "
            f"sparse_seq_len={tuple(sparse_seq_len_cpu.shape)}"
        )

    basic_block_count = 0
    qb_limit = sparse_seq_len_cpu.shape[2]
    kb_limit = sparse_indices_cpu.shape[3]
    for batch_idx in range(batch):
        q_len = int(q_lengths[batch_idx])
        kv_len = int(kv_lengths[batch_idx])
        qb_count = min(qb_limit, math.ceil(q_len / q_block_size))
        for head_idx in range(N_q):
            for qb_idx in range(qb_count):
                q_start = qb_idx * q_block_size
                q_end = min(q_start + q_block_size, q_len)
                if q_start >= q_end:
                    continue
                block_count = int(
                    sparse_seq_len_cpu[batch_idx, head_idx, qb_idx].item()
                )
                block_count = max(0, min(block_count, kb_limit))
                for sparse_idx in range(block_count):
                    kv_block_idx = int(
                        sparse_indices_cpu[
                            batch_idx, head_idx, qb_idx, sparse_idx
                        ].item()
                    )
                    if kv_block_idx < 0:
                        continue
                    kv_start = kv_block_idx * kv_block_size
                    kv_end = min(kv_start + kv_block_size, kv_len)
                    if kv_start >= kv_end:
                        continue
                    basic_block_count += 1

    head_dim = int(D)
    single_basic_block_compute = q_block_size * kv_block_size * head_dim
    multiply_add_compute = 2
    bmm_compute_count = 2
    cube_compute_amount = (
        basic_block_count
        * single_basic_block_compute
        * multiply_add_compute
        * bmm_compute_count
    )
    return {
        "basic_block_count": basic_block_count,
        "basic_block_shape": (q_block_size, kv_block_size, head_dim),
        "single_basic_block_compute": single_basic_block_compute,
        "multiply_add_compute": multiply_add_compute,
        "bmm_compute_count": bmm_compute_count,
        "cube_compute_amount": cube_compute_amount,
    }


def _calc_cube_compute_capacity():
    unit_conversion = 1000
    fractal_m = 16
    fractal_n = 16
    fp8_fractal_k = 32
    min_fractal_compute = fractal_m * fractal_n * fp8_fractal_k
    frequency_ghz = 1.65
    aic_count = 32
    multiply_add_compute = 2
    cube_compute_capacity = (
        unit_conversion
        * min_fractal_compute
        * frequency_ghz
        * aic_count
        * multiply_add_compute
    )
    return {
        "unit_conversion": unit_conversion,
        "fractal_shape": (fractal_m, fractal_n, fp8_fractal_k),
        "min_fractal_compute": min_fractal_compute,
        "frequency_ghz": frequency_ghz,
        "aic_count": aic_count,
        "multiply_add_compute": multiply_add_compute,
        "cube_compute_capacity": cube_compute_capacity,
    }


def _log_attention_compute_stats(
    case_name, q_lengths, kv_lengths, sparse_indices, sparse_seq_len
):
    compute_info = _calc_cube_compute_amount(
        q_lengths, kv_lengths, sparse_indices, sparse_seq_len
    )
    capacity_info = _calc_cube_compute_capacity()
    basic_block_shape = compute_info["basic_block_shape"]
    fractal_shape = capacity_info["fractal_shape"]
    mfu_time = (
        compute_info["cube_compute_amount"] / capacity_info["cube_compute_capacity"]
    )
    logger.info("case_name=%s", case_name)
    logger.info("FLOPS计算过程量:")
    logger.info("基本块数量: %d", compute_info["basic_block_count"])
    logger.info(
        "基本块的shape: q_block_size=%d, kv_block_size=%d, head_dim=%d",
        basic_block_shape[0],
        basic_block_shape[1],
        basic_block_shape[2],
    )
    logger.info("单基本块计算量: %d", compute_info["single_basic_block_compute"])
    logger.info(
        "FLOPS计算公式: 基本块数(%d) * 单基本块计算量(%d) * 乘加计算(%d) * 两次bmm计算(%d) = %d",
        compute_info["basic_block_count"],
        compute_info["single_basic_block_compute"],
        compute_info["multiply_add_compute"],
        compute_info["bmm_compute_count"],
        compute_info["cube_compute_amount"],
    )
    logger.info("算力计算过程量:")
    logger.info(
        "一轮cycle对应最小分型shape: m=%d, n=%d, k=%d(fp8为32)",
        fractal_shape[0],
        fractal_shape[1],
        fractal_shape[2],
    )
    logger.info("一轮cycle对应最小分型计算量: %d", capacity_info["min_fractal_compute"])
    logger.info(
        "算力计算公式: 单位换算(%d) * (一轮cycle对应最小分型计算量(%d) * 频率GHz(%.2f) * "
        "AIC数量(%d) * 乘加计算(%d)) = %.6f",
        capacity_info["unit_conversion"],
        capacity_info["min_fractal_compute"],
        capacity_info["frequency_ghz"],
        capacity_info["aic_count"],
        capacity_info["multiply_add_compute"],
        capacity_info["cube_compute_capacity"],
    )
    logger.info(
        "MFU*时间计算公式: FLOPS(%d) / 算力(%.6f) = MFU * 时间(us) = %.6f",
        compute_info["cube_compute_amount"],
        capacity_info["cube_compute_capacity"],
        mfu_time,
    )
    return compute_info["cube_compute_amount"], mfu_time


def npu_mxfp8_fa(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    q_lengths,
    kv_lengths,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table_torch=None,
    sparse_indices=None,
    sparse_seq_len=None,
    graph=False,
):
    """调用 QuantBlockSparseAttn NPU 算子。"""
    torch_npu.npu.set_device(int(DEVICE_ID))

    if sparse_indices is None or sparse_seq_len is None:
        sparse_indices, sparse_seq_len = _make_reference_sparse_for_lengths(
            q_lengths, kv_lengths
        )

    softmax_scale = float(CASE["softmax_scale"])

    q_layout = canonical_q_input_layout(Q_INPUT_LAYOUT)
    q_input = (
        convert_q_tnd_or_ntd_to_layout(q_fp8, q_lengths, N_q, q_layout, "Q")
        .contiguous()
        .view(FP8_DTYPE)
    )
    q_npu = q_input.npu()
    logger.info("[NPU %s] q=%s", q_layout, q_npu.shape)

    q_scale_e8m0 = fp32_to_e8m0fnu_safe(
        convert_q_scale_tnd_or_ntd_to_layout(dequant_scale_q, q_lengths), "Q descale"
    )
    deq_q_npu = q_scale_e8m0.npu()
    logger.info("[NPU] Q descale layout=TND, shape=%s", q_scale_e8m0.shape)

    if p_scale is not None:
        p_scale_e8m0 = fp32_to_e8m0fnu_safe(p_scale, "P scale")
        p_scale_npu = p_scale_e8m0.npu()
        logger.info(
            "[NPU] P scale dtype=%s, shape=%s", p_scale_e8m0.dtype, p_scale_e8m0.shape
        )
    else:
        p_scale_npu = torch.tensor([], dtype=torch.float8_e8m0fnu).npu()
        logger.info("[NPU] P scale=empty tensor (shape size 0, default 1.0)")
    mask_arg = _build_causal_mask()

    if block_table_torch is None:
        raise ValueError("PA KV cache requires block_table_torch")

    k_pa = mxfp8_pa_preprocessing(
        k_fp8,
        kv_lengths,
        BLOCK_SIZE,
        block_table_torch,
        is_vscale=False,
        is_scale=False,
        kv_layout=KV_CACHE_LAYOUT,
        group_size=32,
        sparse_indices=sparse_indices,
        sparse_seq_len=sparse_seq_len,
    )
    v_pa = mxfp8_pa_preprocessing(
        v_fp8,
        kv_lengths,
        BLOCK_SIZE,
        block_table_torch,
        is_vscale=False,
        is_scale=False,
        kv_layout=KV_CACHE_LAYOUT,
        group_size=32,
        sparse_indices=sparse_indices,
        sparse_seq_len=sparse_seq_len,
    )
    k_input = k_pa.contiguous().view(FP8_DTYPE)
    v_input = v_pa.contiguous().view(FP8_DTYPE)
    k_npu = k_input.npu()
    v_npu = v_input.npu()
    if not IS_CONTIGUOUS:
        kv_cache = torch.stack([k_pa, v_pa], dim=2).npu()
        k_npu = kv_cache[:, :, 0]
        v_npu = kv_cache[:, :, 1]
        logger.info(
            "[NPU] key is_contiguous=%s, value is_contiguous=%s",
            k_npu.is_contiguous(),
            v_npu.is_contiguous(),
        )

    k_scale_pa = mxfp8_pa_preprocessing(
        dequant_scale_k,
        kv_lengths,
        BLOCK_SIZE,
        block_table_torch,
        is_vscale=False,
        is_scale=True,
        kv_layout=KV_CACHE_LAYOUT,
        group_size=32,
        sparse_indices=sparse_indices,
        sparse_seq_len=sparse_seq_len,
    )
    v_scale_pa = mxfp8_pa_preprocessing(
        dequant_scale_v,
        kv_lengths,
        BLOCK_SIZE,
        block_table_torch,
        is_vscale=True,
        is_scale=True,
        kv_layout=KV_CACHE_LAYOUT,
        group_size=32,
        sparse_indices=sparse_indices,
        sparse_seq_len=sparse_seq_len,
    )

    k_scale_e8m0 = fp32_to_e8m0fnu_safe(k_scale_pa, "K PA descale")
    v_scale_e8m0 = fp32_to_e8m0fnu_safe(v_scale_pa, "V PA descale")
    _log_tensor_ranges(
        "converted NPU input range",
        q=q_input,
        k=k_input,
        v=v_input,
        q_descale=q_scale_e8m0,
        k_descale=k_scale_e8m0,
        v_descale=v_scale_e8m0,
    )
    deq_k_npu = k_scale_e8m0.npu()
    deq_v_npu = v_scale_e8m0.npu()
    if not IS_CONTIGUOUS:
        fake_kscale_tensor = torch.ones_like(deq_k_npu)
        fake_vscale_tensor = torch.ones_like(deq_v_npu)
        deq_k_npu = torch.stack([deq_k_npu, fake_kscale_tensor], dim=2)[:, :, 0]
        deq_v_npu = torch.stack([deq_v_npu, fake_vscale_tensor], dim=2)[:, :, 0]
        logger.info(
            "[NPU] deq_k_descale is_contiguous=%s, deq_v_descale is_contiguous=%s",
            deq_k_npu.is_contiguous(),
            deq_v_npu.is_contiguous(),
        )

    logger.info("[NPU PA] kv_layout=%s", KV_CACHE_LAYOUT)
    logger.info("[NPU PA] k=%s, v=%s", k_npu.shape, v_npu.shape)
    logger.info("[NPU PA] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)

    block_table_npu = (
        block_table_torch.npu()
        if isinstance(block_table_torch, torch.Tensor)
        else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
    )
    layout_q = q_layout
    layout_kv = KEY_LAYOUT

    npu_call_args = (
        q_npu,
        k_npu,
        v_npu,
        mask_arg,
        q_lengths,
        kv_lengths,
        cu_seqlens_q,
        cu_seqlens_kv,
        deq_q_npu,
        deq_k_npu,
        deq_v_npu,
        p_scale_npu,
        block_table_npu,
        N_q,
        N_kv,
        softmax_scale,
        layout_q,
        layout_kv,
        sparse_indices,
        sparse_seq_len,
    )
    if graph:
        logger.info(
            "[NPU] 调用 QuantBlockSparseAttn 图模式, layout_q=%s, layout_kv=%s",
            layout_q,
            layout_kv,
        )
        atten_out, lse_out = _call_npu_fa_op_graph(*npu_call_args)
    else:
        logger.info(
            "[NPU] 调用 QuantBlockSparseAttn 单算子, layout_q=%s, layout_kv=%s",
            layout_q,
            layout_kv,
        )
        atten_out, lse_out = _call_npu_fa_op(*npu_call_args)

    npu_output = atten_out
    npu_lse = lse_out
    T_actual = sum(q_lengths)
    if npu_output.shape[0] > T_actual:
        npu_output = npu_output[:T_actual]
    if npu_lse is not None and npu_lse.dim() >= 1 and npu_lse.shape[0] > T_actual:
        npu_lse = npu_lse[:T_actual, ...]
    logger.info(
        "[NPU] output=%s, lse=%s",
        npu_output.shape,
        None if npu_lse is None else npu_lse.shape,
    )
    return npu_output, npu_lse


# ==============================================================================
# Main
# ==============================================================================


def _load_golden_cache():
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache
    return golden_cache


def _cache_case_name(case_id, case_name_arg, total_case_num):
    safe_case_id = case_id.replace("/", "_")
    if case_name_arg is None:
        return safe_case_id
    if total_case_num == 1:
        return case_name_arg
    return f"{case_name_arg}_{safe_case_id}"


def _safe_debug_case_name(case_name):
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in case_name
    )


def _prepare_debug_artifacts(case_name):
    case_dir = os.path.join(CURRENT_DIR, "debug", _safe_debug_case_name(case_name))
    pt_dir = os.path.join(case_dir, "pt")
    precision_dir = os.path.join(case_dir, "precision")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(precision_dir, exist_ok=True)
    for directory, suffix in ((pt_dir, ".pt"), (precision_dir, ".png")):
        for file_name in os.listdir(directory):
            if file_name.endswith(suffix):
                os.remove(os.path.join(directory, file_name))
    return {
        "case_dir": case_dir,
        "pt_dir": pt_dir,
        "precision_dir": precision_dir,
        "log_path": os.path.join(case_dir, "run.log"),
    }


def _add_debug_log_handler(log_path):
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    return file_handler


def _log_case_start(case_idx, total_case_num, case_id, mode, cache_case_name):
    logger.info("")
    logger.info("#" * _CASE_LOG_WIDTH)
    logger.info("# CASE START [%d/%d]", case_idx, total_case_num)
    logger.info("# case_name : %s", case_id)
    logger.info("# cache_name: %s", cache_case_name)
    logger.info("# mode      : %s", ",".join(sorted(mode)))
    logger.info("#" * _CASE_LOG_WIDTH)


def _make_case_result(
    case_id,
    status,
    elapsed,
    out_result=None,
    lse_result=None,
    error=None,
    mfu_time=None,
):
    return {
        "case": case_id,
        "status": status,
        "elapsed": elapsed,
        "out": out_result,
        "lse": lse_result,
        "error": error,
        "mfu_time": mfu_time,
    }


def _result_status(compare_result):
    return compare_result[0] if compare_result is not None else "Skip"


def _result_pct(compare_result):
    return "-" if compare_result is None else f"{compare_result[1]:.6f}%"


def _result_max_error(compare_result):
    return "-" if compare_result is None else f"{compare_result[2]:.6g}"


def _result_mfu_time(item):
    value = item.get("mfu_time")
    if value is None:
        return "-"
    return f"{value:.2f}"


def _log_final_summary(results):
    total = len(results)
    passed = sum(1 for item in results if item["status"] == "Pass")
    skipped = sum(
        1 for item in results if item["status"] in {"Generated", "CpuDone", "NpuDone"}
    )
    failed = total - passed - skipped

    case_col_width = max((len(str(item["case"])) for item in results), default=0)
    case_col_width = max(case_col_width, len("Case"))
    table_width = 92 + case_col_width

    logger.info("")
    logger.info("=" * table_width)
    logger.info("QBSA MXFP8 CASE SUMMARY")
    logger.info("=" * table_width)
    logger.info(
        "%-4s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-*s",
        "No.",
        "Status",
        "Time(s)",
        "mfu*time",
        "OutPct",
        "OutMaxErr",
        "LsePct",
        "LseMaxErr",
        case_col_width,
        "Case",
    )
    logger.info("-" * table_width)
    for idx, item in enumerate(results, 1):
        logger.info(
            "%-4d %-10s %-10.2f %-12s %-12s %-12s %-12s %-12s %-*s",
            idx,
            item["status"],
            item["elapsed"],
            _result_mfu_time(item),
            _result_pct(item["out"]),
            _result_max_error(item["out"]),
            _result_pct(item["lse"]),
            _result_max_error(item["lse"]),
            case_col_width,
            item["case"],
        )
        if item["error"]:
            logger.info("     error: %s", item["error"])
    logger.info("-" * table_width)
    logger.info(
        "Total: %d, Pass: %d, Failed/Error: %d, No-compare: %d",
        total,
        passed,
        failed,
        skipped,
    )
    logger.info("=" * table_width)


def run_one_case(
    case_id,
    case,
    mode,
    case_name,
    cache_dir,
    case_idx=1,
    total_case_num=1,
    precision_dir=None,
    rdv=False,
    rdv_cache_dir=None,
    debug=False,
    graph=False,
):
    start_time = time.time()
    golden_cache = _load_golden_cache()
    set_active_case(case)

    _log_case_start(case_idx, total_case_num, case_id, mode, case_name)
    logger.info("MXFP8 Flash Attention Golden")
    logger.info("输出: 逐元素表格 + 统计汇总 (PctRlt 通过率)")
    logger.info("场景: PA")
    logger.info("Q_INPUT_LAYOUT=%s, QDescale layout=TND", Q_INPUT_LAYOUT)
    logger.info("KV_CACHE_LAYOUT=%s", KV_CACHE_LAYOUT)
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d", B, N_q, N_kv, D)
    logger.info("ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
    logger.info(
        "block_size=%s, sparse_block_size=%s, sparse_count=%s",
        case.get("block_size"),
        case.get("sparse_q_block_size"),
        case.get("sparse_count"),
    )

    if "gen" in mode:
        logger.info("\n[Step 1] 数据生成")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table_torch,
            sparse_indices,
            sparse_seq_len,
            q_lengths,
            kv_lengths,
            cu_seqlens_q,
            cu_seqlens_kv,
        ) = generate_data()
        golden_cache.save_input(
            case_name,
            {
                "q_fp8": q_fp8,
                "k_fp8": k_fp8,
                "v_fp8": v_fp8,
                "dequant_scale_q": dequant_scale_q,
                "dequant_scale_k": dequant_scale_k,
                "dequant_scale_v": dequant_scale_v,
                "p_scale": p_scale,
                "block_table_torch": block_table_torch,
                "sparse_indices": sparse_indices,
                "sparse_seq_len": sparse_seq_len,
                "q_lengths": q_lengths,
                "kv_lengths": kv_lengths,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_kv": cu_seqlens_kv,
            },
            cache_dir=cache_dir,
        )
    else:
        logger.info("\n[Step 1] 加载已保存的输入数据")
        input_path = golden_cache._path(case_name, "input", cache_dir)
        data = torch.load(input_path, weights_only=False)
        q_fp8 = data["q_fp8"]
        k_fp8 = data["k_fp8"]
        v_fp8 = data["v_fp8"]
        dequant_scale_q = data["dequant_scale_q"]
        dequant_scale_k = data["dequant_scale_k"]
        dequant_scale_v = data["dequant_scale_v"]
        p_scale = data["p_scale"]
        block_table_torch = data.get("block_table_torch")
        sparse_indices = data.get("sparse_indices")
        sparse_seq_len = data.get("sparse_seq_len")
        q_lengths = data.get(
            "q_lengths", _get_runtime_seq_lengths(CASE, "actual_seq_q", "S1")
        )
        kv_lengths = data.get(
            "kv_lengths", _get_runtime_seq_lengths(CASE, "actual_seq_kv", "S2")
        )
        cu_seqlens_q = data.get("cu_seqlens_q")
        cu_seqlens_kv = data.get("cu_seqlens_kv")
        if cu_seqlens_q is None:
            cu_seqlens_q = _prefix(q_lengths)
        if cu_seqlens_kv is None:
            cu_seqlens_kv = _prefix(kv_lengths)
        if sparse_indices is None or sparse_seq_len is None:
            sparse_indices, sparse_seq_len = _make_reference_sparse_for_lengths(
                q_lengths, kv_lengths
            )

    _, mfu_time = _log_attention_compute_stats(
        case_id, q_lengths, kv_lengths, sparse_indices, sparse_seq_len
    )

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("\n[Done] 数据已保存，退出")
        return _make_case_result(
            case_id, "Generated", time.time() - start_time, mfu_time=mfu_time
        )

    if rdv and golden_cache.has_cpu_output(case_name, cache_dir=rdv_cache_dir):
        logger.info("\n[Step 2] 复用 CPU Golden")
        logger.info("[RDV] 已找到 case=%s 的 CPU golden，跳过 CPU 生成", case_name)
        cpu_out, cpu_lse = golden_cache.load_cpu_output(
            case_name, cache_dir=rdv_cache_dir
        )
    elif "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        if rdv:
            logger.info("[RDV] 未找到 case=%s 的 CPU golden，按原流程生成", case_name)
        cpu_out, cpu_lse = cpu_mxfp8_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            q_lengths,
            kv_lengths,
            cu_seqlens_q,
            sparse_indices,
            sparse_seq_len,
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cache_dir)
    elif "compare" in mode:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cache_dir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("\n[Done] CPU 输出已保存，退出")
        return _make_case_result(
            case_id, "CpuDone", time.time() - start_time, mfu_time=mfu_time
        )

    if "npu" in mode:
        logger.info("\n[Step 3] NPU 调用")
        atten_out, lse_out = npu_mxfp8_fa(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            q_lengths,
            kv_lengths,
            cu_seqlens_q,
            cu_seqlens_kv,
            block_table_torch,
            sparse_indices,
            sparse_seq_len,
            graph=graph,
        )
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cache_dir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(
            case_name, cache_dir=cache_dir
        )

    if "npu" in mode and "compare" not in mode:
        logger.info("\n[Done] NPU 输出已保存，退出")
        return _make_case_result(
            case_id, "NpuDone", time.time() - start_time, mfu_time=mfu_time
        )

    logger.info("\n[Step 4] Atten OUT 精度对比")
    out_result = result_compare_method.check_result(cpu_out, atten_out, debug=debug)
    if precision_dir is not None:
        result_compare_method.save_precision_map(
            cpu_out,
            atten_out,
            os.path.join(precision_dir, "attention_out.png"),
            "attention_out",
            out_result,
        )

    lse_result = None
    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        lse_result = result_compare_method.check_result(cpu_lse, lse_out, debug=debug)
        if precision_dir is not None:
            result_compare_method.save_precision_map(
                cpu_lse,
                lse_out,
                os.path.join(precision_dir, "softmax_lse.png"),
                "softmax_lse",
                lse_result,
            )

    status = "Pass"
    if _result_status(out_result) != "Pass" or (
        lse_result is not None and _result_status(lse_result) != "Pass"
    ):
        status = "Failed"
    return _make_case_result(
        case_id,
        status,
        time.time() - start_time,
        out_result,
        lse_result,
        mfu_time=mfu_time,
    )


if __name__ == "__main__":
    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="MXFP8 QuantBlockSparseAttn Golden")
    parser.add_argument(
        "--case_files",
        default=None,
        help="case 文件路径，支持逗号分隔；默认文件由是否指定 --csv 决定",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="按 CSV 格式加载 case；不指定 --case_files 时读取 qbsa_mxfp8_test_cases.csv",
    )
    parser.add_argument(
        "--mode",
        default="all",
        help="执行模式，支持逗号组合: all/gen/cpu/npu/compare. 例: --mode=npu,compare",
    )
    parser.add_argument(
        "--case_name",
        "--case-name",
        dest="case_name",
        default=None,
        help="case 名称；不传时默认执行 enable=True 的 case，支持逗号分隔多个名称",
    )
    parser.add_argument(
        "--case_id", default=None, help="兼容旧参数，等价于 --case_name"
    )
    parser.add_argument(
        "--list_cases", action="store_true", help="只列出已加载 case，不执行"
    )
    parser.add_argument(
        "--cache_case_name",
        "--cache-case-name",
        dest="cache_case_name",
        default=None,
        help="缓存文件名前缀；默认使用 case name",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="缓存目录路径（默认 golden_cache/）"
    )
    parser.add_argument(
        "--rdv",
        action="store_true",
        help="CPU golden 缓存存在时直接复用；缓存不存在时按 --mode 原流程生成",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="单 case 调试模式；必须配合 --case_name，输出到 debug/<case_name>/。"
        "含 gen 时数据写入 debug pt 目录；不含 gen 时从 golden_cache 加载(可用 --cache-dir 指定)",
    )
    parser.add_argument(
        "--aclgraph",
        dest="aclgraph",
        action="store_true",
        help="使用图模式 (torchair + torch.compile) 执行 NPU 算子调用",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="不保存 input/cpu_output/npu_output 到磁盘，节省空间和时间",
    )
    parser.add_argument(
        "--case_range",
        default=None,
        help="只跑 CSV 中指定行范围的用例，格式 start:end (1-based, 含两端)。"
        "例如 --case_range 100:200 跑第100到200行(含)的用例",
    )
    args = parser.parse_args()

    raw_parts = {m.strip() for m in args.mode.split(",") if m.strip()}
    invalid = raw_parts - _VALID_MODES
    if invalid:
        parser.error(f"Invalid mode: {invalid}. Valid: {_VALID_MODES}")
    mode = {"gen", "cpu", "npu", "compare"} if "all" in raw_parts else raw_parts

    if args.debug:
        if args.case_name is None or args.case_id is not None:
            parser.error(
                "--debug must be used with --case_name, and does not accept --case_id"
            )
        if args.list_cases:
            parser.error(
                "--debug executes one case and cannot be combined with --list_cases"
            )
        if args.cache_case_name is not None:
            parser.error(
                "--debug manages case naming internally; do not use --cache_case_name"
            )
        if "gen" in mode and args.cache_dir is not None:
            parser.error(
                "--debug with gen mode manages its own pt directory; do not use --cache-dir"
            )

    case_files = args.case_files or (
        _DEFAULT_CASE_CSV if args.csv else _DEFAULT_CASE_FILE
    )
    all_cases = load_test_cases(case_files, use_csv=args.csv)

    if args.list_cases:
        for case_id in all_cases:
            print(f"{case_id} enable={all_cases[case_id].get('enable', True)}")
        sys.exit(0)

    if args.case_name and args.case_id:
        parser.error("--case_name and --case_id cannot be set at the same time")
    selected_case_name = args.case_name if args.case_name is not None else args.case_id
    run_case_ids = resolve_case_names(selected_case_name, all_cases)
    if not run_case_ids:
        parser.error(
            "No runnable case selected. Check enable fields or pass --case_name explicitly."
        )

    if args.case_range:
        try:
            parts = args.case_range.split(":")
            range_start = int(parts[0])
            range_end = int(parts[1]) if len(parts) > 1 else range_start
            range_start = max(1, range_start)
            range_end = min(len(run_case_ids), range_end)
            run_case_ids = run_case_ids[range_start - 1 : range_end]
            logger.info(
                "[RANGE] Running cases %d-%d (%d cases)",
                range_start,
                range_end,
                len(run_case_ids),
            )
        except (ValueError, IndexError):
            parser.error(
                f"Invalid --case_range format: {args.case_range}. Expected start:end (e.g. 100:200)"
            )
        if not run_case_ids:
            parser.error("--case_range resulted in empty selection")
    if args.debug and len(run_case_ids) != 1:
        parser.error(
            "--debug can run exactly one case; pass one name through --case_name"
        )

    debug_artifacts = _prepare_debug_artifacts(run_case_ids[0]) if args.debug else None
    debug_log_handler = (
        _add_debug_log_handler(debug_artifacts["log_path"]) if debug_artifacts else None
    )
    if debug_artifacts:
        logger.info("[DEBUG] case directory: %s", debug_artifacts["case_dir"])
        logger.info("[DEBUG] log file: %s", debug_artifacts["log_path"])
        logger.info("[DEBUG] PT directory: %s", debug_artifacts["pt_dir"])
        logger.info("[DEBUG] precision directory: %s", debug_artifacts["precision_dir"])

    case_results = []
    total_case_num = len(run_case_ids)
    try:
        for case_idx, case_id in enumerate(run_case_ids, 1):
            case_name = _cache_case_name(
                case_id, args.cache_case_name, len(run_case_ids)
            )
            if debug_artifacts:
                cache_dir = (
                    debug_artifacts["pt_dir"] if "gen" in mode else args.cache_dir
                )
            else:
                cache_dir = args.cache_dir
            precision_dir = (
                debug_artifacts["precision_dir"] if debug_artifacts else None
            )
            case_start = time.time()
            try:
                case_results.append(
                    run_one_case(
                        case_id,
                        all_cases[case_id],
                        mode,
                        case_name,
                        cache_dir,
                        case_idx,
                        total_case_num,
                        precision_dir,
                        rdv=args.rdv,
                        rdv_cache_dir=args.cache_dir,
                        debug=args.debug,
                        graph=args.aclgraph,
                    )
                )
            except Exception as err:  # pylint: disable=broad-except
                logger.exception(
                    "CASE ERROR [%d/%d] %s", case_idx, total_case_num, case_id
                )
                case_results.append(
                    _make_case_result(
                        case_id, "Error", time.time() - case_start, error=str(err)
                    )
                )

        _log_final_summary(case_results)
    finally:
        if debug_log_handler is not None:
            logging.getLogger().removeHandler(debug_log_handler)
            debug_log_handler.close()

    if any(item["status"] in {"Failed", "Error"} for item in case_results):
        sys.exit(1)
