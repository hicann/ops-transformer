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
支持：MXFP8 Q/QDescale 公共输入仅为 TND；NPU 侧 K/V 使用 PA KV cache
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
import warnings

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

try:
    from . import result_compare_method
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# TTK installs multiple root handlers that target the same captured stream.
# A propagated record is consequently printed once by every handler.  The
# asset loader imports this module under the private name below; bind that
# instance to one primary handler while leaving standalone pytest/debug
# logging untouched.
if __name__ == "_qbsa_mxfp8_pytest_golden" and logging.getLogger().handlers:
    root_handlers = logging.getLogger().handlers
    primary_handler = next(
        (
            handler
            for handler in root_handlers
            if not isinstance(handler, logging.FileHandler)
        ),
        root_handlers[0],
    )
    logger.handlers.clear()
    logger.addHandler(primary_handler)
    logger.propagate = False

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
}
_CASE_INT_FIELDS = {
    "B",
    "N1",
    "N2",
    "D",
    "block_size",
    "mask_mode",
    "quant_group_size",
    "seed",
    "device_id",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "quant_mode",
    "blocknum",
    "s1_base_size",
    "s2_base_size",
    "max_block_per_batch",
}
_CASE_FLOAT_FIELDS = {
    "p_scale_value",
    "softmax_scale",
}
_CASE_RANGE_FIELDS = {"data_range_q", "data_range_k", "data_range_v"}
_CASE_INT_LIST_FIELDS = {
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
}
_CASE_REQUIRED_RUNTIME_FIELDS = {
    "B",
    "N1",
    "N2",
    "D",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
    "sparse_mode",
    "max_block_per_batch",
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
    "return_softmax_lse",
    "seed",
    "data_range_q",
    "data_range_k",
    "data_range_v",
    "device_id",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "quant_mode",
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
        parsed = value
        if isinstance(value, str) and value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                if not value.endswith("]"):
                    raise
                # JSON 不接受 inf、nan 等 IEEE 拼写。这里是 MXFP8 测试数据
                # 的取值范围，因此用 float() 解析这些值，普通范围仍按 JSON 解析。
                parsed = [item.strip() for item in value[1:-1].split(",")]
        if isinstance(parsed, (list, tuple)):
            if len(parsed) != 2:
                raise ValueError(
                    f"{field_name} must contain exactly [min, max], got: {value}"
                )
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


def _resolve_sparse_mode(case):
    """Validate the shared dense/random sparse mode."""
    resolved = dict(case)
    sparse_mode = resolved.get("sparse_mode")
    if sparse_mode not in ("dense", "random"):
        raise ValueError(
            f"sparse_mode must be 'dense' or 'random', got {sparse_mode!r}"
        )
    resolved["sparse_mode"] = sparse_mode
    return resolved


def validate_mxfp8_case(case):
    """只校验实际参与运行的字段；参考字段允许由外部泛化表自由扩展。"""
    if str(case.get("layout_q", "")).upper() != "TND":
        raise ValueError(
            f"MXFP8 layout_q only supports TND, got {case.get('layout_q')!r}"
        )

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
    if int(case["max_block_per_batch"]) < 0:
        raise ValueError("max_block_per_batch must be non-negative")

    _get_sequence_inputs(case)

    p_scale_value = case.get("p_scale_value")
    if p_scale_value is not None:
        p_scale_value = float(p_scale_value)
        if not math.isfinite(p_scale_value) or p_scale_value <= 0:
            raise ValueError("p_scale_value must be positive and finite")


def _case_list(case, name):
    return [int(item) for item in case[name]]


def _get_sequence_inputs(case):
    """Return the four operator sequence inputs and their effective lengths.

    New MXFP8 cases use cu_seqlens_q + seqused_kv as the two effective-length
    sources.  cu_seqlens_kv and seqused_q are present but must be empty for the
    current operator contract.
    """
    batch = int(case["B"])
    cu_q = _case_list(case, "cu_seqlens_q")
    cu_kv = _case_list(case, "cu_seqlens_kv")
    seq_q = _case_list(case, "seqused_q")
    seq_kv = _case_list(case, "seqused_kv")

    if len(cu_q) != batch + 1 or not cu_q or cu_q[0] != 0:
        raise ValueError(
            f"cu_seqlens_q must start with 0 and contain B+1={batch + 1} values, got {cu_q}"
        )

    q_lengths = [end - start for start, end in zip(cu_q, cu_q[1:])]
    if any(length < 0 for length in q_lengths):
        raise ValueError(f"cu_seqlens_q must be nondecreasing, got {cu_q}")
    if len(seq_kv) != batch or any(length < 0 for length in seq_kv):
        raise ValueError(
            f"seqused_kv must contain B={batch} non-negative values, got {seq_kv}"
        )
    if cu_kv:
        raise ValueError("cu_seqlens_kv must be empty for MXFP8")
    if seq_q:
        raise ValueError("seqused_q must be empty for MXFP8")

    return {
        "cu_seqlens_q": torch.tensor(cu_q, dtype=torch.int32),
        "cu_seqlens_kv": torch.empty((0,), dtype=torch.int32),
        "seqused_q": torch.empty((0,), dtype=torch.int32),
        "seqused_kv": torch.tensor(seq_kv, dtype=torch.int32),
        "q_lengths": q_lengths,
        "kv_lengths": seq_kv,
    }


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
    if case.get("is_contiguous") is None:
        # TTK has no continuity switch.  New CSVs therefore omit this field
        # and use the normal contiguous path.  Keep the optional field only
        # for backward-compatible/manual coverage of the non-contiguous path.
        case["is_contiguous"] = True
    case["softmax_scale"] = case.get("softmax_scale") or (
        1.0 / math.sqrt(float(case["D"]))
    )
    sequence_inputs = _get_sequence_inputs(case)
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        case[name] = sequence_inputs[name].tolist()
    case = _resolve_sparse_mode(case)
    validate_mxfp8_case(case)
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
    global CASE, B, N_q, N_kv, D, Q_LENGTHS, KV_LENGTHS, BLOCK_SIZE, MASK_MODE
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
    global QUANT_MODE_MXFP8, KEY_LAYOUT, SPARSE_INDICES_LAYOUT, OUT_LAYOUT, BLOCK_NUM

    CASE = dict(case)
    B = CASE["B"]
    N_q = CASE["N1"]
    N_kv = CASE["N2"]
    D = CASE["D"]
    sequence_inputs = _get_sequence_inputs(CASE)
    Q_LENGTHS = sequence_inputs["q_lengths"]
    KV_LENGTHS = sequence_inputs["kv_lengths"]
    BLOCK_SIZE = CASE["block_size"]
    MASK_MODE = CASE["mask_mode"]
    FP8_DTYPE = CASE["fp8_dtype"]
    SCALE_DTYPE = CASE["scale_dtype"]
    QUANT_GROUP_SIZE = CASE["quant_group_size"]
    Q_INPUT_LAYOUT = CASE["layout_q"]
    KV_CACHE_LAYOUT = CASE["kv_cache_layout"]
    IS_CONTIGUOUS = CASE.get("is_contiguous", True)
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
    _block_num_raw = CASE.get("blocknum", None)
    if _block_num_raw is not None and int(_block_num_raw) <= 0:
        warnings.warn(
            f"{CASE.get('name', '<unnamed>')}: blocknum is expected to be >= 0; "
            f"got {_block_num_raw} (<= 0), so it will be derived from seqused_kv",
            UserWarning,
            stacklevel=2,
        )
    BLOCK_NUM = (
        int(_block_num_raw)
        if _block_num_raw is not None and int(_block_num_raw) > 0
        else sum(math.ceil(int(seq_len) / BLOCK_SIZE) for seq_len in KV_LENGTHS)
    )


CASE = _load_initial_case()
set_active_case(CASE)

# ==============================================================================
# 固定运行常量
# ==============================================================================
# Q/K/V 使用 fp8_e4m3fn；Q/K/V descale 与 P scale 使用 fp8_e8m0。
# e8m0fnu 最小正数: 2^(-127)，用于替换 descale 中的数值 0 和非有限值。
# e8m0fnu 没有数值 0；字节 0x00 表示 2^(-127)，0xFF 才表示 NaN。
E8M0_MIN_POSITIVE = 2 ** (-127)
E8M0_MAX_FINITE = float(2**127)
_EMAX_MAP = {
    torch.float8_e4m3fn: 8,
}

# ==============================================================================
# 参考式数据生成 helper
# 与 quant_block_sparse_attn_golden.py 保持同一组织方式：
#   case -> rng/generator -> query/key/value/descale/block_table/sparse_indices
# 当前文件的 MXFP8 full-quant 差异：
#   - Q/QDescale 直接生成 TND，不经过其他 Q layout 中间格式
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
        if math.isnan(low) or math.isnan(high):
            if not (math.isnan(low) and math.isnan(high)):
                raise ValueError(
                    "nan data_range bounds must describe a constant nan value, "
                    f"got: [{low}, {high}]"
                )
            return low, high, explicit_bounds
    else:
        radius = float(data_range)
        if not math.isfinite(radius):
            # A non-finite scalar is a requested constant payload, not a
            # symmetric random radius.  This keeps CSV forms such as ``inf``,
            # ``-inf`` and ``nan`` unambiguous.
            return radius, radius, True
        if radius < 0:
            raise ValueError(f"scalar data_range must be non-negative, got: {radius}")
        low, high = -radius, radius
        explicit_bounds = False

    if low > high:
        raise ValueError(f"data_range min must not exceed max, got: [{low}, {high}]")
    return low, high, explicit_bounds


def _rand_float(shape, generator, data_range=1.0):
    low, high, explicit_bounds = _normalize_data_range(data_range)
    if low == high or (math.isnan(low) and math.isnan(high)):
        return torch.full(shape, low, dtype=torch.float32)

    random_low, random_high = low, high
    if not math.isfinite(low) or not math.isfinite(high):
        # An unbounded interval has no uniform distribution.  Generate finite
        # interior samples and pin the endpoints below so ranges such as
        # [-inf, inf] contain both requested infinities deterministically.
        if math.isinf(low):
            random_low = min(high, 0.0) - 1.0 if math.isfinite(high) else -1.0
        if math.isinf(high):
            random_high = max(low, 0.0) + 1.0 if math.isfinite(low) else 1.0

    result = (
        torch.rand(shape, dtype=torch.float32, generator=generator)
        * (random_high - random_low)
        + random_low
    )
    if explicit_bounds and result.numel() > 0:
        if low == float("-inf") and high == float("inf"):
            # Q/K/V all use D as their last dimension.  Put the two endpoints
            # in adjacent D vectors instead of the same vector: a single QK
            # dot product containing both -Inf and +Inf has order-dependent
            # MX group accumulation, while CUDA dequantization yields NaN.
            # Alternating vectors preserves both endpoints without introducing
            # that ambiguity.  Use different D groups for V so its two signs
            # also propagate to different output channels.
            vectors = result.reshape(-1, result.shape[-1])
            vectors[0::2, 0] = low
            if vectors.shape[0] > 1:
                high_index = (
                    QUANT_GROUP_SIZE
                    if result.shape[-1] > QUANT_GROUP_SIZE
                    else result.shape[-1] - 1
                )
                vectors[1::2, high_index] = high
            elif result.shape[-1] > QUANT_GROUP_SIZE:
                vectors[0, QUANT_GROUP_SIZE] = high
            elif result.shape[-1] > 1:
                vectors[0, 1] = high
            else:
                result.reshape(-1)[-1] = high
            return result
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
    finite_values = torch.where(
        torch.isfinite(grouped), torch.abs(grouped), torch.zeros_like(grouped)
    )
    max_vals = torch.max(finite_values, dim=-1)[0]
    # Choose the smallest power-of-two descale that keeps the whole group in
    # the finite E4M3 range.  floor(log2(max)) - emax only accounts for the
    # exponent bits and can still overflow values between 256 and 448.
    fp8_max = torch.finfo(fp8_dtype).max
    shared_exp = torch.ceil(torch.log2(max_vals.clamp(min=1e-12) / fp8_max))
    finite_descale = torch.where(
        max_vals == 0, torch.ones_like(shared_exp), 2**shared_exp
    )
    # E4M3FN has signed finite extrema but no infinity encoding.  Pair an
    # infinite source value with +/-448 and the largest finite E8M0 scale so
    # FP32 dequantization overflows to the requested signed infinity.  NaNs
    # remain in the FP8 payload and are deliberately ignored for group scale.
    has_inf = torch.any(torch.isinf(grouped), dim=-1)
    descale = torch.where(
        has_inf,
        torch.full_like(finite_descale, E8M0_MAX_FINITE),
        finite_descale,
    ).to(torch.float32)

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
    finite_values = torch.where(
        torch.isfinite(grouped), torch.abs(grouped), torch.zeros_like(grouped)
    )
    max_vals = torch.max(finite_values, dim=2)[0]
    fp8_max = torch.finfo(fp8_dtype).max
    shared_exp = torch.ceil(torch.log2(max_vals.clamp(min=1e-12) / fp8_max))
    finite_descale = torch.where(
        max_vals == 0, torch.ones_like(shared_exp), 2**shared_exp
    )
    has_inf = torch.any(torch.isinf(grouped), dim=2)
    descale = torch.where(
        has_inf,
        torch.full_like(finite_descale, E8M0_MAX_FINITE),
        finite_descale,
    ).to(torch.float32)

    descale_expanded = descale.repeat_interleave(group_size, dim=1)[:, :seq_len, :, :]
    quantized = (
        (tensor_fp32[:, :seq_len, :, :] / descale_expanded)
        .clamp(-448.0, 448.0)
        .to(fp8_dtype)
    )
    return quantized, descale


def make_mxfp8_block_table(
    batch,
    seq_lens,
    block_size,
    physical_block_count,
    seed,
    sparse_indices=None,
    sparse_seq_len=None,
    sparse_block_size=None,
    max_block_per_batch=None,
):
    """Generate and validate the MXFP8 ``[B, logical PA pages]`` mapping.

    This is the single block-table construction entry used by both the pytest
    golden and TTK. ``max_block_per_batch`` controls the requested table width;
    the effective width is never smaller than the pages required by seqused_kv.
    """
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens] * batch
    block_nums = [math.ceil(int(seq_len) / block_size) for seq_len in seq_lens]
    required_width = max(block_nums) if block_nums else 0
    requested_width = (
        required_width if max_block_per_batch is None else int(max_block_per_batch)
    )
    if requested_width < 0:
        raise ValueError("max_block_per_batch must be non-negative")
    max_block_num = max(required_width, requested_width)
    if max_block_num == 0:
        return torch.empty((batch, 0), dtype=torch.int32)

    physical_block_count = int(physical_block_count)
    if physical_block_count <= 0:
        raise ValueError(
            "physical_block_count must be positive when block_table is non-empty"
        )
    # Sample with replacement.  Repeated IDs are valid and model physical-page
    # sharing; values in padded columns are harmless because seqused_kv limits
    # the logical pages that the kernel may read.
    rng = random.Random(int(seed) ^ 0x5A17B10C)
    values = [rng.randrange(physical_block_count) for _ in range(batch * max_block_num)]
    block_table = torch.tensor(values, dtype=torch.int32).reshape(batch, max_block_num)
    if (
        sparse_indices is not None
        and sparse_seq_len is not None
        and sparse_block_size is not None
    ):
        _validate_reference_block_table(
            block_table,
            seq_lens,
            block_size,
            physical_block_count,
            sparse_indices,
            sparse_seq_len,
            sparse_block_size,
        )
    return block_table


def _validate_reference_block_table(
    block_table,
    seq_lens,
    block_size,
    physical_block_count,
    sparse_indices,
    sparse_seq_len,
    sparse_block_size,
):
    """Validate table shape/range and every sparse logical-page lookup."""
    if block_size <= 0 or sparse_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if block_size % sparse_block_size != 0:
        raise ValueError("sparse_block_size must divide block_size")

    seq_lens = [int(length) for length in seq_lens]
    logical_counts = [math.ceil(length / block_size) for length in seq_lens]
    minimum_width = max(logical_counts, default=0)
    if block_table.dim() != 2 or block_table.shape[0] != len(seq_lens):
        raise ValueError(
            "invalid block_table shape: "
            f"got {tuple(block_table.shape)}, expected B={len(seq_lens)}"
        )
    if block_table.shape[1] < minimum_width:
        raise ValueError(
            "block_table width is smaller than the maximum logical-page count: "
            f"got {block_table.shape[1]}, need at least {minimum_width}"
        )

    physical_block_count = int(physical_block_count)
    if block_table.numel() > 0 and (
        physical_block_count <= 0
        or torch.any(block_table < 0)
        or torch.any(block_table >= physical_block_count)
    ):
        raise ValueError(
            f"block_table physical IDs must be in [0, {physical_block_count})"
        )

    sparse_blocks_per_page = block_size // sparse_block_size
    for batch_idx, (seq_len, logical_count) in enumerate(zip(seq_lens, logical_counts)):
        max_sparse_blocks = math.ceil(seq_len / sparse_block_size)
        for head_idx in range(sparse_indices.shape[1]):
            for qb_idx in range(sparse_indices.shape[2]):
                count = int(sparse_seq_len[batch_idx, head_idx, qb_idx])
                if count < 0 or count > sparse_indices.shape[-1]:
                    raise ValueError(
                        "invalid sparse_seq_len: "
                        f"B={batch_idx}, N={head_idx}, Qb={qb_idx}, count={count}"
                    )
                sparse_ids = sparse_indices[batch_idx, head_idx, qb_idx, :count].to(
                    torch.long
                )
                if sparse_ids.numel() == 0:
                    continue
                if torch.any(sparse_ids < 0) or torch.any(
                    sparse_ids >= max_sparse_blocks
                ):
                    raise ValueError(
                        "sparse_indices references a block outside seqused_kv: "
                        f"B={batch_idx}, N={head_idx}, Qb={qb_idx}, "
                        f"ids={sparse_ids.tolist()}, max={max_sparse_blocks}"
                    )
                logical_ids = torch.div(
                    sparse_ids, sparse_blocks_per_page, rounding_mode="floor"
                )
                physical_ids = block_table[batch_idx, logical_ids]
                if torch.any(physical_ids < 0) or torch.any(
                    physical_ids >= physical_block_count
                ):
                    raise ValueError(
                        "sparse_indices references an unmapped PA page: "
                        f"B={batch_idx}, N={head_idx}, Qb={qb_idx}, "
                        f"logical={logical_ids.tolist()}, "
                        f"physical={physical_ids.tolist()}"
                    )


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


def _select_blocks(blocks, sparse_mode, rng):
    if not blocks:
        return []
    if sparse_mode == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        random_count = rng.randint(0, len(selected))
        return selected[:random_count]
    if sparse_mode == "dense":
        return blocks[:]
    raise ValueError(f"Unsupported sparse_mode: {sparse_mode!r}")


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
                    selected = _select_blocks(allowed, case["sparse_mode"], rng)
                sparse_seq_len[batch_idx, head_idx, qb_idx] = len(selected)
                if selected:
                    sparse_indices[batch_idx, head_idx, qb_idx, : len(selected)] = (
                        torch.tensor(selected, dtype=torch.int32)
                    )
    return sparse_indices, sparse_seq_len


def _make_reference_sparse_for_lengths(q_lengths, kv_lengths):
    case = _resolve_sparse_mode(CASE)
    return _make_reference_sparse_indices(
        case,
        [int(item) for item in q_lengths],
        [int(item) for item in kv_lengths],
        random.Random(case["seed"]),
    )


# ==============================================================================
# Layout 转换函数 - 数据
# MXFP8 Q/QDescale 公共输入只接受 TND；BNSD 仅作为 K/V 和 PA KV cache 的内部布局。
# ==============================================================================


def require_q_tnd(tensor, seq_lens, num_heads, name="Q"):
    """Return a contiguous MXFP8 TND tensor after one centralized check."""
    if tensor.dim() != 3:
        raise ValueError(
            f"{name} must be a rank-3 TND tensor, got shape {tuple(tensor.shape)}"
        )

    total_s = sum(seq_lens)
    if tensor.shape[0] != total_s or tensor.shape[1] != num_heads:
        raise ValueError(
            f"invalid {name} TND shape: got {tuple(tensor.shape)}, "
            f"expected ({total_s}, {num_heads}, D)"
        )
    return tensor.contiguous()


# ==============================================================================
# Layout 转换函数 - Descale
# QDescale 公共输入为 TND；K/V descale 在生成后按 NPU layout 打包。
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
    scale_tensor = torch.as_tensor(scale)
    if scale_tensor.dtype == SCALE_DTYPE:
        packed = scale_tensor.contiguous()
        nan_byte_count = int((packed.view(torch.uint8) == 0xFF).sum().item())
        if nan_byte_count:
            raise ValueError(f"{name}: {nan_byte_count} e8m0fnu NaN values (0xFF)")
        return packed.clone()

    scale_safe = sanitize_e8m0_scale(scale, name)
    packed = fp32_to_e8m0fnu(scale_safe)
    nan_byte_count = int((packed == 0xFF).sum().item())
    if nan_byte_count:
        raise ValueError(
            f"{name}: {nan_byte_count} values would become e8m0fnu NaN (0xFF)"
        )
    return packed.view(SCALE_DTYPE)


def e8m0fnu_to_fp32(scale_e8m0, name="scale"):
    """Decode E8M0 bytes to the exact FP32 powers of two used by the NPU."""
    scale_tensor = torch.as_tensor(scale_e8m0)
    if scale_tensor.dtype == SCALE_DTYPE:
        packed = scale_tensor.contiguous().view(torch.uint8)
    elif scale_tensor.dtype == torch.uint8:
        packed = scale_tensor.contiguous()
    else:
        raise TypeError(
            f"{name}: expected {SCALE_DTYPE} or torch.uint8, got {scale_tensor.dtype}"
        )

    nan_byte_count = int((packed == 0xFF).sum().item())
    if nan_byte_count:
        raise ValueError(f"{name}: {nan_byte_count} e8m0fnu NaN values (0xFF)")

    exponent = packed.to(torch.int32) - 127
    return torch.ldexp(torch.ones(packed.shape, dtype=torch.float32), exponent)


def pack_qk_scale_for_npu(scale_flat):
    """Pack Q/K D-group descale: (..., Dg) -> (..., Dg//2, 2).
    NPU 要求 Q/K descale 按 (偶, 奇) 对打包，每两个相邻 scale 值合并为一个 [..., 2]
    """
    orig_shape = scale_flat.shape
    last_dim = orig_shape[-1]
    new_shape = orig_shape[:-1] + (last_dim // 2, 2)
    return scale_flat.reshape(new_shape)


def pack_q_scale_tnd_for_npu(scale, seq_lens):
    """Pack the public TND Q descale into the NPU TND descale layout."""
    scale_tnd = require_q_tnd(scale, seq_lens, N_q, "Q descale")
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
    physical_block_count=0,
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
    if physical_block_count > 0:
        physical_block_num = physical_block_count
    else:
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
        for blk_idx in range(block_num[b]):
            blockid = int(bid_table[blk_idx])
            if blockid < 0 or blockid >= physical_block_num:
                continue
            block_offset = blk_idx * pack_block_size
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


def _validate_direct_pa_kv_shapes(key, value, k_descale, v_descale):
    expected = {
        "K": (BLOCK_NUM, N_kv, BLOCK_SIZE, D),
        "V": (BLOCK_NUM, N_kv, BLOCK_SIZE, D),
        "K descale": (BLOCK_NUM, N_kv, BLOCK_SIZE, D // 64, 2),
        "V descale": (
            BLOCK_NUM,
            N_kv,
            math.ceil(BLOCK_SIZE / 64),
            D,
            2,
        ),
    }
    actual = {
        "K": tuple(key.shape),
        "V": tuple(value.shape),
        "K descale": tuple(k_descale.shape),
        "V descale": tuple(v_descale.shape),
    }
    mismatches = [
        f"{name}: got {actual[name]}, expected {shape}"
        for name, shape in expected.items()
        if actual[name] != shape
    ]
    if mismatches:
        raise ValueError("direct PA input shape mismatch; " + "; ".join(mismatches))

    expected_pa_stride = (
        N_kv * BLOCK_SIZE * D,
        BLOCK_SIZE * D,
        D,
        1,
    )
    stride_mismatches = [
        f"{name}: got {tuple(tensor.stride())}, expected {expected_pa_stride}"
        for name, tensor in (("K", key), ("V", value))
        if tuple(tensor.stride()) != expected_pa_stride
    ]
    if stride_mismatches:
        raise ValueError(
            "direct PA_BNBD input stride mismatch; " + "; ".join(stride_mismatches)
        )


def _materialize_pa_bnsd(tensor):
    """Copy a rank-4 PA tensor into the exact segmented PA_BNBD strides.

    PyTorch may preserve a non-canonical stride on singleton dimensions after
    ``permute(...).contiguous()`` because such a tensor is already considered
    contiguous.  The QBSA kernel validates every stride exactly, including a
    size-1 N dimension, so allocate the required strides explicitly.
    """
    if tensor.dim() != 4:
        raise ValueError(
            f"PA_BNBD tensor must be rank 4, got shape {tuple(tensor.shape)}"
        )
    _, num_heads, block_size, head_dim = tensor.shape
    expected_stride = (
        int(num_heads) * int(block_size) * int(head_dim),
        int(block_size) * int(head_dim),
        int(head_dim),
        1,
    )
    materialized = torch.empty_strided(
        tuple(tensor.shape),
        expected_stride,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    materialized.copy_(tensor)
    return materialized


# ==============================================================================
# 数据生成
# ==============================================================================


def generate_mxfp8_inputs(case=None, max_block_per_batch=None, atten_mask_shape=None):
    """Generate test inputs with paired fp8 + descale.

    流程: fp32 随机 → 从 fp32 算 descale → fp8 = clamp(fp32/descale)
    fp8 和 descale 配套生成，保证 fp8 × descale ≈ 原始 fp32，反量化不溢出。
    """
    if case is not None or max_block_per_batch is not None:
        active_case = dict(CASE if case is None else case)
        if max_block_per_batch is not None:
            active_case["max_block_per_batch"] = int(max_block_per_batch)
        validate_mxfp8_case(active_case)
        set_active_case(active_case)

    case = dict(CASE)
    validate_mxfp8_case(case)
    rng = random.Random(case["seed"])
    generator = torch.Generator().manual_seed(case["seed"])

    sequence_inputs = _get_sequence_inputs(case)
    q_lengths = sequence_inputs["q_lengths"]
    kv_lengths = sequence_inputs["kv_lengths"]
    case = _resolve_sparse_mode(case)
    cu_seqlens_q = sequence_inputs["cu_seqlens_q"]
    cu_seqlens_kv = sequence_inputs["cu_seqlens_kv"]
    seqused_q = sequence_inputs["seqused_q"]
    seqused_kv = sequence_inputs["seqused_kv"]
    total_q = int(cu_seqlens_q[-1].item())

    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    head_dim = case["D"]
    # Q: fp32 随机 → MXFP8 配套量化 (per-32-group along D)
    q_fp32 = _rand_float((total_q, n1, head_dim), generator, DATA_RANGE_Q)
    query, q_descale = quantize_mxfp8_qk(
        q_fp32.unsqueeze(0), FP8_DTYPE, QUANT_GROUP_SIZE
    )
    query = query.squeeze(0)
    q_descale = q_descale.squeeze(0)

    # K/V are physical PA-cache inputs, not temporary logical BNSD tensors.
    # Generate exactly the shapes passed to the operator:
    #   K/V       [BlockNum, N2, BlockSize, D]
    #   K descale [BlockNum, N2, BlockSize, D/64, 2]
    #   V descale [BlockNum, N2, ceil(BlockSize/64), D, 2]
    k_fp32 = _rand_float((BLOCK_NUM, n2, BLOCK_SIZE, head_dim), generator, DATA_RANGE_K)
    v_fp32 = _rand_float((BLOCK_NUM, n2, BLOCK_SIZE, head_dim), generator, DATA_RANGE_V)
    _log_tensor_ranges(
        "original FP32 range",
        q=q_fp32,
        k=k_fp32,
        v=v_fp32,
    )
    key, k_descale_flat = quantize_mxfp8_qk(k_fp32, FP8_DTYPE, QUANT_GROUP_SIZE)
    key = _materialize_pa_bnsd(key)
    k_descale = pack_qk_scale_for_npu(k_descale_flat)

    value_bsnd, v_descale_bsgnd = quantize_mxfp8_v(
        v_fp32.permute(0, 2, 1, 3).contiguous(),
        FP8_DTYPE,
        QUANT_GROUP_SIZE,
    )
    value = _materialize_pa_bnsd(value_bsnd.permute(0, 2, 1, 3))
    v_descale_bnsd = v_descale_bsgnd.permute(0, 2, 1, 3).contiguous()
    v_descale = convert_v_scale_to_pa(
        v_descale_bnsd,
        [BLOCK_SIZE] * BLOCK_NUM,
        QUANT_GROUP_SIZE,
    )
    _validate_direct_pa_kv_shapes(key, value, k_descale, v_descale)

    sparse_indices, sparse_seq_len = _make_reference_sparse_indices(
        case, q_lengths, kv_lengths, rng
    )
    block_table = make_mxfp8_block_table(
        batch=batch,
        seq_lens=kv_lengths,
        block_size=BLOCK_SIZE,
        physical_block_count=BLOCK_NUM,
        seed=case["seed"],
        sparse_indices=sparse_indices,
        sparse_seq_len=sparse_seq_len,
        sparse_block_size=case["sparse_kv_block_size"],
        max_block_per_batch=case.get("max_block_per_batch"),
    )
    p_scale_value = case.get("p_scale_value")
    p_scale_type = case.get("p_scale_type", "float8_e8m0fnu")
    p_scale = None
    if p_scale_value is not None:
        p_scale_fp32 = torch.tensor([float(p_scale_value)], dtype=torch.float32)
        if not bool(torch.isfinite(p_scale_fp32).all()) or bool(
            (p_scale_fp32 <= 0).any()
        ):
            raise ValueError(
                "P scale input must be positive and finite after FP32 conversion"
            )
        if p_scale_type == "float32":
            p_scale = p_scale_fp32
        else:
            p_scale = fp32_to_e8m0fnu_safe(
                p_scale_fp32,
                "P scale input",
            )
    atten_mask = _build_causal_mask(atten_mask_shape)

    return {
        "case": case,
        "query": query,
        "key": key,
        "value": value,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "p_scale": p_scale,
        "atten_mask": atten_mask,
        "block_table": block_table,
        "sparse_indices": sparse_indices,
        "sparse_seq_len": sparse_seq_len,
        "q_lengths": q_lengths,
        "kv_lengths": kv_lengths,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
    }


def generate_data():
    """Generate inputs through the reference-style data-generation wrapper."""
    atten_mask_shape = (2048, 2048) if MASK_MODE == 3 else None
    data = generate_mxfp8_inputs(atten_mask_shape=atten_mask_shape)
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
        data["atten_mask"],
        data["block_table"],
        data["sparse_indices"],
        data["sparse_seq_len"],
        data["q_lengths"],
        data["kv_lengths"],
        data["cu_seqlens_q"],
        data["cu_seqlens_kv"],
        data["seqused_q"],
        data["seqused_kv"],
    )


# ==============================================================================
# CPU Golden
# 参考 quant_block_sparse_attn_golden.py 的 sparse block 计算流：
#   - Q/QDescale 公共输入为 TND；K/V 与 K/V descale 使用 PA_BNBD
#   - 按 sparse_indices 中记录的 KV block 顺序收集 positions，并按 256-token C1 粒度做 online 累加
#   - Q/K 使用 per-token D-group descale；V 使用 per-channel S-group descale
#   - 输出 OUT 固定 TND，MXFP8 TND LSE 固定 TN
# ==============================================================================

EMPTY_LSE = -3.4028234663852886e38
MASK_VALUE = -10000.0
# Keep the exact FP32 constants and operation order used by the VF.  For
# large-magnitude negative scores, ``x / log(2)`` and ``x * INV_LN2`` can
# round to different FP32 values before ceil.
LN2 = 0.6931471806
INV_LN2 = 1.4426950409


def _align_up_to_ln2(value, use_quant_matmul):
    value_fp32 = value.to(torch.float32).contiguous()
    if use_quant_matmul:
        value_fp32 = value_fp32.npu()
    result = torch.ceil(value_fp32 * INV_LN2) * LN2
    return result.cpu() if use_quant_matmul else result


def _accumulate_mxfp8_groups_cpu(group_dot, q_scale, k_scale):
    """Accumulate MXFP8 D-groups without materializing each scaled group.

    MX Cube keeps the E8M0 exponent separate while adding consecutive D-group
    dot products.  An intermediate becomes sticky +/-Inf only when the merged
    accumulator, not an individual ``group_dot * group_scale``, exceeds FP32.
    Keeping a finite mantissa plus a shared power-of-two exponent reproduces
    that behavior with CPU FP32 operations and avoids an FP64 fallback.
    """
    q_mantissa, q_exponent = torch.frexp(q_scale.to(torch.float32))
    k_mantissa, k_exponent = torch.frexp(k_scale.to(torch.float32))
    group_mantissa = (
        group_dot
        * q_mantissa.view(-1, 1, group_dot.shape[-1])
        * k_mantissa.view(1, -1, group_dot.shape[-1])
    )
    group_exponent = q_exponent.view(-1, 1, group_dot.shape[-1]) + k_exponent.view(
        1, -1, group_dot.shape[-1]
    )

    accumulator = group_mantissa[..., 0]
    accumulator_exponent = group_exponent[..., 0]
    accumulator_value = torch.ldexp(accumulator, accumulator_exponent)
    sticky_overflow = torch.isinf(accumulator_value)
    sticky_value = torch.where(
        sticky_overflow, accumulator_value, torch.zeros_like(accumulator_value)
    )

    for group_idx in range(1, group_dot.shape[-1]):
        merged_exponent = torch.maximum(
            accumulator_exponent, group_exponent[..., group_idx]
        )
        merged_mantissa = torch.ldexp(
            accumulator, accumulator_exponent - merged_exponent
        ) + torch.ldexp(
            group_mantissa[..., group_idx],
            group_exponent[..., group_idx] - merged_exponent,
        )
        merged_value = torch.ldexp(merged_mantissa, merged_exponent)
        overflow_now = ~sticky_overflow & torch.isinf(merged_value)
        sticky_value = torch.where(overflow_now, merged_value, sticky_value)
        sticky_overflow |= overflow_now
        accumulator = torch.where(sticky_overflow, accumulator, merged_mantissa)
        accumulator_exponent = torch.where(
            sticky_overflow, accumulator_exponent, merged_exponent
        )

    finite_value = torch.ldexp(accumulator, accumulator_exponent)
    return torch.where(sticky_overflow, sticky_value, finite_value)


def _qk_matmul_cpu(q_block, q_scale_block, k_mat, k_scale_mat, head_dim, softmax_scale):
    """CPU torch.matmul path: FP32 dequant + matmul."""
    q_dequant = q_block * _expand_d_group_scale(q_scale_block, head_dim)
    k_dequant = k_mat * _expand_d_group_scale(k_scale_mat, head_dim)
    scores = torch.matmul(q_dequant, k_dequant.transpose(0, 1)) * softmax_scale

    # Applying the descales before matmul can introduce a non-finite score that
    # does not exist in MXFP8 cube arithmetic.  For example, a finite E4M3
    # payload paired with a large E8M0 descale may overflow while materializing
    # Q as FP32.  The subsequent CPU matmul then produces +/-Inf (or Inf * 0 =
    # NaN), even when applying the Q/K group scales after the payload dot would
    # produce a finite score.  Cube follows the latter order: it accumulates
    # finite FP8 payloads in each 32-element group and applies the group
    # descales afterwards.
    #
    # Keep the established dequant + torch.matmul path for normal values and
    # repair only its artificial non-finite results.  NaNs present in either
    # FP8 payload are deliberately excluded from repair and therefore continue
    # to propagate.  A genuine Cube overflow remains non-finite after the
    # grouped recomputation, so replacing it is also semantics-preserving.
    repair_mask = ~torch.isfinite(scores)
    if not bool(repair_mask.any()):
        return scores

    payload_has_nan = torch.isnan(q_block).any(dim=-1).view(-1, 1) | torch.isnan(
        k_mat
    ).any(dim=-1).view(1, -1)
    repair_mask &= ~payload_has_nan
    if not bool(repair_mask.any()):
        return scores

    group_count = math.ceil(head_dim / QUANT_GROUP_SIZE)
    aligned_head_dim = group_count * QUANT_GROUP_SIZE
    q_payload = q_block[..., :head_dim]
    k_payload = k_mat[..., :head_dim]
    if aligned_head_dim != head_dim:
        pad_size = aligned_head_dim - head_dim
        q_payload = torch.nn.functional.pad(q_payload, (0, pad_size))
        k_payload = torch.nn.functional.pad(k_payload, (0, pad_size))

    q_grouped = q_payload.reshape(-1, group_count, QUANT_GROUP_SIZE)
    k_grouped = k_payload.reshape(-1, group_count, QUANT_GROUP_SIZE)
    group_dot = torch.matmul(
        q_grouped.permute(1, 0, 2),
        k_grouped.permute(1, 2, 0),
    ).permute(1, 2, 0)
    # MxMatmulFull carries each E8M0 exponent alongside the finite group dot.
    # Applying q_scale*k_scale to every group first can overflow too early and
    # lose a cancellation that Cube performs before converting its accumulator
    # to FP32.  Reproduce the exponent-aligned, sticky-overflow accumulation.
    cube_order_scores = _accumulate_mxfp8_groups_cpu(
        group_dot,
        q_scale_block[..., :group_count],
        k_scale_mat[..., :group_count],
    )
    cube_order_scores = cube_order_scores * softmax_scale
    return torch.where(repair_mask, cube_order_scores, scores)


def _npu_mxfp8_qk_matmul_impl(q_fp8, k_fp8, q_scale, k_scale, score_scale):
    """C1 QK matmul via npu_quant_matmul (K x Q^T orientation matching cube)."""

    d_group_count = q_fp8.shape[1] // QUANT_GROUP_SIZE

    # Pad Q to minimum M so that the 3D scale transpose is detectable by NPU.
    # When M=1 (e.g. tail Q block), q_scale_packed is [1, Dg//2, 2] and its
    # transpose [Dg//2, 1, 2] has a degenerate stride pattern that NPU cannot
    # recognise as "transposed", causing "x2 and scale transpose are not same".
    m_orig = q_fp8.shape[0]
    min_m = 8
    if m_orig < min_m:
        pad_m = min_m - m_orig
        q_fp8 = torch.nn.functional.pad(q_fp8, (0, 0, 0, pad_m))
        q_scale = torch.nn.functional.pad(q_scale, (0, 0, 0, pad_m), value=1.0)

    q_scale_e8m0 = fp32_to_e8m0fnu_safe(q_scale, "Q golden matmul descale")
    k_scale_e8m0 = fp32_to_e8m0fnu_safe(k_scale, "K golden matmul descale")
    q_scale_packed = pack_qk_scale_for_npu(q_scale_e8m0).view(torch.int8)
    k_scale_packed = pack_qk_scale_for_npu(k_scale_e8m0).view(torch.int8)

    q_npu = q_fp8.to(FP8_DTYPE).contiguous().npu()
    k_npu = k_fp8.to(FP8_DTYPE).contiguous().npu()
    q_scale_npu = q_scale_packed.contiguous().npu()
    k_scale_npu = k_scale_packed.contiguous().npu()

    expected_scale_shape_q = (q_fp8.shape[0], d_group_count // 2, 2)
    expected_scale_shape_k = (k_fp8.shape[0], d_group_count // 2, 2)
    e8m0_dtype = torch_npu.float8_e8m0fnu

    # QBSA C1 is K[S2,D] x Q^T[D,M], not Q[M,D] x K^T[D,S2].
    # The two expressions are mathematical transposes, but MXFP8 cube
    # tiling and accumulation rounding are orientation-dependent.  The
    # difference becomes visible for the large-range M-tail case 00005.
    scores_npu = torch_npu.npu_quant_matmul(
        k_npu,
        q_npu.transpose(0, 1),
        q_scale_npu.transpose(0, 1),
        pertoken_scale=k_scale_npu,
        output_dtype=torch.float32,
        pertoken_scale_dtype=e8m0_dtype,
        scale_dtype=e8m0_dtype,
        group_sizes=[1, 1, QUANT_GROUP_SIZE],
    ).transpose(0, 1)
    # VF applies softmaxScale with an AIV FP32 Muls after C1.  Keep this
    # multiply on NPU as well so large scores do not take a CPU rounding
    # path before max/subtract.
    scores_npu = scores_npu * float(score_scale)
    if m_orig < min_m:
        scores_npu = scores_npu[:m_orig, :]

    return scores_npu.cpu()


def _qk_matmul_npu(q_block, k_mat, q_scale_block, k_scale_mat, softmax_scale):
    """NPU npu_quant_matmul path for C1 QK."""
    return _npu_mxfp8_qk_matmul_impl(
        q_block, k_mat, q_scale_block, k_scale_mat, softmax_scale
    )


def _qk_matmul(
    q_block,
    q_scale_block,
    k_mat,
    k_scale_mat,
    head_dim,
    softmax_scale,
    use_quant_matmul,
):
    """Dispatch C1 QK matmul to CPU or NPU path."""
    if use_quant_matmul:
        return _qk_matmul_npu(q_block, k_mat, q_scale_block, k_scale_mat, softmax_scale)
    return _qk_matmul_cpu(
        q_block, q_scale_block, k_mat, k_scale_mat, head_dim, softmax_scale
    )


def _npu_exp_sub(lhs, rhs):
    """Evaluate exp(lhs - rhs) with NPU FP32 subtraction/exp semantics."""
    lhs_npu = lhs.to(torch.float32).contiguous().npu()
    rhs_npu = rhs.to(torch.float32).contiguous().npu()
    return torch.exp(lhs_npu - rhs_npu).cpu()


def _exp_sub(lhs, rhs, use_quant_matmul):
    """Evaluate exp(lhs - rhs) on the selected golden backend."""
    if use_quant_matmul:
        return _npu_exp_sub(lhs, rhs)
    lhs_fp32 = lhs.to(torch.float32)
    rhs_fp32 = rhs.to(torch.float32)
    return torch.exp(lhs_fp32 - rhs_fp32)


def _softmax_row_is_active(local_max):
    """Match the VF's finite-sentinel softmax state transition.

    The VF reduces each row from the finite ``-FLT_MAX`` sentinel.  Therefore
    that exact max value means the row has never started. NaN is deliberately
    active because ``NaN != -FLT_MAX`` in the kernel and must keep propagating.
    """
    return local_max != EMPTY_LSE


def _npu_mxfp8_pv_matmul_impl(p_fp8, v_fp8, p_scale, v_scale):
    """C2 PV matmul via npu_quant_matmul."""

    m_size, k_size = p_fp8.shape
    n_size = v_fp8.shape[1]
    group_count = k_size // QUANT_GROUP_SIZE

    p_scale_fp32 = p_scale.to(torch.float32)
    v_scale_fp32 = v_scale.to(torch.float32)

    # VF converts the online-max rescale through BF16 before E8M0.  Keep that
    # conversion point here; otherwise values very close to 2**n can select a
    # neighbouring E8M0 exponent on the CPU side.
    p_scale_for_pack = p_scale.to(torch.bfloat16).to(torch.float32)
    p_scale_e8m0 = fp32_to_e8m0fnu_safe(p_scale_for_pack, "P golden matmul descale")
    v_scale_e8m0 = fp32_to_e8m0fnu_safe(v_scale, "V golden matmul descale")
    p_scale_packed = pack_qk_scale_for_npu(p_scale_e8m0).view(torch.int8)
    v_scale_packed = (
        v_scale_e8m0.reshape(group_count // 2, 2, n_size)
        .permute(0, 2, 1)
        .contiguous()
        .view(torch.int8)
    )

    p_npu = p_fp8.to(FP8_DTYPE).contiguous().npu()
    v_npu = v_fp8.to(FP8_DTYPE).contiguous().npu()
    p_scale_npu = p_scale_packed.contiguous().npu()
    v_scale_npu = v_scale_packed.contiguous().npu()

    e8m0_dtype = torch_npu.float8_e8m0fnu

    result_npu = torch_npu.npu_quant_matmul(
        p_npu,
        v_npu,
        v_scale_npu,
        pertoken_scale=p_scale_npu,
        output_dtype=torch.float32,
        pertoken_scale_dtype=e8m0_dtype,
        scale_dtype=e8m0_dtype,
        group_sizes=[1, 1, QUANT_GROUP_SIZE],
    )

    return result_npu.cpu()


def _pv_matmul_cpu(
    subloop_results,
    m_new,
    v_tensor,
    v_scale,
    physical_blocks,
    block_offsets,
    n2_idx,
    nq,
    head_dim,
    l_run,
):
    """CPU torch.matmul path for C2 PV accumulation."""
    pv = torch.zeros((nq, head_dim), dtype=torch.float32)
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
        spb = physical_blocks[subloop_start:subloop_end]
        sbo = block_offsets[subloop_start:subloop_end]
        v_mat = v_tensor[spb, n2_idx, sbo, :]
        v_group_idx = torch.div(sbo, QUANT_GROUP_SIZE * 2, rounding_mode="floor")
        v_group_pair = torch.remainder(
            torch.div(sbo, QUANT_GROUP_SIZE, rounding_mode="floor"),
            2,
        )
        v_scale_mat = v_scale[spb, n2_idx, v_group_idx, :, v_group_pair].to(
            torch.float32
        )
        p_scaled = p_quant_subloop * subloop_rescale.view(nq, 1)
        v_dequant = v_mat * v_scale_mat
        pv_subloop = torch.matmul(p_scaled, v_dequant)

        # Dequantizing V before the payload dot can create an artificial Inf.
        # A masked or underflowed FP8 probability then evaluates as 0 * Inf in
        # torch.matmul and contaminates the result with NaN.  MXFP8 cube uses
        # the opposite order: it accumulates finite P/V payloads in each
        # 32-element group and applies the group descales afterwards.
        #
        # Preserve the established fast path and repair only non-finite values
        # introduced by its operation order.  NaNs already present in the P or
        # V payload must keep propagating and are deliberately not repaired.
        repair_mask = ~torch.isfinite(pv_subloop)
        if bool(repair_mask.any()):
            payload_has_nan = torch.isnan(p_quant_subloop).any(dim=-1).view(
                nq, 1
            ) | torch.isnan(v_mat).any(dim=0).view(1, head_dim)
            repair_mask &= ~payload_has_nan

            if bool(repair_mask.any()):
                subloop_k = p_quant_subloop.shape[1]
                group_count = math.ceil(subloop_k / QUANT_GROUP_SIZE)
                aligned_k = group_count * QUANT_GROUP_SIZE
                p_payload = p_quant_subloop
                v_payload = v_mat
                if aligned_k != subloop_k:
                    pad_k = aligned_k - subloop_k
                    p_payload = torch.nn.functional.pad(p_payload, (0, pad_k))
                    v_payload = torch.nn.functional.pad(v_payload, (0, 0, 0, pad_k))

                p_grouped = p_payload.reshape(nq, group_count, QUANT_GROUP_SIZE)
                v_grouped = v_payload.reshape(group_count, QUANT_GROUP_SIZE, head_dim)
                group_dot = torch.matmul(p_grouped.unsqueeze(-2), v_grouped).squeeze(-2)
                group_scale = subloop_rescale.view(nq, 1, 1) * v_scale_mat[
                    ::QUANT_GROUP_SIZE
                ].view(1, group_count, head_dim)
                scaled_group_dot = torch.where(
                    group_dot == 0,
                    torch.zeros_like(group_dot),
                    group_dot * group_scale,
                )
                cube_order_pv = scaled_group_dot.sum(dim=1)
                pv_subloop = torch.where(repair_mask, cube_order_pv, pv_subloop)

        pv += pv_subloop
        round_sum += p_subloop.sum(dim=-1) * subloop_rescale
    return pv, round_sum


def _pv_matmul_npu(
    subloop_results,
    m_new,
    v_tensor,
    v_scale,
    physical_blocks,
    block_offsets,
    n2_idx,
    nq,
    l_run,
):
    """NPU npu_quant_matmul path for C2 PV accumulation."""
    p_c2_chunks = []
    p_scale_c2_chunks = []
    v_c2_chunks = []
    v_scale_c2_chunks = []
    round_sum = torch.zeros_like(l_run)
    for (
        subloop_start,
        subloop_end,
        subloop_max,
        p_subloop,
        p_quant_subloop,
    ) in subloop_results:
        subloop_rescale = _npu_exp_sub(subloop_max, m_new)
        subloop_rescale = torch.where(
            torch.isfinite(subloop_max) & torch.isfinite(m_new),
            subloop_rescale,
            torch.zeros_like(m_new),
        )
        subloop_rescale = torch.where(
            torch.isfinite(subloop_rescale),
            subloop_rescale,
            torch.zeros_like(subloop_rescale),
        )
        spb = physical_blocks[subloop_start:subloop_end]
        sbo = block_offsets[subloop_start:subloop_end]
        v_mat = v_tensor[spb, n2_idx, sbo, :]
        v_group_idx = torch.div(sbo, QUANT_GROUP_SIZE * 2, rounding_mode="floor")
        v_group_pair = torch.remainder(
            torch.div(sbo, QUANT_GROUP_SIZE, rounding_mode="floor"),
            2,
        )
        v_scale_mat = v_scale[spb, n2_idx, v_group_idx, :, v_group_pair].to(
            torch.float32
        )
        subloop_k = subloop_end - subloop_start
        subloop_group_count = math.ceil(subloop_k / QUANT_GROUP_SIZE)
        p_c2_chunks.append(p_quant_subloop)
        p_scale_c2_chunks.append(
            subloop_rescale.view(nq, 1).expand(nq, subloop_group_count)
        )
        v_c2_chunks.append(v_mat)
        v_scale_c2_chunks.append(v_scale_mat[::QUANT_GROUP_SIZE])
        round_sum += p_subloop.sum(dim=-1) * subloop_rescale

    p_c2 = torch.cat(p_c2_chunks, dim=1)
    v_c2 = torch.cat(v_c2_chunks, dim=0)
    p_scale_c2 = torch.cat(p_scale_c2_chunks, dim=1)
    v_scale_c2 = torch.cat(v_scale_c2_chunks, dim=0)
    c2_k = p_c2.shape[1]
    c2_k_aligned = math.ceil(c2_k / 64) * 64
    if c2_k_aligned != c2_k:
        pad_k = c2_k_aligned - c2_k
        pad_groups = pad_k // QUANT_GROUP_SIZE
        p_c2 = torch.nn.functional.pad(p_c2, (0, pad_k))
        v_c2 = torch.nn.functional.pad(v_c2, (0, 0, 0, pad_k))
        p_scale_c2 = torch.nn.functional.pad(p_scale_c2, (0, pad_groups), value=1.0)
        v_scale_c2 = torch.nn.functional.pad(
            v_scale_c2, (0, 0, 0, pad_groups), value=1.0
        )
    pv = _npu_mxfp8_pv_matmul_impl(p_c2, v_c2, p_scale_c2, v_scale_c2)
    return pv, round_sum


def _pv_matmul(
    subloop_results,
    m_new,
    v_tensor,
    v_scale,
    physical_blocks,
    block_offsets,
    n2_idx,
    nq,
    l_run,
    head_dim,
    use_quant_matmul,
):
    """Dispatch C2 PV matmul to CPU or NPU path."""
    if use_quant_matmul:
        return _pv_matmul_npu(
            subloop_results,
            m_new,
            v_tensor,
            v_scale,
            physical_blocks,
            block_offsets,
            n2_idx,
            nq,
            l_run,
        )
    return _pv_matmul_cpu(
        subloop_results,
        m_new,
        v_tensor,
        v_scale,
        physical_blocks,
        block_offsets,
        n2_idx,
        nq,
        head_dim,
        l_run,
    )


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
    q_tensor, q_scale, cu_seqlens_q, batch_idx, q_start, q_end, head_idx
):
    """Gather one Q block from the only supported MXFP8 Q layout, TND."""
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
    block_table_torch,
    use_quant_matmul=False,
):
    """Reference-style CPU golden for MXFP8 block sparse attention."""
    layout_q = "TND"
    q_tensor = q_fp8
    q_scale = dequant_scale_q
    if block_table_torch is None:
        raise ValueError("PA CPU golden requires block_table_torch")

    # K/V and their descales are already physical PA-cache tensors.  The final
    # physical cache is the source of truth: repeated block_table values make
    # logical pages read the same physical data without any packing/overwrite
    # order ambiguity.
    block_table = torch.as_tensor(block_table_torch, dtype=torch.int32)
    _validate_direct_pa_kv_shapes(k_fp8, v_fp8, dequant_scale_k, dequant_scale_v)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)
    k_scale = torch.as_tensor(dequant_scale_k, dtype=torch.float32)
    v_scale = torch.as_tensor(dequant_scale_v, dtype=torch.float32)

    total_q = int(cu_seqlens_q[-1].item())
    batch = len(q_lengths)
    n1 = N_q
    n2 = N_kv
    group = n1 // n2
    head_dim = D
    softmax_scale = float(CASE["softmax_scale"])
    p_scale_value = 1.0
    if p_scale is not None and p_scale.numel() > 0:
        if p_scale.dtype == torch.float32:
            p_scale_value = float(p_scale.reshape(-1)[0].item())
        else:
            p_scale_fp32 = e8m0fnu_to_fp32(p_scale, "CPU P scale")
            p_scale_value = float(p_scale_fp32.reshape(-1)[0].item())
    ln_p_scale = 0.0 if p_scale_value == 1.0 else math.log(p_scale_value)

    attention_out = torch.zeros((total_q, n1, head_dim), dtype=torch.float32)
    softmax_lse = torch.full((total_q, n1), EMPTY_LSE, dtype=torch.float32)

    qb_max = math.ceil(max(q_lengths) / SPARSE_BLOCK_SIZE)
    logger.info(
        "[CPU Golden] reference sparse flow: layout_q=%s, OUT=TND, LSE=TN", layout_q
    )
    logger.info(
        "[CPU Golden] QK/C2 path: %s, MXFP8 group_size=%d",
        "npu_quant_matmul" if use_quant_matmul else "torch.matmul (CPU)",
        QUANT_GROUP_SIZE,
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
                    cu_seqlens_q,
                    batch_idx,
                    q_start,
                    q_end,
                    head_idx,
                )

                # Use the same finite sentinel as the kernel's online max state.
                m_run = torch.full((nq,), EMPTY_LSE, dtype=torch.float32)
                l_run = torch.zeros((nq,), dtype=torch.float32)
                acc = torch.zeros((nq, head_dim), dtype=torch.float32)

                for task_chunks in all_task_chunks:
                    positions = [pos for chunk in task_chunks for pos in chunk]
                    pos_tensor = torch.as_tensor(positions, dtype=torch.long)

                    logical_blocks = torch.div(
                        pos_tensor, BLOCK_SIZE, rounding_mode="floor"
                    )
                    block_offsets = torch.remainder(pos_tensor, BLOCK_SIZE)
                    physical_blocks = block_table[batch_idx, logical_blocks].to(
                        torch.long
                    )
                    if torch.any(physical_blocks < 0) or torch.any(
                        physical_blocks >= k_tensor.shape[0]
                    ):
                        raise ValueError(
                            "sparse position references an unmapped PA block: "
                            f"batch={batch_idx}, logical_blocks={logical_blocks.tolist()}, "
                            f"physical_blocks={physical_blocks.tolist()}"
                        )

                    k_mat = k_tensor[physical_blocks, n2_idx, block_offsets, :]
                    k_scale_mat = k_scale[
                        physical_blocks, n2_idx, block_offsets, :, :
                    ].flatten(-2)
                    valid_mask = _valid_mask_for_positions(
                        q_indices, positions, q_len, kv_len
                    )
                    scores = _qk_matmul(
                        q_block,
                        q_scale_block,
                        k_mat,
                        k_scale_mat,
                        head_dim,
                        softmax_scale,
                        use_quant_matmul,
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
                        # VF stores the ln2-aligned score max. ln(pScale) is
                        # subtracted only from the value consumed by
                        # FusedExpSub; it is not part of the online max state.
                        m_subloop = m_run
                        round_active = torch.zeros((nq,), dtype=torch.bool)
                        round_end_idx = min(round_idx + 2, len(chunk_offsets))
                        for subloop_idx in range(round_idx, round_end_idx):
                            subloop_start, subloop_end = chunk_offsets[subloop_idx]
                            s_subloop = scores[:, subloop_start:subloop_end]
                            vm_subloop = valid_mask[:, subloop_start:subloop_end]

                            masked_scores = torch.where(
                                vm_subloop,
                                s_subloop,
                                torch.full_like(s_subloop, EMPTY_LSE),
                            )
                            # VF order: score max -> softmax scale -> ceil to
                            # an ln2 multiple -> merge with the online max.
                            local_max = masked_scores.max(dim=-1).values
                            # Kernel max reduction starts from -FLT_MAX, so
                            # valid -Inf and exact -FLT_MAX cannot start a row.
                            local_max = torch.maximum(
                                local_max,
                                torch.full_like(local_max, EMPTY_LSE),
                            )
                            subloop_active = _softmax_row_is_active(local_max)
                            local_max = _align_up_to_ln2(local_max, use_quant_matmul)
                            # CUDA fused attention treats a finite negative score
                            # whose ln2-aligned max overflows to -Inf like an empty
                            # softmax contribution.  Positive overflow remains
                            # active and propagates NaN through sum=0/max=+Inf.
                            subloop_active &= ~torch.isneginf(local_max)
                            round_active |= subloop_active
                            subloop_started = _softmax_row_is_active(m_subloop)
                            m_candidate = torch.where(
                                subloop_started,
                                torch.maximum(m_subloop, local_max),
                                local_max,
                            )
                            m_subloop = torch.where(
                                subloop_active, m_candidate, m_subloop
                            )

                            # VF subtracts ln(pScale) from the aligned max
                            # immediately before FusedExpSub. Therefore the
                            # generated probability is
                            # exp(score - aligned_max + ln(pScale)).
                            p_subloop = _exp_sub(
                                s_subloop,
                                m_subloop.view(nq, 1) - ln_p_scale,
                                use_quant_matmul,
                            )
                            p_subloop = torch.where(
                                vm_subloop & subloop_active.view(nq, 1),
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
                        run_started = _softmax_row_is_active(m_before_round)
                        history_rescale = _exp_sub(
                            m_before_round, m_new, use_quant_matmul
                        )
                        history_rescale = torch.where(
                            run_started & torch.isfinite(m_new),
                            history_rescale,
                            torch.zeros_like(m_new),
                        )
                        history_rescale = torch.where(
                            torch.isfinite(history_rescale),
                            history_rescale,
                            torch.zeros_like(history_rescale),
                        )

                        pv, round_sum = _pv_matmul(
                            subloop_results,
                            m_new,
                            v_tensor,
                            v_scale,
                            physical_blocks,
                            block_offsets,
                            n2_idx,
                            nq,
                            l_run,
                            head_dim,
                            use_quant_matmul,
                        )

                        acc = acc * history_rescale.view(nq, 1) + pv
                        l_run = l_run * history_rescale + round_sum
                        m_run = torch.where(round_active, m_new, m_run)

                # Match the master kernel's final guards. LastDivNewVF writes
                # zero when sum == 0, RowInvalidUpdateVF writes zero when max
                # is the -FLT_MAX sentinel, and ComputeLseOutputVF maps either
                # condition to the same sentinel.
                row_active = _softmax_row_is_active(m_run)
                sum_nonzero = l_run != 0
                output_active = row_active & sum_nonzero
                lse_active = output_active
                safe_l = torch.where(sum_nonzero, l_run, torch.ones_like(l_run))
                attn = acc / safe_l.view(nq, 1)
                attn = torch.where(
                    output_active.view(nq, 1),
                    attn,
                    torch.zeros_like(attn),
                )
                lse = torch.log(safe_l) + m_run

                for local_idx in range(nq):
                    out_idx = q_base + q_start + local_idx
                    if bool(output_active[local_idx].item()):
                        attention_out[out_idx, head_idx] = attn[local_idx].to(
                            torch.bfloat16
                        )
                    if bool(lse_active[local_idx].item()):
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


def _optional_npu_tensor(tensor):
    if tensor is None or torch.as_tensor(tensor).numel() == 0:
        return None
    return _to_npu(tensor)


def _prepare_npu_metadata(
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    block_table,
    q_n,
    kv_n,
    layout_q,
    layout_kv,
    sparse_indices,
    sparse_seq_len,
):
    """准备 metadata 算子输入，并生成主算子依赖的 metadata。"""
    cu_seqlens_q = _to_npu(cu_seqlens_q)
    cu_seqlens_kv = _optional_npu_tensor(cu_seqlens_kv)
    seqused_q = _optional_npu_tensor(seqused_q)
    seqused_kv = _to_npu(seqused_kv)
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
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
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
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
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
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
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
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
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


def _build_causal_mask(shape=None):
    if shape is None or len(shape) != 2:
        return None
    shape = tuple(int(dim) for dim in shape)
    return torch.tril(torch.ones(shape, dtype=torch.uint8)).T.contiguous()


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
    atten_mask,
    q_lengths,
    kv_lengths,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
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

    q_layout = "TND"
    q_input = require_q_tnd(q_fp8, q_lengths, N_q, "Q").view(FP8_DTYPE)
    q_npu = q_input.npu()
    logger.info("[NPU %s] q=%s", q_layout, q_npu.shape)

    q_scale_e8m0 = fp32_to_e8m0fnu_safe(
        pack_q_scale_tnd_for_npu(dequant_scale_q, q_lengths), "Q descale"
    )
    deq_q_npu = q_scale_e8m0.npu()
    logger.info("[NPU] Q descale layout=TND, shape=%s", q_scale_e8m0.shape)

    if p_scale is None:
        p_scale_npu = None
        logger.info("[NPU] P scale=None (optional input omitted, default 1.0)")
    elif p_scale.numel() == 0:
        # Build the empty optional input from raw bytes.  This avoids dispatching
        # any cast/fill kernel for E8M0, which is not supported by aclnnFill.
        p_scale_npu = torch.empty((0,), dtype=torch.uint8).view(SCALE_DTYPE).npu()
        logger.info("[NPU] P scale=empty tensor (shape size 0, default 1.0)")
    else:
        if p_scale.dtype == torch.float32:
            p_scale_npu = p_scale.npu()
            logger.info("[NPU] P scale dtype=float32, shape=%s", p_scale_npu.shape)
        else:
            p_scale_e8m0 = fp32_to_e8m0fnu_safe(p_scale, "P scale")
            p_scale_npu = p_scale_e8m0.npu()
            logger.info(
                "[NPU] P scale dtype=%s, shape=%s",
                p_scale_e8m0.dtype,
                p_scale_e8m0.shape,
            )
    mask_arg = None if atten_mask is None else atten_mask.npu()

    if block_table_torch is None:
        raise ValueError("PA KV cache requires block_table_torch")

    _validate_direct_pa_kv_shapes(k_fp8, v_fp8, dequant_scale_k, dequant_scale_v)
    k_pa = k_fp8
    v_pa = v_fp8
    k_input = k_pa.contiguous().view(FP8_DTYPE)
    v_input = v_pa.contiguous().view(FP8_DTYPE)
    k_npu = k_input.npu()
    v_npu = v_input.npu()
    if not IS_CONTIGUOUS:
        # PA_BNBD permits padding only in paBlockStride.  Put the segment axis
        # directly after the PA block axis so N/BS/D retain dense inner strides.
        kv_cache = torch.stack([k_pa, v_pa], dim=1).npu()
        k_npu = kv_cache[:, 0]
        v_npu = kv_cache[:, 1]
        logger.info(
            "[NPU] key is_contiguous=%s, value is_contiguous=%s",
            k_npu.is_contiguous(),
            v_npu.is_contiguous(),
        )

    k_scale_pa = dequant_scale_k
    v_scale_pa = dequant_scale_v

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
        # In MXFP8 quant_mode=2 only K/V may use segmented PA block strides.
        # K/V descale tensors must retain their standard contiguous layouts.
        logger.info(
            "[NPU] segmented K/V keep descales contiguous: k=%s, v=%s",
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
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
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


def _cpu_golden_cache_name(case_name, p_scale, use_quant_matmul):
    """Key CPU golden by matmul backend and the effective P-scale value.

    Including the backend prevents CPU torch.matmul and npu_quant_matmul
    golden outputs from reusing each other's cache. It also intentionally
    avoids pre-fix cache files whose key did not encode the backend.
    """
    backend = "npu_quant_matmul" if use_quant_matmul else "torch_matmul"
    cache_prefix = f"{case_name}_{backend}_p_scale"
    if p_scale is None or p_scale.numel() == 0:
        return f"{cache_prefix}_default"
    if p_scale.dtype == torch.float32:
        p_scale_fp32_val = float(p_scale.reshape(-1)[0].item())
        return f"{cache_prefix}_fp32_{p_scale_fp32_val}"
    p_scale_e8m0 = fp32_to_e8m0fnu_safe(p_scale, "CPU cache P scale")
    raw = p_scale_e8m0.contiguous().view(torch.uint8).reshape(-1)
    if raw.numel() != 1:
        raise ValueError(f"P scale must contain one E8M0 value, got {raw.numel()}")
    return f"{cache_prefix}_{int(raw.item()):02x}"


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
    use_quant_matmul=False,
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
    logger.info("Q_LENGTHS=%s, KV_LENGTHS=%s", Q_LENGTHS, KV_LENGTHS)
    logger.info(
        "block_size=%s, sparse_block_size=%s, sparse_mode=%s, max_block_per_batch=%s",
        case.get("block_size"),
        case.get("sparse_q_block_size"),
        case.get("sparse_mode"),
        case.get("max_block_per_batch"),
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
            atten_mask,
            block_table_torch,
            sparse_indices,
            sparse_seq_len,
            q_lengths,
            kv_lengths,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
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
                "atten_mask": atten_mask,
                "block_table_torch": block_table_torch,
                "sparse_indices": sparse_indices,
                "sparse_seq_len": sparse_seq_len,
                "q_lengths": q_lengths,
                "kv_lengths": kv_lengths,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_kv": cu_seqlens_kv,
                "seqused_q": seqused_q,
                "seqused_kv": seqused_kv,
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
        atten_mask = data["atten_mask"]
        block_table_torch = data.get("block_table_torch")
        sparse_indices = data.get("sparse_indices")
        sparse_seq_len = data.get("sparse_seq_len")
        sequence_inputs = _get_sequence_inputs(CASE)
        q_lengths = data.get("q_lengths", sequence_inputs["q_lengths"])
        kv_lengths = data.get("kv_lengths", sequence_inputs["kv_lengths"])
        cu_seqlens_q = data.get("cu_seqlens_q")
        cu_seqlens_kv = data.get("cu_seqlens_kv")
        seqused_q = data.get("seqused_q")
        seqused_kv = data.get("seqused_kv")
        if cu_seqlens_q is None:
            cu_seqlens_q = sequence_inputs["cu_seqlens_q"]
        if cu_seqlens_kv is None:
            cu_seqlens_kv = sequence_inputs["cu_seqlens_kv"]
        if seqused_q is None:
            seqused_q = sequence_inputs["seqused_q"]
        if seqused_kv is None:
            seqused_kv = sequence_inputs["seqused_kv"]
        if sparse_indices is None or sparse_seq_len is None:
            sparse_indices, sparse_seq_len = _make_reference_sparse_for_lengths(
                q_lengths, kv_lengths
            )

    # Normalize E8M0 P-scale input to ensure CPU and NPU receive the same tensor.
    # FP32 P-scale is passed through as-is.
    if p_scale is not None and p_scale.numel() > 0 and p_scale.dtype != torch.float32:
        p_scale = fp32_to_e8m0fnu_safe(p_scale, "Shared P scale")
    cpu_cache_name = _cpu_golden_cache_name(case_name, p_scale, use_quant_matmul)
    logger.info("[CACHE] CPU golden key=%s", cpu_cache_name)

    _, mfu_time = _log_attention_compute_stats(
        case_id, q_lengths, kv_lengths, sparse_indices, sparse_seq_len
    )

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("\n[Done] 数据已保存，退出")
        return _make_case_result(
            case_id, "Generated", time.time() - start_time, mfu_time=mfu_time
        )

    if rdv and golden_cache.has_cpu_output(cpu_cache_name, cache_dir=rdv_cache_dir):
        logger.info("\n[Step 2] 复用 CPU Golden")
        logger.info(
            "[RDV] 已找到 case=%s 的 CPU golden，跳过 CPU 生成",
            cpu_cache_name,
        )
        cpu_out, cpu_lse = golden_cache.load_cpu_output(
            cpu_cache_name, cache_dir=rdv_cache_dir
        )
    elif "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        if rdv:
            logger.info(
                "[RDV] 未找到 case=%s 的 CPU golden，按原流程生成",
                cpu_cache_name,
            )
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
            block_table_torch,
            use_quant_matmul=use_quant_matmul,
        )
        golden_cache.save_cpu_output(
            cpu_cache_name, cpu_out, cpu_lse, cache_dir=cache_dir
        )
    elif "compare" in mode:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(
            cpu_cache_name, cache_dir=cache_dir
        )

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
            atten_mask,
            q_lengths,
            kv_lengths,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
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
    parser.add_argument(
        "--quant_matmul",
        action="store_true",
        help="使用 npu_quant_matmul 而非默认的 CPU torch.matmul 做 QK/PV 计算",
    )
    args = parser.parse_args()

    use_quant_matmul = args.quant_matmul
    logger.info(
        "[Config] matmul backend: %s",
        "npu_quant_matmul" if args.quant_matmul else "torch.matmul (CPU)",
    )

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
                        use_quant_matmul=use_quant_matmul,
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
