# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 910B functional and precision tests for GroupedMatmulQuant."""

import importlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
import torch

from golden import (
    check_mixed_tolerance,
    grouped_matmul_quant_golden,
    pack_int4_to_int32,
    round_to_bfloat16,
)


@dataclass(frozen=True)
class PrecisionCase:
    name: str
    m: int
    k: int
    n: int
    group_list: Optional[Tuple[int, ...]]
    scale_group_size: int
    dtype: torch.dtype
    seed: int
    int4_boundary_values: bool = False


CASES = (
    PrecisionCase(
        "fp16_single_tail_split_k", 17, 64, 80, None, 32, torch.float16, 20260901, True
    ),
    PrecisionCase(
        "bf16_zero_token_expert", 33, 128, 64, (0, 1, 33), 64, torch.bfloat16, 20260902
    ),
    PrecisionCase(
        "fp16_imbalanced_groups", 257, 96, 48, (1, 2, 257), 32, torch.float16, 20260903
    ),
    # With 20 cube cores: (M/128) * (N/256) == 180, selecting non-Split-K.
    PrecisionCase(
        "fp16_non_split_k", 5760, 32, 1024, None, 32, torch.float16, 20260904
    ),
)


def _import_registration_modules() -> Dict[str, str]:
    module_names = os.getenv(
        "GROUPED_MATMUL_QUANT_MODULES", "npu_ops_transformer_ext"
    ).split(",")
    diagnostics = {}
    for module_name in module_names:
        module_name = module_name.strip()
        if not module_name:
            continue
        try:
            module = importlib.import_module(module_name)
            diagnostics[module_name] = (
                f"loaded from {getattr(module, '__file__', '<unknown>')}"
            )
        except (ImportError, OSError, RuntimeError) as error:
            diagnostics[module_name] = f"{type(error).__name__}: {error}"
    return diagnostics


def _resolve_attr(path: str):
    value = torch
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _resolve_operator():
    import_diagnostics = _import_registration_modules()
    configured = os.getenv("GROUPED_MATMUL_QUANT_OP")
    candidates = (
        [configured]
        if configured
        else ["ops.npu_ops_transformer_ext.grouped_matmul_quant"]
    )
    resolution_errors = {}
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return _resolve_attr(candidate), candidate
        except (AttributeError, RuntimeError) as error:
            resolution_errors[candidate] = f"{type(error).__name__}: {error}"
    import_details = (
        "; ".join(
            f"{module_name}: {detail}"
            for module_name, detail in import_diagnostics.items()
        )
        or "no registration module configured"
    )
    resolution_details = "; ".join(
        f"{candidate}: {detail}" for candidate, detail in resolution_errors.items()
    )
    pytest.fail(
        "GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set "
        "GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch). "
        f"Import diagnostics: {import_details}. Resolution diagnostics: {resolution_details}"
    )


def _rounded_input(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(values.astype(np.float32)).to(dtype)


def _to_numpy_float(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().numpy()


def _make_inputs(case: PrecisionCase):
    rng = np.random.default_rng(case.seed)
    x = _rounded_input(rng.uniform(-1.0, 1.0, size=(case.m, case.k)), case.dtype)
    group_num = 1 if case.group_list is None else len(case.group_list)
    logical_weight = rng.integers(
        -8, 8, size=(group_num, case.k, case.n), dtype=np.int8
    )
    if case.int4_boundary_values:
        logical_weight.reshape(-1)[0::2] = -8
        logical_weight.reshape(-1)[1::2] = 7
    packed = torch.from_numpy(pack_int4_to_int32(logical_weight).copy())

    scale_shape = (group_num, case.k // case.scale_group_size, case.n)
    scale = _rounded_input(rng.uniform(0.01, 0.10, size=scale_shape), case.dtype)
    offset = _rounded_input(rng.uniform(-1.0, 1.0, size=scale_shape), torch.float16)
    group_list = (
        None
        if case.group_list is None
        else torch.tensor(case.group_list, dtype=torch.int64)
    )
    return x, packed, scale, offset, group_list


def _call_operator(op, x, packed, scale, offset, group_list, scale_group_size):
    result = op(
        x.npu(),
        packed.npu(),
        scale.npu(),
        offset.npu(),
        None if group_list is None else group_list.npu(),
        scale_group_size,
    )
    assert isinstance(result, torch.Tensor), (
        "grouped_matmul_quant schema declares a Tensor result, "
        f"but the registered implementation returned {type(result).__name__}"
    )
    torch.npu.synchronize()
    return result


def test_int4_pack_round_trip():
    rng = np.random.default_rng(20260900)
    logical = rng.integers(-8, 8, size=(3, 32, 48), dtype=np.int8)
    packed = pack_int4_to_int32(logical)
    from golden import unpack_int32_to_int4

    np.testing.assert_array_equal(unpack_int32_to_int4(packed), logical)


def test_int4_pack_known_storage_layout():
    logical = np.zeros((1, 32, 32), dtype=np.int8)
    logical[0, 0, :16] = np.arange(-8, 8, dtype=np.int8)
    logical[0, 17, 18] = 7

    expected = np.zeros((1, 2, 2, 16, 2), dtype=np.int32)
    # Nibbles [-8, ..., -1] and [0, ..., 7], least-significant nibble first.
    expected[0, 0, 0, 0, 0] = -19088744  # 0xFEDCBA98 as signed INT32
    expected[0, 0, 0, 0, 1] = 1985229328  # 0x76543210
    # (K, N) = (17, 18) maps to (K1, N1, K0, word, nibble) = (1, 1, 1, 0, 2).
    expected[0, 1, 1, 1, 0] = 7 << 8

    np.testing.assert_array_equal(pack_int4_to_int32(logical), expected)


def test_golden_nonzero_offset_and_zero_token_expert():
    logical = np.zeros((2, 32, 16), dtype=np.int8)
    logical[1, :, 0::2] = -8
    logical[1, :, 1::2] = 7
    packed = pack_int4_to_int32(logical)
    scale = np.full((2, 1, 16), 0.5, dtype=np.float16)
    offset = np.ones((2, 1, 16), dtype=np.float16)

    golden = grouped_matmul_quant_golden(
        np.ones((2, 32), dtype=np.float16), packed, scale, offset, (0, 2), 32, "float16"
    )
    expected_row = np.empty(16, dtype=np.float64)
    expected_row[0::2] = -112.0
    expected_row[1::2] = 128.0
    np.testing.assert_array_equal(golden, np.stack((expected_row, expected_row)))


def test_mixed_tolerance_ratio_cap_and_nonfinite_guards():
    golden = np.zeros(100, dtype=np.float64)

    within_cap = golden.copy()
    within_cap[0] = 0.05
    result = check_mixed_tolerance(within_cap, golden, "float16")
    assert result.passed
    assert result.matched_ratio == pytest.approx(0.99)

    above_cap = golden.copy()
    above_cap[0] = 0.2
    assert not check_mixed_tolerance(above_cap, golden, "float16").passed

    nonfinite = golden.copy()
    nonfinite[0] = np.nan
    assert not check_mixed_tolerance(nonfinite, golden, "float16").passed


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_grouped_matmul_quant_precision(case: PrecisionCase):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    op, op_path = _resolve_operator()
    x, packed, scale, offset, group_list = _make_inputs(case)
    actual = _call_operator(
        op, x, packed, scale, offset, group_list, case.scale_group_size
    )
    assert tuple(actual.shape) == (case.m, case.n)
    assert actual.dtype == case.dtype
    assert actual.device.type == "npu"

    dtype_name = "float16" if case.dtype == torch.float16 else "bfloat16"
    x_np = _to_numpy_float(x)
    scale_np = _to_numpy_float(scale)
    if dtype_name == "bfloat16":
        scale_np = round_to_bfloat16(scale_np)
    golden = grouped_matmul_quant_golden(
        x_np,
        packed.numpy(),
        scale_np,
        _to_numpy_float(offset),
        case.group_list,
        case.scale_group_size,
        dtype_name,
    )
    actual_np = _to_numpy_float(actual)
    result = check_mixed_tolerance(actual_np, golden, dtype_name)
    print(
        f"case={case.name} op={op_path} dtype={dtype_name} matched_ratio={result.matched_ratio:.6f} "
        f"max_abs_error={result.max_abs_error:.8e} max_rel_error={result.max_rel_error:.8e} "
        f"atol={result.atol:.8e} rtol={result.rtol:.8e} cap={result.max_abs_error_limit:.8e}"
    )
    assert result.passed, (
        f"{case.name} failed mixed tolerance: matched_ratio={result.matched_ratio:.6f}, "
        f"max_abs_error={result.max_abs_error:.8e}"
    )
