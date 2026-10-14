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

"""TTK three-way comparison with an in-process all-FP32 bench."""

import importlib.util
import logging
import sys
from pathlib import Path

import torch


# Only the current testcase is retained, bounded across a long CSV run.
_CASES = {}
TOLERANCE = {
    dtype: {
        "standard": "cross_check",
        "level": "L0",
        "mare_ratio": 10.0,
        "mere_ratio": 2.0,
        "rmse_ratio": 2.0,
        "small_value": 2**-8 if dtype == "bfloat16" else 2**-10,
        "small_value_atol": 2**-16,
    }
    for dtype in ("float16", "bfloat16", "float32")
}

# Arguments forwarded to the shared CPU reference adapter.
_REFERENCE_ARGUMENTS = (
    "q",
    "k",
    "v",
    "block_table",
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
    "sinks",
    "softmax_scale",
    "mask_mode",
    "win_left",
    "win_right",
    "max_seqlen_q",
    "max_seqlen_kv",
    "layout_q",
    "layout_kv",
    "layout_out",
    "return_softmax_lse",
)


def load_sibling_module(stem, name):
    """Lazily load a sibling impl module by a stable module name."""
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def enabled():
    try:
        from ttk.utilities.container_utils import get_global_storage
    except ImportError:
        return False
    return getattr(get_global_storage(), "compare_method", None) == "cross_check"


def prepare(arguments, testcase_name):
    """Compute before TTK Promote can change the original input precision."""
    _CASES.clear()
    if not enabled():
        return
    if testcase_name is None:
        raise ValueError("FlashAttn cross_check requires testcase_name")
    golden_module = load_sibling_module("golden", "fa_ttk_flash_attn_golden")
    bench_module = load_sibling_module("fp32_bench", "fa_ttk_flash_attn_fp32_bench")
    arguments = {
        name: value.detach().cpu() if torch.is_tensor(value) else value
        for name, value in arguments.items()
        if name in _REFERENCE_ARGUMENTS
    }
    # golden 本身已按 FlashMLA golden 的方式对齐算子中间精度(未归一化 exp
    # 权重 cast 到输入精度再进 PV), 直接占用 third_party; bench 全 FP32
    # 无 cast, 作为共同基准。
    golden = golden_module._run_reference(golden_module.FlashAttnGolden, arguments)
    bench = golden_module._run_reference(bench_module.FlashAttnFp32Bench, arguments)
    _CASES[str(testcase_name)] = (golden, bench)
    logging.info(
        "[%s] FlashAttn cross_check prepared (mixed-precision golden, all-FP32 bench)",
        testcase_name,
    )


def get_case(testcase_name):
    try:
        return _CASES[str(testcase_name)]
    except KeyError as exc:
        raise RuntimeError(
            "FlashAttn cross_check requires customize_inputs for this testcase; "
            "cached bench is missing (manual golden replay is not supported)"
        ) from exc


def compare(*outputs, testcase_name=None):
    from ttk.core_modules.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.comparison.resolve import resolve_tolerance

    compare_module = load_sibling_module("compare", "fa_ttk_flash_attn_compare")
    comparator = compare_module.COMPARATOR

    if len(outputs) < 2 or len(outputs) % 2 != 0:
        raise ValueError(
            "FlashAttn cross_check expects NPU outputs followed by golden outputs"
        )
    half = len(outputs) // 2
    _, bench_outputs = get_case(testcase_name)
    if len(bench_outputs) != half:
        raise ValueError("FlashAttn cross_check output count differs from bench")
    results = []
    for index, (npu, golden, bench) in enumerate(
        zip(outputs[:half], outputs[half:], bench_outputs)
    ):
        if golden is None:
            if bench is not None:
                raise ValueError(
                    "FlashAttn cross_check output suppression differs from bench"
                )
            results.append({"pass": True, "precision": "SUPPRESSED"})
            continue
        if npu is None or bench is None:
            results.append({"pass": False, "precision": "NO_OUTPUT"})
            continue
        dtype = str(npu.dtype).removeprefix("torch.")
        params = resolve_tolerance(TOLERANCE, None, None, [dtype], "cross_check")[
            0
        ].params
        arrays = [comparator.as_float32(value) for value in (npu, golden, bench)]
        if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
            results.append({"pass": False, "precision": "shape_mismatch"})
            continue
        # Same operand mapping as SparseFlashAttention: bench is the baseline,
        # existing all-FP32 golden occupies TTK's third_party argument.
        cross_check = CrossCheckComparison(
            arrays[0], arrays[2], index, dtype, params, third_party=arrays[1]
        )
        precision, log, passed, metrics = cross_check.compare()
        logging.info(
            "[%s] FlashAttn cross_check output %d: %s metrics=%s",
            testcase_name,
            index,
            precision,
            metrics,
        )
        results.append(
            {
                "pass": passed,
                "precision": precision,
                "metrics": metrics,
                "error_info": None if passed else log,
            }
        )
    return results
