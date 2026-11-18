# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TestSpec for ``flash_gdn_grad_ttk.npu`` (TTK E2E pathway).

``golden`` is the FP64 token-level recurrence (device-independent truth),
``third_party`` the FP32 recurrence carrying the operator's storage rounding —
Normalized Q/K values use BF16 storage while recurrent states remain FP32,
matching the staged implementation. Acceptance is TTK's L2 preset (2.0 / 1.2 / 1.2).
"""

from __future__ import annotations

import logging

import numpy as np

__spec__ = {"flash_gdn_grad_ttk.npu": "FlashGdnGradTestSpec"}

#: Optional inputs whose golden dtype follows the corresponding input tensor.
_GOLDEN_ABI_DTYPE = {
    "dq": "input",
    "dk": "input",
    "dv": "input",
    "dh0": "float32",
    "dg": "input",
    "dbeta": "input",
    "dA_log": "float32",
    "ddt_bias": "float32",
}

_FLOAT_ATTRS = ("scale",)
_BOOL_ATTRS = ("use_beta_sigmoid_in_kernel",)


def _log_output_comparison(name, output, expected_golden):
    """Read-only diagnostic table; the verdict stays with native TTK."""
    if output is None or expected_golden is None:
        logging.info("[%s] output/golden missing", name)
        return
    actual, expected = np.asarray(output), np.asarray(expected_golden)
    if actual.shape != expected.shape:
        logging.info(
            "[%s] shape mismatch: %s vs %s", name, actual.shape, expected.shape
        )
        return
    if actual.size == 0:
        logging.info("[%s] empty output", name)
        return
    flat_actual = actual.reshape(-1).astype(np.float64)
    flat_expected = expected.reshape(-1).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.where(
            flat_actual == flat_expected, 0.0, np.abs(flat_actual - flat_expected)
        )
    finite = errors[np.isfinite(errors)]
    max_abs = float(finite.max()) if finite.size else float("nan")
    worst = np.argsort(np.nan_to_num(errors, nan=np.inf))[-5:][::-1]
    lines = [
        f"[{name}] elements={actual.size} max_finite_abs_diff={max_abs:.6e} "
        f"nonfinite_npu={int(np.count_nonzero(~np.isfinite(flat_actual)))} "
        f"nonfinite_golden={int(np.count_nonzero(~np.isfinite(flat_expected)))}",
        f"{'Index':>12} {'GoldenOut':>18} {'NpuOut':>18} {'AbsDiff':>18}",
    ]
    for index in worst:
        lines.append(
            f"{int(index):12d} {flat_expected[index]:18.9e} "
            f"{flat_actual[index]:18.9e} {errors[index]:18.9e}"
        )
    logging.info("\n%s", "\n".join(lines))


class FlashGdnGradTestSpec:
    """Golden / peer / L2 acceptance standard for the GDN backward operator."""

    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }

    #: Dotted path resolved on the XPU (CPU) peer server.
    third_party = {"torch": "flash_gdn_grad_ttk_golden.bf16_peer"}

    def pre_compare(*outputs_and_goldens):
        """Log read-only value tables; never modifies arrays or the verdict."""
        try:
            total = len(outputs_and_goldens) // 2
            for index in range(total):
                _log_output_comparison(
                    f"output[{index}]",
                    outputs_and_goldens[index],
                    outputs_and_goldens[total + index],
                )
        except Exception as error:  # diagnostics must never fail a case
            logging.info("comparison table skipped: %s", error)
