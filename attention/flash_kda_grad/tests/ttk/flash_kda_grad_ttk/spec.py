# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TestSpec for ``flash_kda_grad_ttk.npu`` (TTK E2E pathway).

Both output dtypes use TTK's built-in cross_check L2 preset without custom
thresholds or alternate tolerance profiles.

The three roles are intentionally played by three different code paths:

``golden``
    Token-level recurrence in FP32, no BF16 storage anywhere — the same
    construction as ``test_flash_kda_grad_ut._token_golden`` (plus autograd). This
    is the device-independent truth: it does **not** reproduce the operator's
    BF16 storage, so the operator's quantization cost shows up honestly in the
    ``cross_check`` numerator while the BF16 chunkwise peer carries the same
    cost in the denominator. (Reusing a BF16-emulating golden would make the
    denominator collapse to zero and turn the ratio test into a last-bit
    counting contest.)

``third_party``
    The chunkwise staged reference with the operator's BF16 cache rounding.
    This is the *peer* of ``cross_check``: it has the same error budget as the
    device kernel, which is what the ratio test needs (a zero-error peer would
    make ``mare_ratio`` explode and every case fail).

``tolerance``
    ``cross_check`` with ``level: L2`` — TTK's acceptance preset
    (``mare_ratio`` 2.0, ``mere_ratio`` 1.2, ``rmse_ratio`` 1.2). Declared for
    both output dtypes: BF16 for ``dq/dk/dv/dg/dbeta``, FP32 for
    ``dh0/dA/dBias``.

Why this composition is the right acceptance standard for a *backward* op:

* the NPU kernel and the peer consume a **bit-identical** forward cache (the
  cache is built once, host-side, and uploaded), so the comparison isolates the
  backward math instead of folding in forward-kernel error;
* the golden never touches the cache, so a broken cache would show up as a
  golden-vs-both divergence rather than hiding inside the peer;
* L2 is relative to the peer's own error, i.e. "not worse than the reference
  implementation", which is the acceptance semantics used across CANN ops.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

# TTK can load this plugin by file path, outside the package import context.
from flash_kda_grad_ttk import golden

__spec__ = {"flash_kda_grad_ttk.npu": "FlashKdaGradTestSpec"}

_GOLDEN_ABI_DTYPE = {
    "dq": "input",
    "dk": "input",
    "dv": "input",
    "dh0": "float32",
    "dg": "input",
    "dbeta": "input",
    "dA": "float32",
    "dBias": "float32",
}

_FLOAT_ATTRS = ("scale", "lower_bound")
_BOOL_ATTRS = (
    "allow_neg_eigval",
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "use_beta_sigmoid_in_kernel",
)


def _log_compare_rows(label, indices, actual, golden):
    """Display values only; these elementwise errors are not L2 verdicts."""
    lines = [
        label,
        f"{'Index':>12} {'GoldenOut':>18} {'NpuOut':>18} {'AbsDiff':>18} {'RateDiff':>18}",
    ]
    previous = -1
    for index in indices:
        if label == "First / last elements" and index > previous + 1:
            lines.append(f"{'...':>12} {'...':>18} {'...':>18} {'...':>18} {'...':>18}")
        expected, value = float(golden[index]), float(actual[index])
        # Equal infinities have zero error; NaNs remain explicit in the table.
        error = 0.0 if value == expected else abs(value - expected)
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = 0.0 if error == 0 else float(np.divide(error, abs(expected)))
        lines.append(
            f"{index:12d} {expected:18.9e} {value:18.9e} {error:18.9e} {relative:18.9e}"
        )
        previous = index
    logging.info("\n%s", "\n".join(lines))


def _log_output_comparison(name, output, golden):
    """Bound diagnostics to 20 edge rows + 10 worst rows, scanning in chunks."""
    separator = "-" * 100
    if output is None or golden is None:
        logging.info(
            "\n%s\n[%s] Output=%s Golden=%s (no value table)",
            separator,
            name,
            "None" if output is None else "present",
            "None" if golden is None else "present",
        )
        return
    actual, expected = np.asarray(output), np.asarray(golden)
    logging.info(
        "\n%s\n[%s] Output shape=%s dtype=%s; Golden shape=%s dtype=%s; elements=%d",
        separator,
        name,
        actual.shape,
        actual.dtype,
        expected.shape,
        expected.dtype,
        actual.size,
    )
    if actual.shape != expected.shape:
        logging.info(
            "Shape mismatch; value table skipped. Native TTK determines the verdict."
        )
        return
    if actual.size == 0:
        logging.info(
            "Empty output: no elements to display. Native TTK determines the verdict."
        )
        return
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    # Only block-sized float64 temporaries; never cast an entire large gradient.
    worst_indices = np.empty(0, dtype=np.int64)
    worst_errors = np.empty(0, dtype=np.float64)
    max_abs = float("nan")
    nonfinite_actual = nonfinite_expected = 0
    for start in range(0, actual.size, 65536):
        values = actual[start : start + 65536].astype(np.float64)
        references = expected[start : start + 65536].astype(np.float64)
        nonfinite_actual += int(np.count_nonzero(~np.isfinite(values)))
        nonfinite_expected += int(np.count_nonzero(~np.isfinite(references)))
        with np.errstate(invalid="ignore", over="ignore"):
            errors = np.where(values == references, 0.0, np.abs(values - references))
        finite_errors = errors[np.isfinite(errors)]
        if finite_errors.size:
            block_max = float(finite_errors.max())
            max_abs = max(max_abs, block_max) if np.isfinite(max_abs) else block_max
        # Put nonfinite differences first so NaN/Inf problems remain visible.
        ranks = np.nan_to_num(errors, nan=np.inf, posinf=np.inf)
        count = min(10, ranks.size)
        selected = np.argpartition(ranks, ranks.size - count)[-count:]
        candidates = np.concatenate((worst_indices, start + selected))
        candidate_errors = np.concatenate((worst_errors, ranks[selected]))
        order = np.lexsort((candidates, -candidate_errors))[:10]
        worst_indices, worst_errors = candidates[order], candidate_errors[order]
    edges = sorted(
        set(range(min(10, actual.size)))
        | set(range(max(0, actual.size - 10), actual.size))
    )
    _log_compare_rows("First / last elements", edges, actual, expected)
    _log_compare_rows(
        "Largest absolute differences (diagnostic, not failed-element count)",
        worst_indices,
        actual,
        expected,
    )
    logging.info(
        "[%s] elements=%d max_finite_abs_diff=%.9e nonfinite_npu=%d nonfinite_golden=%d\n%s",
        name,
        actual.size,
        max_abs,
        nonfinite_actual,
        nonfinite_expected,
        separator,
    )


class FlashKdaGradTestSpec:
    """Golden / peer / L2 acceptance standard for the FlashKDA backward operator."""

    #: TTK acceptance preset: three-way cross check at the strictest level.
    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }

    #: Dotted API path: resolved on the XPU server (`X-Spec-*`), so the peer runs
    #: with the same named inputs and attributes as the operator under test.
    third_party = {"torch": "flash_kda_grad_ttk.golden.bf16_peer"}

    def pre_compare(*outputs_and_goldens):
        """Read-only INFO tables; leave all arrays and the native verdict untouched."""
        if os.environ.get("KDA_COMPARE_DETAILS", "1") == "0":
            return
        logging.info(
            "KDA comparison details: RateDiff=AbsDiff/abs(GoldenOut) (zero golden: 0 or inf). "
            "Tables are diagnostic only; the following native cross_check L2 comparison "
            "determines PASS/FAIL, including the third-party reference."
        )
        count = len(_GOLDEN_ABI_DTYPE)
        if len(outputs_and_goldens) != 2 * count:
            logging.warning(
                "KDA comparison table skipped: expected %d output/golden slots, got %d",
                2 * count,
                len(outputs_and_goldens),
            )
            return
        for index, name in enumerate(_GOLDEN_ABI_DTYPE):
            _log_output_comparison(
                name, outputs_and_goldens[index], outputs_and_goldens[count + index]
            )
        # No return value: TTK continues with its original unmodified arrays.

    def golden(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        do,
        dht,
        cu_seqlens=None,
        seqused=None,
        *,
        scale=None,
        gate_mode="bounded",
        lower_bound=-5.0,
        allow_neg_eigval=False,
        layout="BNSD",
        cache_source="ref",
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        disable_recompute=None,
        **kwargs,
    ):
        """FP32 token autograd reference, cast to each output's ABI dtype."""
        # Recompute controls device cache delivery, not the FP32 golden math.
        del cache_source, disable_recompute, kwargs
        grads = golden.token_golden_backward(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            do,
            dht,
            scale=float(scale) if scale is not None else float(q.shape[-1]) ** -0.5,
            gate_mode=str(gate_mode).lower(),
            lower_bound=None if lower_bound is None else float(lower_bound),
            allow_neg_eigval=bool(allow_neg_eigval),
            layout=layout,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
            use_gate_in_kernel=bool(use_gate_in_kernel),
            use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        )
        input_dtype = q.dtype if q.is_floating_point() else torch.bfloat16
        out = []
        for name, value in zip(_GOLDEN_ABI_DTYPE, grads):
            if value is None:
                out.append(None)
                continue
            dtype = input_dtype if _GOLDEN_ABI_DTYPE[name] == "input" else torch.float32
            out.append(value.to(dtype))
        return out
