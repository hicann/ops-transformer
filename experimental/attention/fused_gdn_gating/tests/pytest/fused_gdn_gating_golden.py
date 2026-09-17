# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import gc
import logging

import torch
import torch_npu

import result_compare_method

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

DEVICE_ID = 0
SEED = 42

torch.npu.config.allow_internal_format = True


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure PyTorch)
# ---------------------------------------------------------------------------


def golden_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU golden reference for fused_gdn_gating.

    Uses the same softplus threshold semantics as the kernel:
        where(beta * x <= threshold, log(1 + exp(beta * x)) / beta, x)

    Returns:
        g:           [1, batch, num_heads], fp32.
        beta_output: [1, batch, num_heads], original dtype.
    """
    batch, num_heads = a.shape
    compute_dtype = torch.float32

    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    x = a_f + dt_bias_expanded
    beta_x = beta * x
    softplus_o = torch.where(
        beta_x <= threshold,
        torch.log1p(torch.exp(beta_x)) / beta,
        x,
    )

    g = -torch.exp(A_log_expanded) * softplus_o
    g = g.unsqueeze(0)

    beta_output = torch.sigmoid(b_f).to(b.dtype)
    beta_output = beta_output.unsqueeze(0)

    return g, beta_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(
    num_heads: int,
    batch: int,
    dtype: torch.dtype,
    param_dtype: torch.dtype = torch.float32,
    seed: int = SEED,
):
    """Build random tensors on CPU for both golden and NPU execution."""
    torch.manual_seed(seed)
    A_log = torch.randn(num_heads, dtype=param_dtype)
    dt_bias = torch.randn(num_heads, dtype=param_dtype)
    a = torch.randn(batch, num_heads, dtype=dtype)
    b = torch.randn(batch, num_heads, dtype=dtype)
    return A_log, a, b, dt_bias


def force_softplus_threshold_cases(
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float,
    threshold: float,
) -> None:
    """Force beta * (a + dt_bias) to cover threshold and non-threshold paths."""
    if a.shape[0] < 4 or a.shape[1] < 4:
        return

    dt_bias[:4] = 0
    boundary = threshold / beta
    a[0, 0] = boundary + 2.0  # linear branch
    a[1, 1] = boundary  # softplus branch at equality
    a[2, 2] = boundary - 0.5  # softplus branch below threshold
    a[3, 3] = -boundary - 2.0  # negative softplus input


# ---------------------------------------------------------------------------
# NPU op execution
# ---------------------------------------------------------------------------


def npu_op_exec(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the operator on NPU and return CPU tensors."""
    if not A_log.is_contiguous():
        A_log = A_log.contiguous()
    if not dt_bias.is_contiguous():
        dt_bias = dt_bias.contiguous()

    g, beta_output = torch.ops.custom.npu_fused_gdn_gating(
        A_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        float(beta),
        float(threshold),
    )
    return g.cpu(), beta_output.cpu()


# ---------------------------------------------------------------------------
# Shape / dtype validation
# ---------------------------------------------------------------------------


def check_output_shapes(
    npu_g: torch.Tensor,
    npu_beta: torch.Tensor,
    batch: int,
    num_heads: int,
    dtype: torch.dtype,
):
    """Validate output shapes and dtypes."""
    assert npu_g.shape == (1, batch, num_heads), f"unexpected g shape: {npu_g.shape}"
    assert npu_g.dtype == torch.float32, f"unexpected g dtype: {npu_g.dtype}"
    assert npu_beta.shape == (1, batch, num_heads), (
        f"unexpected beta shape: {npu_beta.shape}"
    )
    assert npu_beta.dtype == dtype, (
        f"unexpected beta dtype: {npu_beta.dtype}, expected {dtype}"
    )


# ---------------------------------------------------------------------------
# Eager run (CPU golden + NPU + compare)
# ---------------------------------------------------------------------------


def run_fused_gdn_gating_eager(case):
    """Run fused_gdn_gating on CPU (golden) and NPU, then compare results.

    Args:
        case: dict with keys num_heads, batch, dtype, param_dtype, beta,
              threshold, force_threshold.
    """
    torch_npu.npu.set_device(DEVICE_ID)

    num_heads = case["num_heads"]
    batch = case["batch"]
    dtype = case["dtype"]
    param_dtype = case["param_dtype"]
    beta = case["beta"]
    threshold = case["threshold"]
    force_threshold = case["force_threshold"]

    # Generate inputs
    A_log, a, b, dt_bias = make_inputs(num_heads, batch, dtype, param_dtype)

    # Force threshold cases
    if force_threshold:
        force_softplus_threshold_cases(a, dt_bias, beta, threshold)

    # CPU golden
    g_golden, beta_out_golden = golden_fused_gdn_gating(
        A_log, a, b, dt_bias, beta, threshold
    )

    # NPU execution
    g_npu, beta_out_npu = npu_op_exec(A_log, a, b, dt_bias, beta, threshold)

    # Shape / dtype validation
    check_output_shapes(g_npu, beta_out_npu, batch, num_heads, dtype)

    logger.info(
        "num_heads=%d, batch=%d, dtype=%s, param_dtype=%s, beta=%s, threshold=%s",
        num_heads,
        batch,
        dtype,
        param_dtype,
        beta,
        threshold,
    )
    logger.info("A_log: shape %s, dtype %s", A_log.shape, A_log.dtype)
    logger.info("a: shape %s, dtype %s", a.shape, a.dtype)
    logger.info("b: shape %s, dtype %s", b.shape, b.dtype)
    logger.info("dt_bias: shape %s, dtype %s", dt_bias.shape, dt_bias.dtype)

    # Result comparison
    print(
        "--------------------------------------------------------------check g output--------------------------------------------------------------"
    )
    g_result, g_pct = result_compare_method.check_result(g_golden, g_npu)
    print(
        "--------------------------------------------------------------check beta_output--------------------------------------------------------------"
    )
    beta_result, beta_pct = result_compare_method.check_result(
        beta_out_golden, beta_out_npu
    )

    del A_log, a, b, dt_bias, g_golden, beta_out_golden, g_npu, beta_out_npu
    gc.collect()
    torch.npu.empty_cache()

    assert g_result == "Pass", (
        f"{case['case_id']} g precision check failed: pass_rate={g_pct:.4f}%"
    )
    assert beta_result == "Pass", (
        f"{case['case_id']} beta_output precision check failed: pass_rate={beta_pct:.4f}%"
    )
