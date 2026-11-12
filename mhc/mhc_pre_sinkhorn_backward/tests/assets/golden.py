# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MhcPreSinkhornBackward golden reference and TestSpec.

Operator: MhcPreSinkhornBackward
Function: gradient computation for MhcPreSinkhorn backward
Platform: Ascend NPU

================================================================================
Operator dimension constraints (official definition)
================================================================================

Input dimension constraints
    grad_hin:       [B, S, D] or [T, D]       (T = B*S, D = feature_dim)
    grad_h_post:    [B, S, N] or [T, N]       (N = hidden_dim)
    grad_h_res:     [B, S, N, N] or [T, N, N] (Sinkhorn matrix gradient)
    x:              [B, S, N, D] or [T, N, D] (Input feature matrix)
    phi:            [N²+2N, N*D]              (Global projection matrix; key tensor)
    alpha:          [3]                        (Three scaling coefficients)
    bias:           [N²+2N]                    (Global bias)
    h_pre:          [B, S, N] or [T, N]       (Sigmoid gate output)
    hc_before_norm: [B, S, N²+2N] or [T, N²+2N] (Projection output)
    inv_rms:        [B, S, 1] or [T, 1]       (Reciprocal RMS)
    sum_out:        [2*iter, B, S, N]         (Sum values saved during Sinkhorn iterations)
    norm_out:       [2*iter, B, S, N, N]      (Norm values saved during Sinkhorn iterations)

Output dimension constraints
    grad_x:         [B, S, N, D]              (Gradient with respect to x)
    grad_phi:       [N²+2N, N*D]              (Gradient with respect to phi; key tensor)
    grad_alpha:     [3]                        (Gradient with respect to alpha)
    grad_bias:      [N²+2N]                    (Gradient with respect to bias)

Kernel and ACLNN register through __spec__ and share _compute.
Import ml_dtypes only for NumPy bf16 conversions, allowing third_party to run remotely without the dependency installed.
"""

import numpy as np
import torch

__spec__ = {
    "mhc_pre_sinkhorn_backward": "MhcPreSinkhornBackwardTestSpec",
    "aclnnMhcPreSinkhornBackward": "AclnnMhcPreSinkhornBackwardTestSpec",
}


def _np_bfloat16():
    from ml_dtypes import bfloat16

    return bfloat16


def _torch_to_numpy(t):
    """torch.Tensor -> numpy.ndarray, supporting bf16 through a view(int16) bridge."""
    if t is None:
        return None
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(_np_bfloat16())
    return t.numpy()


def _is_fp64(a):
    if isinstance(a, torch.Tensor):
        return a.dtype == torch.float64
    return isinstance(a, np.ndarray) and a.dtype == np.float64


def _torch_dtype_of(a):
    """Return the input's torch dtype; bridge NumPy bf16 through int16 and map other dtypes by name."""
    if isinstance(a, torch.Tensor):
        return a.dtype
    d = np.dtype(a.dtype)
    return torch.bfloat16 if d == np.dtype(_np_bfloat16()) else getattr(torch, d.name)


def _as_torch(a, dtype):
    """Convert to a torch.Tensor of the specified dtype, bridging NumPy bf16 through int16."""
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a.detach().to(dtype)
    if isinstance(a, np.ndarray) and a.dtype == _np_bfloat16():
        return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16).to(dtype)
    return torch.from_numpy(np.asarray(a)).to(dtype)


def _to_output(value, template):
    """Convert the result to the output slot's dtype/device; return it unchanged if no template is provided."""
    if value is None:
        return None
    if isinstance(template, torch.Tensor):
        return value.to(dtype=template.dtype, device=template.device)
    if isinstance(template, np.ndarray):
        return _torch_to_numpy(value).astype(template.dtype, copy=False)
    return value


def _numel(t):
    """Return the total element count for a numpy.ndarray or torch.Tensor."""
    return t.numel() if isinstance(t, torch.Tensor) else np.size(t)


def _backward_inputs_empty(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
):
    """Check whether data inputs are empty; exclude phi/alpha/bias from the check."""
    return (
        _numel(grad_hin) == 0
        or _numel(grad_h_post) == 0
        or _numel(grad_h_res) == 0
        or _numel(x) == 0
        or _numel(h_pre) == 0
        or _numel(hc_before_norm) == 0
        or _numel(inv_rms) == 0
        or _numel(sum_out) == 0
        or _numel(norm_out) == 0
    )


def _sigmoid_grad(z, dy):
    sigma = torch.sigmoid(z)
    return dy * sigma * torch.sigmoid(-z)


def _rms_norm_grad(x, inv_rms, grad_inv_rms):
    nc = x.size(-1)
    return -(inv_rms**3) * x * grad_inv_rms / float(nc)


def _exp_grad(x, y_grad):
    n = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(n, device=x.device)[None, :]
    is_max = idx == arange_idx
    y = (x - x_max).exp()
    weighted = y * y_grad
    sum_all = weighted.sum(dim=-1, keepdim=True)
    # Match Select: non-max lanes must not evaluate 0 * Inf/NaN.
    correction = torch.where(is_max, sum_all, torch.zeros_like(sum_all))
    return weighted - correction


def _sinkhorn_grad(grad_h_res, sum_out, norm_out, hc_eps=1e-6):
    bs, seq_len, n, _ = grad_h_res.shape
    x_grad = grad_h_res
    iters_times_2, B, S, n = sum_out.shape
    iters = iters_times_2 // 2
    sum_out = sum_out.reshape(iters, 2, B, S, n)
    norm_out = norm_out.reshape(iters, 2, B, S, n, n)
    for i in reversed(range(iters)):
        row_sum = sum_out[i][0].view(bs, seq_len, n, 1)
        col_sum = sum_out[i][1].view(bs, seq_len, 1, n)
        x_row_normed = norm_out[i][0].view(bs, seq_len, n, n)
        grad_x_row_normed = x_grad / col_sum - (
            x_grad * x_row_normed / (col_sum**2)
        ).sum(dim=-2, keepdim=True)
        if i == 0:
            row_sum = row_sum - hc_eps
            x_row_normed = x_row_normed - hc_eps
        x_grad = grad_x_row_normed / row_sum - (
            grad_x_row_normed * x_row_normed / row_sum
        ).sum(dim=-1, keepdim=True)
    return x_grad


def _torch_backward(
    grad_hin_t,
    grad_h_post_t,
    grad_h_res_t,
    x_t,
    phi_t,
    alpha_t,
    bias_t,
    h_pre_t,
    hc_before_norm_t,
    inv_rms_t,
    sum_out_t,
    norm_out_t,
    hc_eps=1e-6,
):
    """Analytic backward preserving the promoted input precision."""
    is_3d = x_t.dim() == 3
    if _backward_inputs_empty(
        grad_hin_t,
        grad_h_post_t,
        grad_h_res_t,
        x_t,
        h_pre_t,
        hc_before_norm_t,
        inv_rms_t,
        sum_out_t,
        norm_out_t,
    ):

        def _zeros(t):
            return torch.zeros(t.shape, dtype=torch.float32, device=t.device)

        return _zeros(x_t), _zeros(phi_t), _zeros(alpha_t), _zeros(bias_t)

    if is_3d:
        x_t = x_t.unsqueeze(0)
        grad_hin_t = grad_hin_t.unsqueeze(0)
        grad_h_post_t = grad_h_post_t.unsqueeze(0)
        grad_h_res_t = grad_h_res_t.unsqueeze(0)
        h_pre_t = h_pre_t.unsqueeze(0)
        hc_before_norm_t = hc_before_norm_t.unsqueeze(0)
        inv_rms_t = inv_rms_t.unsqueeze(0)
        sum_out_t = sum_out_t.unsqueeze(1)
        norm_out_t = norm_out_t.unsqueeze(1)

    B, S, n, d = x_t.shape

    grad_x_from_hin = grad_hin_t[..., None, :] * h_pre_t[..., None]
    grad_h_pre = (grad_hin_t[..., None, :] * x_t).sum(-1)

    norm_out_forward = hc_before_norm_t * inv_rms_t
    z_pre = norm_out_forward[..., :n] * alpha_t[0] + bias_t[:n]
    z_post = norm_out_forward[..., n : 2 * n] * alpha_t[1] + bias_t[n : 2 * n]
    z_res = (norm_out_forward[..., 2 * n :] * alpha_t[2] + bias_t[2 * n :]).reshape(
        B, S, n, n
    )

    grad_z_pre = _sigmoid_grad(z_pre, grad_h_pre)
    grad_z_post = 2 * _sigmoid_grad(z_post, grad_h_post_t)
    # grad_h_res may be (B,S,N*N) or (B,S,N,N); reshape to (B,S,N,N) for _sinkhorn_grad.
    grad_h_res_4d = (
        grad_h_res_t.reshape(B, S, n, n) if grad_h_res_t.dim() == 3 else grad_h_res_t
    )
    sk_grad = _sinkhorn_grad(grad_h_res_4d, sum_out_t, norm_out_t, hc_eps)
    grad_z_res = _exp_grad(z_res, sk_grad).flatten(2)

    grad_bias = torch.cat(
        [grad_z_pre.sum((0, 1)), grad_z_post.sum((0, 1)), grad_z_res.sum((0, 1))]
    )

    grad_alpha_pre = (norm_out_forward[..., :n] * grad_z_pre).sum()
    grad_alpha_post = (norm_out_forward[..., n : 2 * n] * grad_z_post).sum()
    grad_alpha_res = (norm_out_forward[..., 2 * n :] * grad_z_res).sum()
    # torch.stack keeps the device (torch.tensor would force CPU on the XPU path).
    grad_alpha = torch.stack([grad_alpha_pre, grad_alpha_post, grad_alpha_res])

    grad_norm_out = torch.cat(
        [grad_z_pre * alpha_t[0], grad_z_post * alpha_t[1], grad_z_res * alpha_t[2]],
        dim=-1,
    )

    grad_hc_before_norm = grad_norm_out * inv_rms_t
    grad_inv_rms = (hc_before_norm_t * grad_norm_out).sum(-1, keepdim=True)

    x_flat = x_t.flatten(2)
    grad_x_from_rms = _rms_norm_grad(x_flat, inv_rms_t, grad_inv_rms)

    grad_x_from_matmul = grad_hc_before_norm @ phi_t
    grad_phi = grad_hc_before_norm.flatten(0, 1).T @ x_flat.flatten(0, 1)

    grad_x = (
        grad_x_from_hin.view(x_t.shape)
        + grad_x_from_rms.view(x_t.shape)
        + grad_x_from_matmul.view(x_t.shape)
    )

    if is_3d:
        grad_x = grad_x.view(-1, n, d)

    return grad_x, grad_phi, grad_alpha, grad_bias


def _compute(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    phi,
    alpha,
    bias,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
    hc_eps=1e-6,
):
    """Analytic backward shared by Kernel and ACLNN.

    Compute in fp64 when any input is fp64; otherwise use fp32.
    grad_x follows x's dtype; other outputs retain the computation precision.
    """
    arrays = (
        grad_hin,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
    )
    # Normalize mixed-precision inputs to satisfy matmul dtype requirements.
    compute = torch.float64 if any(_is_fp64(a) for a in arrays) else torch.float32
    grad_x, grad_phi, grad_alpha, grad_bias = _torch_backward(
        *(_as_torch(a, compute) for a in arrays),
        hc_eps,
    )
    grad_x = grad_x.to(_torch_dtype_of(x))
    return grad_x, grad_phi, grad_alpha, grad_bias


def mhc_pre_sinkhorn_backward_golden(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    phi,
    alpha,
    bias,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
    hc_eps=1e-6,
    **kwargs,
):
    """Kernel golden reference with NumPy inputs and outputs."""
    del kwargs
    grad_x, grad_phi, grad_alpha, grad_bias = _compute(
        grad_hin,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps,
    )
    return (
        _torch_to_numpy(grad_x),
        grad_phi.numpy(),
        grad_alpha.numpy(),
        grad_bias.numpy(),
    )


def aclnn_mhc_pre_sinkhorn_backward_golden(
    gradHin,
    gradHPost,
    gradHRes,
    x,
    phi,
    alpha,
    bias,
    hPre,
    hcBeforeNorm,
    invRms,
    sumOut,
    normOut,
    hcEps,
    gradX,
    gradPhi,
    gradAlpha,
    gradBias,
    **kwargs,
):
    """
    Aclnn golden for aclnnMhcPreSinkhornBackward.
    All the parameters (name & order) follow
    function `aclnnMhcPreSinkhornBackwardGetWorkspaceSize` in @aclnn_mhc_pre_sinkhorn_backward.h
    without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
    the Tensors in the parameters are all torch.Tensor.
    Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Convert results to each output slot's dtype/device.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors: gradX, gradPhi, gradAlpha, gradBias
    """
    del kwargs

    hc_eps_val = float(hcEps) if hcEps is not None else 1e-6

    result = _compute(
        gradHin,
        gradHPost,
        gradHRes,
        x,
        phi,
        alpha,
        bias,
        hPre,
        hcBeforeNorm,
        invRms,
        sumOut,
        normOut,
        hc_eps_val,
    )
    templates = (gradX, gradPhi, gradAlpha, gradBias)
    return tuple(
        _to_output(value, template) for value, template in zip(result, templates)
    )


# ============================================================================
# third_party torch reference
# ============================================================================


def _ref_sigmoid_grad(z, dy, scale=1.0):
    exp_neg = torch.exp(-z)
    denom = exp_neg + 1.0
    sigma = torch.ones_like(denom) / denom
    gate = torch.ones_like(z) / (torch.exp(z) + 1.0)
    gz = (sigma * gate) * scale
    return gz * dy


def _ref_exp_grad(x, y_grad):
    x_max, idx = x.max(dim=-1, keepdim=True)
    is_max = idx == torch.arange(x.size(-1), device=x.device)
    exp_shifted = torch.exp(x - x_max)
    weighted = exp_shifted * y_grad
    weighted_sum = weighted.sum(dim=-1, keepdim=True)
    correction = torch.where(is_max, weighted_sum, torch.zeros_like(weighted_sum))
    return weighted - correction


def _ref_sinkhorn_grad(grad_h_res, sum_out, norm_out, hc_eps=1e-6):
    """VFSinkhornBackwardPass arithmetic order, including sequential n-term sums."""
    bs, seq_len, n, _ = grad_h_res.shape
    iters = sum_out.size(0) // 2
    sums = sum_out.reshape(iters, 2, bs, seq_len, n)
    norms = norm_out.reshape(iters, 2, bs, seq_len, n, n)
    grad = grad_h_res
    for i in reversed(range(iters)):
        norm = norms[i, 0]
        col_sum = sums[i, 1].unsqueeze(-2)
        product = grad * norm
        correction = torch.zeros_like(product[..., :1, :])
        for k in range(n):
            correction = correction + product[..., k : k + 1, :]
        denom_squared = col_sum * col_sum
        correction = correction / denom_squared
        grad = grad / col_sum
        grad = grad - correction

        row_sum = sums[i, 0].unsqueeze(-1)
        if i == 0:
            norm = norm - hc_eps
            row_sum = row_sum - hc_eps
        product = grad * norm
        correction = torch.zeros_like(product[..., :1])
        for k in range(n):
            correction = correction + product[..., k : k + 1]
        grad = grad - correction
        grad = grad / row_sum
    return grad


def _ref_rms_norm_grad(x, inv_rms, grad_inv_rms):
    coefficient = inv_rms * inv_rms
    coefficient = coefficient * inv_rms
    coefficient = coefficient * grad_inv_rms
    coefficient = coefficient * (-1.0 / float(x.size(-1)))
    return x * coefficient


def _ref_hf32_round(t):
    """Round fp32 operands to HF32 precision (10-bit mantissa, RNE).

    Explicit rounding makes CPU/CUDA references independent of backend TF32
    selection. Non-finite values are preserved.
    """
    if t.dtype != torch.float32:
        return t
    t = t.contiguous()
    bits = t.view(torch.int32)
    rounding_bias = 0x0FFF + ((bits >> 13) & 1)
    rounded = (bits + rounding_bias) & -0x2000
    out = rounded.view(torch.float32)
    return torch.where(torch.isfinite(t), out, t)


def _ref_hf32_matmul(a, b):
    return _ref_hf32_round(a) @ _ref_hf32_round(b)


def _ref_torch_backward(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    phi,
    alpha,
    bias,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
    hc_eps=1e-6,
):
    """Torch reference with explicit HF32 operand rounding for both GEMMs."""
    is_3d = x.dim() == 3
    if any(
        t.numel() == 0
        for t in (
            grad_hin,
            grad_h_post,
            grad_h_res,
            x,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        )
    ):
        # Empty batches still have nonempty parameter gradients, all of which are zero.
        def _zeros(t):
            return torch.zeros_like(t, dtype=torch.float32)

        return _zeros(x), _zeros(phi), _zeros(alpha), _zeros(bias)

    # Materialize non-contiguous inputs for subsequent view operations.
    x = x.float().contiguous()
    grad_hin = grad_hin.float().contiguous()
    grad_h_post = grad_h_post.float().contiguous()
    grad_h_res = grad_h_res.float().contiguous()
    phi = phi.float().contiguous()
    alpha = alpha.float().contiguous()
    bias = bias.float().contiguous()
    h_pre = h_pre.float().contiguous()
    hc_before_norm = hc_before_norm.float().contiguous()
    inv_rms = inv_rms.float().contiguous()
    sum_out = sum_out.float().contiguous()
    norm_out = norm_out.float().contiguous()

    if is_3d:
        x = x.unsqueeze(0)
        grad_hin = grad_hin.unsqueeze(0)
        grad_h_post = grad_h_post.unsqueeze(0)
        grad_h_res = grad_h_res.unsqueeze(0)
        h_pre = h_pre.unsqueeze(0)
        hc_before_norm = hc_before_norm.unsqueeze(0)
        inv_rms = inv_rms.unsqueeze(0)
        sum_out = sum_out.unsqueeze(1)
        norm_out = norm_out.unsqueeze(1)

    b, s, n, d = x.shape

    grad_x_from_hin = grad_hin[..., None, :] * h_pre[..., None]
    grad_h_pre = (grad_hin[..., None, :] * x).sum(-1)

    nf = hc_before_norm * inv_rms
    z_pre = nf[..., :n] * alpha[0] + bias[:n]
    z_post = nf[..., n : 2 * n] * alpha[1] + bias[n : 2 * n]
    z_res = (nf[..., 2 * n :] * alpha[2] + bias[2 * n :]).reshape(b, s, n, n)

    grad_z_pre = _ref_sigmoid_grad(z_pre, grad_h_pre)
    grad_z_post = _ref_sigmoid_grad(z_post, grad_h_post, 2.0)
    grad_h_res_4d = (
        grad_h_res.reshape(b, s, n, n) if grad_h_res.dim() == 3 else grad_h_res
    )
    sk_grad = _ref_sinkhorn_grad(grad_h_res_4d, sum_out, norm_out, hc_eps)
    grad_z_res = _ref_exp_grad(z_res, sk_grad).flatten(2)

    grad_bias = torch.cat(
        [grad_z_pre.sum((0, 1)), grad_z_post.sum((0, 1)), grad_z_res.sum((0, 1))]
    )

    grad_alpha = torch.stack(
        [
            (nf[..., :n] * grad_z_pre).sum(-1).sum(),
            (nf[..., n : 2 * n] * grad_z_post).sum(-1).sum(),
            (nf[..., 2 * n :] * grad_z_res).sum(-1).sum(),
        ]
    )

    grad_norm_out = torch.cat(
        [grad_z_pre * alpha[0], grad_z_post * alpha[1], grad_z_res * alpha[2]], dim=-1
    )
    grad_hc_before_norm = grad_norm_out * inv_rms
    # Match the kernel's AR reduction over the full fusion axis.
    grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)

    x_flat = x.flatten(2)
    grad_x_from_rms = _ref_rms_norm_grad(x_flat, inv_rms, grad_inv_rms)
    # Operands are rounded explicitly; disable backend TF32 to avoid double rounding.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32 if x.is_cuda else None
    try:
        if x.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        grad_x_from_matmul = _ref_hf32_matmul(grad_hc_before_norm, phi)
        grad_phi = _ref_hf32_matmul(
            grad_hc_before_norm.flatten(0, 1).T, x_flat.flatten(0, 1)
        )
    finally:
        if x.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    grad_x = (
        grad_x_from_hin.view(x.shape)
        + grad_x_from_rms.view(x.shape)
        + grad_x_from_matmul.view(x.shape)
    )
    if is_3d:
        grad_x = grad_x.view(-1, n, d)

    return grad_x, grad_phi, grad_alpha, grad_bias


class MhcPreSinkhornBackwardTorchImpl:
    """Kernel cross_check reference; parameter names match the operator definition."""

    def __call__(
        self,
        grad_hin,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps=1e-6,
        **kwargs,
    ):
        del kwargs
        grad_x, grad_phi, grad_alpha, grad_bias = _ref_torch_backward(
            grad_hin,
            grad_h_post,
            grad_h_res,
            x,
            phi,
            alpha,
            bias,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
            float(hc_eps),
        )
        if x.dtype in (torch.bfloat16, torch.float16):
            grad_x = grad_x.to(x.dtype)
        return [grad_x, grad_phi, grad_alpha, grad_bias]


class MhcPreSinkhornBackwardAclnnTorchImpl:
    """ACLNN cross_check reference; camelCase names are required for input binding."""

    def __call__(
        self,
        gradHin,
        gradHPost,
        gradHRes,
        x,
        phi,
        alpha,
        bias,
        hPre,
        hcBeforeNorm,
        invRms,
        sumOut,
        normOut,
        hcEps=1e-6,
        **kwargs,
    ):
        return MhcPreSinkhornBackwardTorchImpl()(
            gradHin,
            gradHPost,
            gradHRes,
            x,
            phi,
            alpha,
            bias,
            hPre,
            hcBeforeNorm,
            invRms,
            sumOut,
            normOut,
            hcEps,
            **kwargs,
        )


# ============================================================================
# TestSpec
# ============================================================================

_CROSS_CHECK_TOLERANCE = {
    "bfloat16": {"standard": "cross_check"},
    "float16": {"standard": "cross_check"},
    "float32": {"standard": "cross_check"},
}


class MhcPreSinkhornBackwardTestSpec:
    """Kernel-path test spec for MhcPreSinkhornBackward."""

    golden = mhc_pre_sinkhorn_backward_golden
    third_party = {"torch": MhcPreSinkhornBackwardTorchImpl}
    tolerance = _CROSS_CHECK_TOLERANCE


class AclnnMhcPreSinkhornBackwardTestSpec:
    """ACLNN test spec using the input.py generator via __input__."""

    golden = aclnn_mhc_pre_sinkhorn_backward_golden
    third_party = {"torch": MhcPreSinkhornBackwardAclnnTorchImpl}
    tolerance = _CROSS_CHECK_TOLERANCE
