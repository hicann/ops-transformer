#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NumPy/Torch golden implementation for MhcPreSinkhorn ST cases."""

import numpy as np

try:
    from ttk.utilities.dtypes import numpy_bfloat16, numpy_to_torch_tensor, torch_to_numpy_tensor
except ImportError:
    numpy_bfloat16 = None
    numpy_to_torch_tensor = None
    torch_to_numpy_tensor = None


__golden__ = {
    "aclnn": {"aclnnMhcPreSinkhorn": "aclnn_mhc_pre_sinkhorn_golden"},
    "kernel": {"mhc_pre_sinkhorn": "mhc_pre_sinkhorn_golden"},
}

__input__ = {
    "aclnn": {"aclnnMhcPreSinkhorn": "aclnn_mhc_pre_sinkhorn_input"},
    "kernel": {"mhc_pre_sinkhorn": "mhc_pre_sinkhorn_input"},
}


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if torch_to_numpy_tensor is not None:
            try:
                return torch_to_numpy_tensor(value)
            except (TypeError, RuntimeError):
                pass
        return value.float().numpy()
    if hasattr(value, "cpu"):
        value = value.cpu()
        try:
            return value.numpy()
        except TypeError:
            return value.float().numpy()
    return np.asarray(value)


def _to_output(value, template, use_torch=False):
    if value is None:
        return None
    if use_torch and hasattr(template, "device"):
        import torch

        try:
            result = torch.as_tensor(value, device=template.device)
        except TypeError:
            # PyTorch cannot construct a tensor directly from NumPy's custom BF16 dtype.
            result = torch.as_tensor(np.asarray(value, dtype=np.float32), device=template.device)
        return result.to(dtype=template.dtype)
    if isinstance(template, np.ndarray):
        return np.asarray(value).astype(template.dtype, copy=False)
    return np.asarray(value)


def _sigmoid(value):
    value = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-value))


def _fill_tensor(tensor, value):
    if tensor is None:
        return
    if isinstance(tensor, np.ndarray):
        tensor[...] = np.asarray(value, dtype=np.float32).astype(tensor.dtype, copy=False)
        return
    tensor.fill_(value)


def _set_stable_inputs(x, phi, alpha, bias, hc_mult=4):
    d = _to_numpy(x).shape[-1]
    _fill_tensor(x, 0.125)
    _fill_tensor(phi, 1.0 / np.sqrt(float(hc_mult * d)))
    _fill_tensor(alpha, 0.25)
    _fill_tensor(bias, 0.0)


def _empty_outputs():
    return [np.empty((0,), dtype=np.float32) for _ in range(5)]


def _compute(x, phi, alpha, bias, hc_mult=4, num_iters=20, hc_eps=1e-6,
             norm_eps=1e-6, need_backward=True):
    x_np = _to_numpy(x)
    x_dtype = x_np.dtype
    x_f32 = x_np.astype(np.float32)
    if x_f32.ndim == 3:
        x_f32 = x_f32[None, ...]
    if x_f32.ndim != 4:
        raise ValueError(f"x must be rank 3 or 4, got {x_f32.ndim}")

    phi_f32 = _to_numpy(phi).astype(np.float32)
    alpha_f32 = _to_numpy(alpha).astype(np.float32)
    bias_f32 = _to_numpy(bias).astype(np.float32)
    batch_shape = x_f32.shape[:2]
    d = x_f32.shape[-1]
    flat_x = x_f32.reshape(*batch_shape, hc_mult * d)

    inv_rms = 1.0 / np.sqrt(np.mean(flat_x * flat_x, axis=-1, keepdims=True) + norm_eps)
    hc_before_norm = np.matmul(flat_x, phi_f32.T)
    normalized = hc_before_norm * inv_rms

    h_pre = _sigmoid(normalized[..., :hc_mult] * alpha_f32[0] + bias_f32[:hc_mult]) + hc_eps
    h_post = 2.0 * _sigmoid(
        normalized[..., hc_mult:2 * hc_mult] * alpha_f32[1]
        + bias_f32[hc_mult:2 * hc_mult]
    )
    residual_logits = (
        normalized[..., 2 * hc_mult:] * alpha_f32[2]
        + bias_f32[2 * hc_mult:]
    ).reshape(*batch_shape, hc_mult, hc_mult)

    sum_out = np.empty((2 * num_iters, *batch_shape, hc_mult), dtype=np.float32)
    norm_out = np.empty((2 * num_iters, *batch_shape, hc_mult, hc_mult), dtype=np.float32)
    row_max = np.max(residual_logits, axis=-1, keepdims=True)
    row_exp = np.exp(residual_logits - row_max)
    row_sum = np.sum(row_exp, axis=-1)
    sum_out[0] = row_sum + hc_eps
    current = row_exp / row_sum[..., None] + hc_eps
    norm_out[0] = current

    if num_iters > 0:
        column_sum = np.sum(current, axis=-2)
        sum_out[1] = column_sum + hc_eps
        current = current / (column_sum[..., None, :] + hc_eps)
        norm_out[1] = current

    for iteration in range(1, num_iters):
        row_sum = np.sum(current, axis=-1)
        sum_out[2 * iteration] = row_sum + hc_eps
        current = current / (row_sum[..., :, None] + hc_eps)
        norm_out[2 * iteration] = current
        column_sum = np.sum(current, axis=-2)
        sum_out[2 * iteration + 1] = column_sum + hc_eps
        current = current / (column_sum[..., None, :] + hc_eps)
        norm_out[2 * iteration + 1] = current

    h_res = current.reshape(*batch_shape, hc_mult * hc_mult)
    hin = np.sum(x_f32 * h_pre[..., :, None], axis=-2)
    if x_dtype.name == "bfloat16":
        if numpy_bfloat16 is not None:
            hin = hin.astype(numpy_bfloat16())
    else:
        hin = hin.astype(x_dtype, copy=False)

    if not need_backward:
        h_pre, hc_before_norm, inv_rms, sum_out, norm_out = _empty_outputs()
    return (
        hin,
        h_post.astype(np.float32),
        h_res.astype(np.float32),
        h_pre.astype(np.float32),
        hc_before_norm.astype(np.float32),
        inv_rms.astype(np.float32),
        sum_out,
        norm_out,
    )


def mhc_pre_sinkhorn_golden(x, phi, alpha, bias, hc_mult=4, num_iters=20,
                            hc_eps=1e-6, norm_eps=1e-6, need_backward=True,
                            **kwargs):
    del kwargs
    return _compute(x, phi, alpha, bias, hc_mult, num_iters, hc_eps,
                    norm_eps, need_backward)


def mhc_pre_sinkhorn_input(x, phi, alpha, bias, hc_mult=4, **kwargs):
    del kwargs
    _set_stable_inputs(x, phi, alpha, bias, int(hc_mult))
    return x, phi, alpha, bias


def aclnn_mhc_pre_sinkhorn_golden(
    x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, needBackward,
    hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, **kwargs
):
    use_torch = kwargs.get("use_torch", False)
    result = _compute(x, phi, alpha, bias, int(hcMult), int(numIters),
                      float(hcEps), float(normEps), bool(needBackward))
    templates = (hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut)
    return tuple(_to_output(value, template, use_torch)
                 for value, template in zip(result, templates))


def aclnn_mhc_pre_sinkhorn_input(
    x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, needBackward,
    hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut, **kwargs
):
    del numIters, hcEps, normEps, needBackward, hin, hPost, hRes
    del hPre, hcBeforeNorm, invRms, sumOut, normOut, kwargs
    _set_stable_inputs(x, phi, alpha, bias, int(hcMult))
