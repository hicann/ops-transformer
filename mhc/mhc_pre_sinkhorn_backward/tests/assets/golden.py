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
MhcPreSinkhornBackward 算子 Golden实现

算子原型：MhcPreSinkhornBackward
功能：MhcPreSinkhorn反向传播梯度计算
平台：Ascend NPU (arch35)

================================================================================
算子维度约束（官方定义）
================================================================================

【输入维度约束】
    grad_hin:       [B, S, D] 或 [T, D]       (T = B*S, D = feature_dim)
    grad_h_post:    [B, S, N] 或 [T, N]       (N = hidden_dim)
    grad_h_res:     [B, S, N, N] 或 [T, N, N] (Sinkhorn矩阵梯度)
    x:              [B, S, N, D] 或 [T, N, D] (输入特征矩阵)
    phi:            [N²+2N, N*D]              (全局投影矩阵) ★关键
    alpha:          [3]                        (三路缩放系数)
    bias:           [N²+2N]                    (全局偏置)
    h_pre:          [B, S, N] 或 [T, N]       (sigmoid门控输出)
    hc_before_norm: [B, S, N²+2N] 或 [T, N²+2N] (投影后输出)
    inv_rms:        [B, S, 1] 或 [T, 1]       (RMS倒数)
    sum_out:        [2*iter, B, S, N]         (Sinkhorn迭代sum记录)
    norm_out:       [2*iter, B, S, N, N]      (Sinkhorn迭代norm记录)

【输出维度约束】
    grad_x:         [B, S, N, D]              (对x的梯度)
    grad_phi:       [N²+2N, N*D]              (对phi的梯度) ★关键
    grad_alpha:     [3]                        (对alpha的梯度)
    grad_bias:      [N²+2N]                    (对bias的梯度)

【关键理解】
    1. phi形状：[N²+2N, N*D] = [24, 256]
       - N²+2N = 16+8 = 24 (三路输出维度)
       - N*D = 4*64 = 256 (输入特征维度)

    2. phi作用：全局投影矩阵
       - 正向：hc_before_norm = x_flat @ phi.T + bias
       - x_flat: [T, N*D] = [16, 256]
       - phi.T: [N*D, N²+2N] = [256, 24]
       - 输出：[T, N²+2N] = [16, 24]

    3. grad_phi计算公式（矩阵乘法反向）：
       - grad_phi = grad_hc_before_norm.T @ x_flat
       - grad_hc.T: [N²+2N, T] = [24, 16]
       - x_flat: [T, N*D] = [16, 256]
       - 结果：[N²+2N, N*D] = [24, 256] ✓

================================================================================
"""

import numpy as np
import torch
from ml_dtypes import bfloat16 as np_bfloat16


def _torch_to_numpy(t):
    """torch.Tensor -> numpy.ndarray，支持 bf16（通过 view(int16) 桥接）"""
    if t is None:
        return None
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np_bfloat16)
    return t.numpy()


def _numpy_to_torch(t, dtype=torch.float32):
    """numpy.ndarray -> torch.Tensor，支持 bf16 输入（先 view int16 读出再转目标 dtype）"""
    if t is None:
        return None
    if isinstance(t, np.ndarray) and t.dtype == np_bfloat16:
        return torch.from_numpy(t.view(np.int16).copy()).view(torch.bfloat16).to(dtype)
    if hasattr(t, "numpy"):
        return t.to(dtype)
    return torch.from_numpy(t).to(dtype)


__golden__ = {
    "kernel": {"mhc_pre_sinkhorn_backward": "mhc_pre_sinkhorn_backward_golden"},
    "aclnn": {"aclnnMhcPreSinkhornBackward": "aclnn_mhc_pre_sinkhorn_backward_golden"},
}


def _sigmoid_grad(z, dy, is_pre=False, hc_eps=1e-6):
    y = torch.sigmoid(z)
    sigma = y - hc_eps if is_pre else y
    return dy * sigma * (1 - sigma)


def _rms_norm_grad(x, inv_rms, grad_inv_rms):
    nc = x.size(-1)
    return -(inv_rms**3) * x * grad_inv_rms / float(nc)


def _exp_grad(x, y_grad):
    n = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(n, device=x.device)[None, :]
    is_max = idx == arange_idx
    y = (x - x_max).exp()
    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)
    return y * y_grad - is_max * sum_all


def _sinkhorn_grad(grad_h_res, sum_out, norm_out):
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
        x_grad = grad_x_row_normed / row_sum - (
            grad_x_row_normed * x_row_normed / row_sum
        ).sum(dim=-1, keepdim=True)
    return x_grad


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
    """
    MhcPreSinkhornBackward 标杆函数。
    所有输入均为 numpy.ndarray，返回 numpy.ndarray。
    """

    grad_hin_t = _numpy_to_torch(grad_hin)
    grad_h_post_t = _numpy_to_torch(grad_h_post)
    grad_h_res_t = _numpy_to_torch(grad_h_res)
    x_t = _numpy_to_torch(x)
    phi_t = _numpy_to_torch(phi)
    alpha_t = _numpy_to_torch(alpha)
    bias_t = _numpy_to_torch(bias)
    h_pre_t = _numpy_to_torch(h_pre)
    hc_before_norm_t = _numpy_to_torch(hc_before_norm)
    inv_rms_t = _numpy_to_torch(inv_rms)
    sum_out_t = _numpy_to_torch(sum_out)
    norm_out_t = _numpy_to_torch(norm_out)

    is_3d = x_t.dim() == 3
    is_empty = x_t.numel() == 0
    if is_empty:
        grad_x_np = np.zeros(x.shape, dtype=np.float32)
        grad_phi_np = np.zeros(phi.shape, dtype=np.float32)
        grad_alpha_np = np.zeros(alpha.shape, dtype=np.float32)
        grad_bias_np = np.zeros(bias.shape, dtype=np.float32)
        return grad_x_np, grad_phi_np, grad_alpha_np, grad_bias_np

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

    grad_z_pre = _sigmoid_grad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)
    grad_z_post = 2 * _sigmoid_grad(z_post, grad_h_post_t, is_pre=False, hc_eps=hc_eps)
    # grad_h_res可能为(B,S,N*N)或(B,S,N,N), 统一reshape为(B,S,N,N)供_sinkhorn_grad处理
    grad_h_res_4d = (
        grad_h_res_t.reshape(B, S, n, n) if grad_h_res_t.dim() == 3 else grad_h_res_t
    )
    sk_grad = _sinkhorn_grad(grad_h_res_4d, sum_out_t, norm_out_t)
    grad_z_res = _exp_grad(z_res, sk_grad).flatten(2)

    grad_bias = torch.cat(
        [grad_z_pre.sum((0, 1)), grad_z_post.sum((0, 1)), grad_z_res.sum((0, 1))]
    )

    grad_alpha_pre = (norm_out_forward[..., :n] * grad_z_pre).sum()
    grad_alpha_post = (norm_out_forward[..., n : 2 * n] * grad_z_post).sum()
    grad_alpha_res = (norm_out_forward[..., 2 * n :] * grad_z_res).sum()
    grad_alpha = torch.tensor(
        [grad_alpha_pre, grad_alpha_post, grad_alpha_res], dtype=torch.float32
    )

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

    is_bf16 = isinstance(x, np.ndarray) and x.dtype == np_bfloat16
    if is_bf16:
        grad_x = grad_x.to(torch.bfloat16)
        grad_x_np = _torch_to_numpy(grad_x)
    else:
        grad_x_np = grad_x.numpy()
    grad_phi_np = grad_phi.numpy()
    grad_alpha_np = grad_alpha.numpy()
    grad_bias_np = grad_bias.numpy()
    return grad_x_np, grad_phi_np, grad_alpha_np, grad_bias_np


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

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors: gradX, gradPhi, gradAlpha, gradBias
    """

    if isinstance(x, torch.Tensor):
        x_dtype = x.dtype
    elif isinstance(x, np.ndarray) and x.dtype == np_bfloat16:
        x_dtype = torch.bfloat16
    elif isinstance(x, np.ndarray):
        x_dtype = None
    else:
        x_dtype = None

    grad_hin_np = (
        _torch_to_numpy(gradHin) if isinstance(gradHin, torch.Tensor) else gradHin
    )
    grad_h_post_np = (
        _torch_to_numpy(gradHPost) if isinstance(gradHPost, torch.Tensor) else gradHPost
    )
    grad_h_res_np = (
        _torch_to_numpy(gradHRes) if isinstance(gradHRes, torch.Tensor) else gradHRes
    )
    x_np = _torch_to_numpy(x) if isinstance(x, torch.Tensor) else x
    phi_np = _torch_to_numpy(phi) if isinstance(phi, torch.Tensor) else phi
    alpha_np = _torch_to_numpy(alpha) if isinstance(alpha, torch.Tensor) else alpha
    bias_np = _torch_to_numpy(bias) if isinstance(bias, torch.Tensor) else bias
    h_pre_np = _torch_to_numpy(hPre) if isinstance(hPre, torch.Tensor) else hPre
    hc_before_norm_np = (
        _torch_to_numpy(hcBeforeNorm)
        if isinstance(hcBeforeNorm, torch.Tensor)
        else hcBeforeNorm
    )
    inv_rms_np = _torch_to_numpy(invRms) if isinstance(invRms, torch.Tensor) else invRms
    sum_out_np = _torch_to_numpy(sumOut) if isinstance(sumOut, torch.Tensor) else sumOut
    norm_out_np = (
        _torch_to_numpy(normOut) if isinstance(normOut, torch.Tensor) else normOut
    )

    hc_eps_val = float(hcEps) if hcEps is not None else 1e-6

    grad_x_np, grad_phi_np, grad_alpha_np, grad_bias_np = (
        mhc_pre_sinkhorn_backward_golden(
            grad_hin_np,
            grad_h_post_np,
            grad_h_res_np,
            x_np,
            phi_np,
            alpha_np,
            bias_np,
            h_pre_np,
            hc_before_norm_np,
            inv_rms_np,
            sum_out_np,
            norm_out_np,
            hc_eps=hc_eps_val,
        )
    )

    grad_x_tensor = _numpy_to_torch(grad_x_np)
    grad_phi_tensor = _numpy_to_torch(grad_phi_np)
    grad_alpha_tensor = _numpy_to_torch(grad_alpha_np)
    grad_bias_tensor = _numpy_to_torch(grad_bias_np)

    if x_dtype == torch.bfloat16:
        grad_x_tensor = grad_x_tensor.to(torch.bfloat16)
    elif x_dtype == torch.float16:
        grad_x_tensor = grad_x_tensor.to(torch.float16)

    return grad_x_tensor, grad_phi_tensor, grad_alpha_tensor, grad_bias_tensor
