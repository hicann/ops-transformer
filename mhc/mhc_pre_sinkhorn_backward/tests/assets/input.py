# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch

__input__ = {
    "aclnn": {
        "aclnnMhcPreSinkhornBackward": "mhc_pre_sinkhorn_backward_input",
        "aclnnMhcPreSinkhornBackwardV2": "mhc_pre_sinkhorn_backward_v2_input",
    },
    "kernel": {"mhc_pre_sinkhorn_backward": "mhc_pre_sinkhorn_backward_input_kernel"},
}


def _sigmoid_grad(z, dy):
    sigma = torch.sigmoid(z)
    return dy * sigma * (1 - sigma)


def _rms_norm_grad(x, inv_rms, grad_inv_rms):
    nc = x.size(-1)
    return -(inv_rms**3) * x * grad_inv_rms / float(nc)


def _exp_grad(x, y_grad):
    nd = x.size(-1)
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(nd, device=x.device)[None, :]
    is_max = idx == arange_idx
    y = (x - x_max).exp()
    sum_all = (y * y_grad).sum(dim=-1, keepdim=True)
    return y * y_grad - is_max * sum_all


def _sinkhorn_grad(grad_h_res, sum_out, norm_out, hc_eps=1e-6):
    bs, seq_len, n, _ = grad_h_res.shape
    x_grad = grad_h_res
    it2, B_, S_, n_ = sum_out.shape
    iters = it2 // 2
    sum_out = sum_out.reshape(iters, 2, B_, S_, n_)
    norm_out = norm_out.reshape(iters, 2, B_, S_, n_, n_)
    for i in reversed(range(iters)):
        row_sum = sum_out[i][0].view(bs, seq_len, n, 1)
        col_sum = sum_out[i][1].view(bs, seq_len, 1, n)
        xrn = norm_out[i][0].view(bs, seq_len, n, n)
        gxrn = x_grad / col_sum - (x_grad * xrn / (col_sum**2)).sum(
            dim=-2, keepdim=True
        )
        if i == 0:
            row_sum = row_sum - hc_eps
            xrn = xrn - hc_eps
        x_grad = gxrn / row_sum - (gxrn * xrn / row_sum).sum(dim=-1, keepdim=True)
    return x_grad


def _copy_view(dst, src):
    """就地填充；broadcast 视图（stride 0，多元素共享同一存储）无法直接 copy_
    （torch 对重叠目标一律拒绝），沿 stride-0 维将 src 取常量后，改经覆盖物理
    元素的非重叠存储视图写入，保证写入合法且语义一致。"""
    st = dst.stride()
    if dst.numel() > 1 and 0 in st:
        sel = tuple(0 if s == 0 else slice(None) for s in st)
        core = src[sel]
        shape = tuple(sz for sz, s in zip(dst.shape, st) if s != 0)
        strides = tuple(s for s in st if s != 0)
        phys = dst.as_strided(shape, strides, dst.storage_offset())
        phys.copy_(core.reshape(shape))
        return
    dst.copy_(src)


def mhc_pre_sinkhorn_backward_input(
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
    grad_x_out,
    grad_phi_out,
    grad_alpha_out,
    grad_bias_out,
    **kwargs,
):
    is_3d = x.dim() == 3
    n = x.shape[-2]
    c = x.shape[-1]
    sinkhorn_iters = int(sum_out.shape[0] // 2)
    hc_eps_val = float(hc_eps)
    hm = n * n + 2 * n

    _copy_view(x, torch.randn_like(x))
    _copy_view(phi, torch.randn(phi.shape, dtype=torch.float32, device=x.device))
    _copy_view(alpha, torch.randn(3, dtype=torch.float32, device=x.device) * 0.1)
    _copy_view(bias, torch.randn(hm, dtype=torch.float32, device=x.device))

    x_fp32 = x.float()
    prod_nd = n * c
    x_flat = x_fp32.reshape(*x_fp32.shape[:-2], prod_nd)

    hcbn = torch.nn.functional.linear(x_flat, phi) + bias
    inv_rms_val = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + hc_eps_val)
    nf = hcbn * inv_rms_val

    z_pre = nf[..., :n] * alpha[0] + bias[:n]
    z_post = nf[..., n : 2 * n] * alpha[1] + bias[n : 2 * n]
    z_res = nf[..., 2 * n :] * alpha[2] + bias[2 * n :]

    h_pre_val = torch.sigmoid(z_pre) + hc_eps_val
    h_post = 2 * torch.sigmoid(z_post)

    comb = z_res.reshape(*z_res.shape[:-1], n, n)
    comb = torch.exp(comb - comb.max(dim=-1, keepdim=True).values)

    sum_out_list = []
    norm_out_list = []
    for it in range(sinkhorn_iters):
        row_sum = comb.sum(dim=-1, keepdim=True)
        if it == 0:
            comb = comb / row_sum + hc_eps_val
        else:
            comb = comb / (row_sum + hc_eps_val)
        norm_out_list.append(comb)
        sum_out_list.append(row_sum.squeeze(-1) + hc_eps_val)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + hc_eps_val)
        sum_out_list.append(col_sum.squeeze(-2) + hc_eps_val)
        norm_out_list.append(comb)

    sum_out_val = torch.stack(sum_out_list, dim=0)
    norm_out_val = torch.stack(norm_out_list, dim=0)
    hin = (h_pre_val.unsqueeze(-1) * x_fp32).sum(-2).to(x.dtype)

    _copy_view(h_pre, h_pre_val.detach())
    _copy_view(hc_before_norm, hcbn.detach())
    _copy_view(inv_rms, inv_rms_val.detach())
    _copy_view(sum_out, sum_out_val.detach())
    _copy_view(norm_out, norm_out_val.detach())

    _copy_view(grad_hin, torch.randn(grad_hin.shape, dtype=x.dtype, device=x.device))
    _copy_view(
        grad_h_post, torch.randn(h_pre_val.shape, dtype=torch.float32, device=x.device)
    )
    _copy_view(
        grad_h_res, torch.randn(grad_h_res.shape, dtype=torch.float32, device=x.device)
    )

    if is_3d:
        x4 = x_fp32.unsqueeze(0)
        gh4 = grad_hin.float().unsqueeze(0)
        gp4 = grad_h_post.unsqueeze(0)
        ghres4 = grad_h_res.unsqueeze(0)
        hp4 = h_pre_val.unsqueeze(0)
        hcbn4 = hcbn.unsqueeze(0)
        invr4 = inv_rms_val.unsqueeze(0)
        suma = sum_out_val.unsqueeze(1)
        norm4 = norm_out_val.unsqueeze(1)
    else:
        x4, gh4, gp4, ghres4 = x_fp32, grad_hin.float(), grad_h_post, grad_h_res
        hp4, hcbn4, invr4 = h_pre_val, hcbn, inv_rms_val
        suma, norm4 = sum_out_val, norm_out_val

    B4, S4, n4, d4 = x4.shape

    grad_x_hin = gh4[..., None, :] * hp4[..., None]
    g_hpre = (gh4[..., None, :] * x4).sum(-1)

    nf4 = hcbn4 * invr4
    zp4 = nf4[..., :n4] * alpha[0] + bias[:n4]
    zpost4 = nf4[..., n4 : 2 * n4] * alpha[1] + bias[n4 : 2 * n4]
    zres4 = (nf4[..., 2 * n4 :] * alpha[2] + bias[2 * n4 :]).reshape(B4, S4, n4, n4)

    gzp4 = _sigmoid_grad(zp4, g_hpre)
    gzpost4 = 2 * _sigmoid_grad(zpost4, gp4)
    # ghres4 may be (B,S,N*N) or (B,S,N,N); reshape to (B,S,N,N) for _sinkhorn_grad.
    ghres4_4d = ghres4.reshape(B4, S4, n4, n4) if ghres4.dim() == 3 else ghres4
    skg4 = _sinkhorn_grad(ghres4_4d, suma, norm4, hc_eps_val)
    gzres4 = _exp_grad(zres4, skg4).flatten(2)

    gold_bias = torch.cat([gzp4.sum((0, 1)), gzpost4.sum((0, 1)), gzres4.sum((0, 1))])
    ga_pre = (nf4[..., :n4] * gzp4).sum()
    ga_post = (nf4[..., n4 : 2 * n4] * gzpost4).sum()
    ga_res = (nf4[..., 2 * n4 :] * gzres4).sum()
    gold_alpha = torch.tensor(
        [ga_pre, ga_post, ga_res], dtype=torch.float32, device=x.device
    )

    gn4 = torch.cat([gzp4 * alpha[0], gzpost4 * alpha[1], gzres4 * alpha[2]], dim=-1)
    ghcbn4 = gn4 * invr4
    ginvr4 = (hcbn4 * gn4).sum(-1, keepdim=True)
    gx_rms4 = _rms_norm_grad(x4.flatten(2), invr4, ginvr4)
    gx_mm4 = ghcbn4 @ phi
    gold_phi = ghcbn4.flatten(0, 1).T @ x4.flatten(2).flatten(0, 1)
    gold_x = grad_x_hin + gx_rms4.view(x4.shape) + gx_mm4.view(x4.shape)

    if is_3d:
        gold_x = gold_x.view(x.shape[0], n4, d4)

    _copy_view(grad_x_out, gold_x.to(x.dtype))
    _copy_view(grad_phi_out, gold_phi)
    _copy_view(grad_alpha_out, gold_alpha)
    _copy_view(grad_bias_out, gold_bias)

    # 原地写法：张量已就地填充，返回 None（TTK 契约，输出槽不入返回值）。
    return None


def mhc_pre_sinkhorn_backward_v2_input(
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
    pre_mix,
    grad_pre,
    hc_eps,
    enable_hf32,
    grad_x_out,
    grad_phi_out,
    grad_alpha_out,
    grad_bias_out,
    grad_pre_mix_out,
    **kwargs,
):
    """Generate V2 inputs, reusing the V1 forward/intermediate setup.

    The V2 API keeps the same forward intermediates as V1 and only changes
    the fold source for ``grad_x`` plus the pre gate's incoming gradient.
    """
    del kwargs

    # Populate all mandatory tensors and forward intermediates exactly as the
    # original V1 generator does.  Its temporary outputs are overwritten below.
    mhc_pre_sinkhorn_backward_input(
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
        grad_x_out,
        grad_phi_out,
        grad_alpha_out,
        grad_bias_out,
    )

    n = x.shape[-2]
    c = x.shape[-1]
    is_3d = x.dim() == 3
    eps = float(hc_eps)

    if pre_mix is not None:
        _copy_view(
            pre_mix, torch.rand(pre_mix.shape, dtype=torch.float32, device=x.device)
        )
    if grad_pre is not None:
        _copy_view(
            grad_pre, torch.randn(grad_pre.shape, dtype=torch.float32, device=x.device)
        )

    x_fp32 = x.float()
    hcbn = hc_before_norm.float()
    invr = inv_rms.float()
    alpha_f = alpha.float()
    bias_f = bias.float()
    gh = grad_hin.float()
    gp = grad_h_post.float()
    if is_3d:
        x4 = x_fp32.unsqueeze(0)
        gh4 = gh.unsqueeze(0)
        gp4 = gp.unsqueeze(0)
        hcbn4 = hcbn.unsqueeze(0)
        invr4 = invr.unsqueeze(0)
        hpre4 = h_pre.float().unsqueeze(0)
        sum4 = sum_out.float().unsqueeze(1)
        norm4 = norm_out.float().unsqueeze(1)
        pm4 = pre_mix.unsqueeze(0) if pre_mix is not None else None
        gpr4 = grad_pre.unsqueeze(0) if grad_pre is not None else None
    else:
        x4, gh4, gp4 = x_fp32, gh, gp
        hcbn4, invr4, hpre4 = hcbn, invr, h_pre.float()
        sum4, norm4 = sum_out.float(), norm_out.float()
        pm4, gpr4 = pre_mix, grad_pre

    B, S, n4, d4 = x4.shape
    mix_weight = pm4 if pm4 is not None else hpre4
    grad_x_hin = gh4[..., None, :] * mix_weight[..., None]
    grad_h_pre = (gh4[..., None, :] * x4).sum(-1)
    if pm4 is None:
        d_pre = grad_h_pre if gpr4 is None else grad_h_pre + gpr4.float()
    else:
        d_pre = gpr4.float() if gpr4 is not None else torch.zeros_like(grad_h_pre)

    nf = hcbn4 * invr4
    z_pre = nf[..., :n4] * alpha_f[0] + bias_f[:n4]
    z_post = nf[..., n4 : 2 * n4] * alpha_f[1] + bias_f[n4 : 2 * n4]
    z_res = (nf[..., 2 * n4 :] * alpha_f[2] + bias_f[2 * n4 :]).reshape(B, S, n4, n4)
    grad_z_pre = _sigmoid_grad(z_pre, d_pre)
    grad_z_post = 2 * _sigmoid_grad(z_post, gp4)
    grad_h_res4 = grad_h_res.float().unsqueeze(0) if is_3d else grad_h_res.float()
    # grad_h_res 允许平铺 (..., N*N) 形式，_sinkhorn_grad 要求 (..., N, N)。
    grad_h_res4 = (
        grad_h_res4.reshape(*grad_h_res4.shape[:-1], n4, n4)
        if grad_h_res4.dim() == 3
        else grad_h_res4
    )
    sk_grad = _sinkhorn_grad(grad_h_res4, sum4, norm4, eps)
    grad_z_res = _exp_grad(z_res, sk_grad).flatten(2)

    grad_bias = torch.cat(
        [grad_z_pre.sum((0, 1)), grad_z_post.sum((0, 1)), grad_z_res.sum((0, 1))]
    )
    grad_alpha = torch.stack(
        [
            (nf[..., :n4] * grad_z_pre).sum(),
            (nf[..., n4 : 2 * n4] * grad_z_post).sum(),
            (nf[..., 2 * n4 :] * grad_z_res).sum(),
        ]
    )
    grad_norm = torch.cat(
        [grad_z_pre * alpha_f[0], grad_z_post * alpha_f[1], grad_z_res * alpha_f[2]],
        dim=-1,
    )
    grad_hc = grad_norm * invr4
    grad_inv = (hcbn4 * grad_norm).sum(-1, keepdim=True)
    grad_x_rms = _rms_norm_grad(x4.flatten(2), invr4, grad_inv)
    grad_x_mm = grad_hc @ phi.float()
    grad_phi = grad_hc.flatten(0, 1).T @ x4.flatten(2).flatten(0, 1)
    grad_x = grad_x_hin + grad_x_rms.view(x4.shape) + grad_x_mm.view(x4.shape)
    if is_3d:
        grad_x = grad_x.view(x.shape[0], n4, d4)

    _copy_view(grad_x_out, grad_x.to(grad_x_out.dtype))
    _copy_view(grad_phi_out, grad_phi.to(grad_phi_out.dtype))
    _copy_view(grad_alpha_out, grad_alpha.to(grad_alpha_out.dtype))
    _copy_view(grad_bias_out, grad_bias.to(grad_bias_out.dtype))
    if grad_pre_mix_out is not None:
        if pre_mix is None:
            grad_pre_mix_out.zero_()
        else:
            grad_pm = grad_h_pre.view(-1, n4) if is_3d else grad_h_pre
            _copy_view(grad_pre_mix_out, grad_pm.to(grad_pre_mix_out.dtype))

    # 原地写法：张量已就地填充，返回 None（TTK 契约，输出槽不入返回值）。
    return None


def mhc_pre_sinkhorn_backward_input_kernel(
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
    pre_mix=None,
    grad_pre=None,
    hc_eps=1e-6,
    enable_hf32=True,
    **kwargs,
):
    """Generate NumPy test inputs using forward-computed intermediates to avoid divergence in Sinkhorn backward."""
    del kwargs, enable_hf32
    rng = np.random.default_rng(0)
    eps = float(hc_eps)
    n = x.shape[-2]
    c = x.shape[-1]
    iters = sum_out.shape[0] // 2
    hm = n * n + 2 * n

    # Cast values to the input dtype before computing forward intermediates.
    x[...] = rng.standard_normal(x.shape).astype(x.dtype)
    phi[...] = rng.standard_normal(phi.shape).astype(phi.dtype)
    alpha[...] = (rng.standard_normal(3) * 0.1).astype(alpha.dtype)
    bias[...] = rng.standard_normal(hm).astype(bias.dtype)
    if pre_mix is not None:
        pre_mix[...] = rng.random(pre_mix.shape).astype(pre_mix.dtype)
    if grad_pre is not None:
        grad_pre[...] = rng.standard_normal(grad_pre.shape).astype(grad_pre.dtype)

    x_f32 = x.astype(np.float32)
    x_flat = x_f32.reshape(*x_f32.shape[:-2], n * c)
    hcbn = x_flat @ phi.astype(np.float32).T + bias.astype(np.float32)
    inv_rms_val = 1.0 / np.sqrt((x_flat**2).mean(axis=-1, keepdims=True) + eps)
    nf = hcbn * inv_rms_val

    z_pre = nf[..., :n] * alpha[0] + bias[:n]
    z_res = nf[..., 2 * n :] * alpha[2] + bias[2 * n :]
    h_pre_val = 1.0 / (1.0 + np.exp(-z_pre)) + eps

    comb = z_res.reshape(*z_res.shape[:-1], n, n)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))

    sum_out_list = []
    norm_out_list = []
    for it in range(iters):
        row_sum = comb.sum(axis=-1, keepdims=True)
        if it == 0:
            comb = comb / row_sum + eps
        else:
            comb = comb / (row_sum + eps)
        norm_out_list.append(comb)
        sum_out_list.append(row_sum.squeeze(-1) + eps)
        col_sum = comb.sum(axis=-2, keepdims=True)
        comb = comb / (col_sum + eps)
        sum_out_list.append(col_sum.squeeze(-2) + eps)
        norm_out_list.append(comb)

    h_pre[...] = h_pre_val.astype(h_pre.dtype)
    hc_before_norm[...] = hcbn.astype(hc_before_norm.dtype)
    inv_rms[...] = inv_rms_val.astype(inv_rms.dtype)
    sum_out[...] = np.stack(sum_out_list, axis=0).astype(sum_out.dtype)
    norm_out[...] = np.stack(norm_out_list, axis=0).astype(norm_out.dtype)

    grad_hin[...] = rng.standard_normal(grad_hin.shape).astype(grad_hin.dtype)
    grad_h_post[...] = rng.standard_normal(grad_h_post.shape).astype(grad_h_post.dtype)
    grad_h_res[...] = rng.standard_normal(grad_h_res.shape).astype(grad_h_res.dtype)

    return (
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
        pre_mix,
        grad_pre,
    )
