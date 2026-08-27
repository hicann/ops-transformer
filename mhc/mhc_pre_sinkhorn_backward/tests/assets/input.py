# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

__input__ = {
    "aclnn": {"aclnnMhcPreSinkhornBackward": "mhc_pre_sinkhorn_backward_input"},
}


def _sigmoid_grad(z, dy, is_pre=False, hc_eps=1e-6):
    y = torch.sigmoid(z)
    sigma = y - hc_eps if is_pre else y
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


def _sinkhorn_grad(grad_h_res, sum_out, norm_out):
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
        x_grad = gxrn / row_sum - (gxrn * xrn / row_sum).sum(dim=-1, keepdim=True)
    return x_grad


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

    x.copy_(torch.randn_like(x))
    phi.copy_(torch.randn(phi.shape, dtype=torch.float32, device=x.device))
    alpha.copy_(torch.randn(3, dtype=torch.float32, device=x.device) * 0.1)
    bias.copy_(torch.randn(hm, dtype=torch.float32, device=x.device))

    x_fp32 = x.float()
    prod_nd = n * c
    x_flat = x_fp32.reshape(*x_fp32.shape[:-2], prod_nd)

    hcbn = torch.nn.functional.linear(x_flat, phi) + bias
    inv_rms_val = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + hc_eps_val)
    nf = hcbn * inv_rms_val

    z_pre = nf[..., :n] * alpha[0] + bias[:n]
    z_post = nf[..., n : 2 * n] * alpha[1] + bias[n : 2 * n]
    z_res = nf[..., 2 * n :] * alpha[2] + bias[2 * n :]

    h_pre_val = torch.sigmoid(z_pre + hc_eps_val)
    h_post = 2 * torch.sigmoid(z_post)

    comb = z_res.reshape(*z_res.shape[:-1], n, n)
    comb = torch.exp(comb - comb.max(dim=-1, keepdim=True).values)

    sum_out_list = []
    norm_out_list = []
    for _ in range(sinkhorn_iters):
        row_sum = comb.sum(dim=-1, keepdim=True)
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

    h_pre.copy_(h_pre_val.detach())
    hc_before_norm.copy_(hcbn.detach())
    inv_rms.copy_(inv_rms_val.detach())
    sum_out.copy_(sum_out_val.detach())
    norm_out.copy_(norm_out_val.detach())

    grad_hin.copy_(torch.randn(grad_hin.shape, dtype=x.dtype, device=x.device))
    grad_h_post.copy_(
        torch.randn(h_pre_val.shape, dtype=torch.float32, device=x.device)
    )
    grad_h_res.copy_(
        torch.randn(grad_h_res.shape, dtype=torch.float32, device=x.device)
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

    gzp4 = _sigmoid_grad(zp4, g_hpre, is_pre=True, hc_eps=hc_eps_val)
    gzpost4 = 2 * _sigmoid_grad(zpost4, gp4, is_pre=False, hc_eps=hc_eps_val)
    # ghres4可能为(B,S,N*N)或(B,S,N,N), 统一reshape为(B,S,N,N)供_sinkhorn_grad处理
    ghres4_4d = ghres4.reshape(B4, S4, n4, n4) if ghres4.dim() == 3 else ghres4
    skg4 = _sinkhorn_grad(ghres4_4d, suma, norm4)
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

    grad_x_out.copy_(gold_x.to(x.dtype))
    grad_phi_out.copy_(gold_phi)
    grad_alpha_out.copy_(gold_alpha)
    grad_bias_out.copy_(gold_bias)

    return [
        grad_hin,
        grad_h_post,
        grad_h_res,
        x.detach(),
        phi.detach(),
        alpha.detach(),
        bias.detach(),
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
    ]
