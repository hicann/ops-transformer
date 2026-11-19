# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic raw/activated GDN inputs without projection stages."""

import torch


def make_inputs(
    tokens,
    heads,
    beta_dtype,
    gate_dtype,
    *,
    seed=20261112,
    key_dim=128,
    value_dim=None,
    query_heads=1,
    batch=1,
    state_dtype=torch.float32,
    layout="TND",
):
    value_dim = key_dim if value_dim is None else value_dim
    random = torch.Generator().manual_seed(seed)
    total = batch * tokens
    q = (0.08 * torch.randn(total, query_heads, key_dim, generator=random)).bfloat16()
    k = (0.08 * torch.randn(total, query_heads, key_dim, generator=random)).bfloat16()
    v = (0.3 * torch.randn(total, heads, value_dim, generator=random)).bfloat16()
    raw_beta = torch.randn(total, heads, generator=random)
    raw_gate = torch.randn(total, heads, generator=random)
    beta = raw_beta.bfloat16() if beta_dtype == torch.bfloat16 else raw_beta.sigmoid()
    gate = (
        raw_gate.bfloat16()
        if gate_dtype == torch.bfloat16
        else -torch.nn.functional.softplus(raw_gate)
    )

    def arrange(x):
        if layout == "NTD":
            return x.transpose(0, 1).contiguous()
        if layout in ("BSND", "BNSD"):
            x = x.reshape(batch, tokens, *x.shape[1:])
            if layout == "BNSD":
                x = x.transpose(1, 2)
        return x.contiguous()

    state = (0.15 * torch.randn(batch, heads, key_dim, value_dim, generator=random)).to(
        state_dtype
    )
    return tuple(arrange(x) for x in (q, k, v, beta, gate)), state


def activation_kwargs(beta_dtype, gate_dtype, heads, *, device="cpu"):
    gate_raw = gate_dtype == torch.bfloat16
    return dict(
        use_beta_sigmoid_in_kernel=beta_dtype == torch.bfloat16,
        use_gate_in_kernel=gate_raw,
        A_log=torch.zeros(heads, dtype=torch.float32, device=device)
        if gate_raw
        else None,
    )
