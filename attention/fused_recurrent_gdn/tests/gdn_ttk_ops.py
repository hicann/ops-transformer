# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""TTK precision adapter for the static TND variant of ARC-011."""

import torch
from assets.golden import recurrent_gdn
from assets.inputs import activation_kwargs


def customize_inputs(q, k, v, beta, g, initial_state, *, seed=101, **kwargs):
    generator = torch.Generator().manual_seed(seed)

    def rand(template):
        return torch.randn(template.shape, generator=generator)

    def normalize(template):
        x = (0.08 * rand(template)).bfloat16().double()
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    query, key, value = normalize(q), normalize(k), rand(v)
    gate = -torch.linspace(
        -1.2, -0.7, g.shape[-1]
    ).exp() * torch.nn.functional.softplus(rand(g))
    values = (
        query,
        key,
        value,
        rand(beta) if beta.dtype == torch.bfloat16 else rand(beta).sigmoid(),
        rand(g) if g.dtype == torch.bfloat16 else gate,
        0.02 * rand(initial_state),
    )
    return tuple(
        x.to(dtype=t.dtype, device=t.device)
        for x, t in zip(values, (q, k, v, beta, g, initial_state), strict=True)
    )


def fused_recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float = 1.0,
    seed: int = 101,
):
    import cann_ops_transformer_gdn_custom  # noqa: F401

    with torch.inference_mode():
        return torch.ops.cann_ops_transformer.fused_recurrent_gdn.default(
            q,
            k,
            v,
            beta,
            g,
            initial_state,
            scale=scale,
            output_final_state=True,
            **activation_kwargs(beta.dtype, g.dtype, v.shape[1], device=q.device),
        )


def golden(q, k, v, beta, g, initial_state, *, scale=1.0, seed=101):
    out, state = recurrent_gdn(
        q,
        k,
        v,
        beta,
        g,
        initial_state,
        scale,
        **activation_kwargs(beta.dtype, g.dtype, v.shape[1]),
    )
    return out.bfloat16(), state.to(initial_state.dtype)


class RecurrentGdnTestSpec:
    golden = staticmethod(golden)
    customize_inputs = staticmethod(customize_inputs)
    tolerance = {
        "bfloat16": {"standard": "isclose", "atol": 0.01, "rtol": 0.01, "ptol": 0.0},
        "float32": {"standard": "isclose", "atol": 1e-4, "rtol": 1e-3, "ptol": 0.0},
    }


__spec__ = {"gdn_ttk_ops.fused_recurrent_gdn": "RecurrentGdnTestSpec"}
