# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared pytest/TTK fixtures for the scalar-gate GDN contract."""

import torch

INPUT_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "metadata",
    "initial_state",
    "A_log",
    "dt_bias",
    "cu_seqlens",
    "seqused",
)
DEFAULTS = dict(
    scale=None,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    disable_recompute=True,
    layout="TND",
)


def effective(attributes):
    omitted = set(attributes.get("omitted_attributes", "").split(","))
    return {
        name: default if name in omitted else attributes.get(name, default)
        for name, default in DEFAULTS.items()
    }


def configuration(inputs, attributes):
    flags = effective(attributes)
    q, _, v = inputs[:3]
    layout = flags["layout"]
    if layout == "TND":
        storage, hq, dk = q.shape
        hv, dv = v.shape[1:]
        batch = None
    elif layout == "BSND":
        batch, storage, hq, dk = q.shape
        hv, dv = v.shape[2:]
    elif layout == "BNSD":
        batch, hq, storage, dk = q.shape
        hv, dv = v.shape[1], v.shape[-1]
    else:
        raise ValueError("layout must be TND, BSND, or BNSD")
    if dk != 128 or dv != 128 or hv < hq or hv % hq:
        raise ValueError("GDN requires Dk=Dv=128 and Nv a positive multiple of Nqk")
    cu_value = attributes.get("cu_seqlens_value")
    used_value = attributes.get("seqused_value")
    lengths = attributes.get("valid_lengths")
    if lengths is None:
        lengths = used_value
    if lengths is None and cu_value is not None:
        lengths = [b - a for a, b in zip(cu_value, cu_value[1:])]
    if lengths is None:
        lengths = [storage] * (batch or 1)
    lengths = [int(n) for n in lengths]
    if not lengths or any(n <= 0 or n > storage for n in lengths):
        raise ValueError("valid_lengths must contain positive lengths within storage")
    if batch is not None and len(lengths) != batch:
        raise ValueError("valid_lengths count must equal batch size")
    batch = len(lengths)
    if layout == "TND" and sum(lengths) != storage:
        raise ValueError("TND valid_lengths must cover every packed token")
    cu = [0] + torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()
    if cu_value is not None and list(cu_value) != cu:
        raise ValueError("cu_seqlens_value disagrees with valid_lengths")
    if used_value is not None and list(used_value) != lengths:
        raise ValueError("seqused_value disagrees with valid_lengths")
    return dict(
        flags=flags,
        layout=layout,
        batch=batch,
        storage=storage,
        hq=hq,
        hv=hv,
        dk=dk,
        dv=dv,
        lengths=lengths,
        cu=cu,
        has_cu=layout == "TND" or cu_value is not None,
        has_used=used_value is not None
        or (layout != "TND" and any(n != storage for n in lengths)),
    )


def to_layout(value, c):
    if c["layout"] == "TND":
        return torch.cat(
            [value[b, :n] for b, n in enumerate(c["lengths"])]
        ).contiguous()
    if c["layout"] == "BNSD":
        return value.transpose(1, 2).contiguous()
    return value.contiguous()


def sequences(value, c):
    if c["layout"] == "TND":
        return [value[a:b] for a, b in zip(c["cu"], c["cu"][1:])]
    if c["layout"] == "BNSD":
        value = value.transpose(1, 2)
    return [value[b, :n] for b, n in enumerate(c["lengths"])]


def make_metadata(c, *, host_fixture=False):
    from cann_ops_transformer import flash_gdn_metadata

    if host_fixture:
        # Explicit Host/Meta unit-test fixture; normal TTK generation always uses AICPU.
        from cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn_metadata import (
            make_header,
        )

        words = make_header(
            c["batch"], c["hq"], c["hv"], c["storage"], c["dk"], c["dv"], c["layout"]
        )
        return torch.tensor(words + c["lengths"], dtype=torch.int32, device="cpu")
    return flash_gdn_metadata(
        torch.tensor(c["cu"], dtype=torch.int32) if c["has_cu"] else None,
        torch.tensor(c["lengths"], dtype=torch.int32) if c["has_used"] else None,
        batch_size=c["batch"],
        nheads_qk=c["hq"],
        nheads_v=c["hv"],
        seqlens=c["storage"],
        headdim_qk=c["dk"],
        headdim_v=c["dv"],
        enable_grad=False,
        layout=c["layout"],
    )


def make(inputs, attributes, *, host_fixture=False):
    c = configuration(inputs, attributes)
    f = c["flags"]
    b, s, hq, hv = (
        c["batch"],
        max(c["lengths"]) if c["layout"] == "TND" else c["storage"],
        c["hq"],
        c["hv"],
    )
    gen = torch.Generator().manual_seed(attributes.get("seed", 101))
    amplitude = float(attributes.get("distribution", 1.0))

    def rand(shape):
        return torch.randn(shape, generator=gen) * amplitude

    q, k = [rand((b, s, hq, c["dk"])) * 0.08 for _ in range(2)]
    if not f["use_qk_l2norm_in_kernel"]:
        q, k = [
            x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)
        ]
    v = rand((b, s, hv, c["dv"]))
    g, beta = [rand((b, s, hv)) for _ in range(2)]
    a_log = torch.linspace(-1.2, -0.7, hv)
    bias_mode = attributes.get("dt_bias_mode", "none")
    if bias_mode not in ("none", "zero", "random"):
        raise ValueError("dt_bias_mode must be none, zero, or random")
    bias = rand((hv,)) * 0.1 if bias_mode == "random" else torch.zeros(hv)
    if not f["use_gate_in_kernel"]:
        g = -a_log.exp() * torch.nn.functional.softplus(g + bias)
    if not f["use_beta_sigmoid_in_kernel"]:
        beta = beta.sigmoid()
    state_mode = attributes.get("initial_state_mode", "none")
    if state_mode not in ("none", "omit", "zero", "random"):
        raise ValueError("initial_state_mode must be none, omit, zero, or random")
    state = (
        rand((b, hv, c["dv"], c["dk"])) * 0.02
        if state_mode == "random"
        else torch.zeros(b, hv, c["dv"], c["dk"])
    )
    values = [
        to_layout(x, c).to(template.dtype)
        for x, template in zip((q, k, v, g, beta), inputs[:5])
    ]
    return values + [
        make_metadata(c, host_fixture=host_fixture),
        state,
        a_log,
        bias,
        torch.tensor(c["cu"], dtype=torch.int32),
        torch.tensor(c["lengths"], dtype=torch.int32),
    ]


def customize_inputs(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    **kwargs,
):
    inputs = [
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
    ]
    generated = make(inputs, kwargs)
    for name, target, value in zip(INPUT_NAMES, inputs, generated):
        if target is None or target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(
                f"{name}: CSV shape/dtype must be {tuple(value.shape)}/{value.dtype}"
            )
        target.copy_(value.to(target.device))
    return tuple(inputs)


def call_kwargs(inputs, attributes):
    c = configuration(inputs, attributes)
    kwargs = dict(zip(INPUT_NAMES, inputs))
    # Real metadata is AICPU-generated on the same device as q; Meta keeps shape only.
    kwargs["metadata"] = kwargs["metadata"].to(inputs[0].device).contiguous()
    for name, present in (("cu_seqlens", c["has_cu"]), ("seqused", c["has_used"])):
        kwargs[name] = kwargs[name].cpu().contiguous() if present else None
    if attributes.get("initial_state_mode", "none") in ("none", "omit"):
        kwargs["initial_state"] = None
    if not c["flags"]["use_gate_in_kernel"]:
        kwargs["A_log"] = kwargs["dt_bias"] = None
    elif attributes.get("dt_bias_mode", "none") == "none":
        kwargs["dt_bias"] = None
    omitted = set(attributes.get("omitted_attributes", "").split(","))
    kwargs.update(
        {key: value for key, value in c["flags"].items() if key not in omitted}
    )
    if attributes.get("initial_state_mode") == "omit":
        kwargs.pop("initial_state")
    return kwargs
