# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch dispatcher bindings for the Flash KDA DSL entry points."""

from importlib import import_module
from pathlib import Path
import os

import torch

from cann_ops_transformer.op_builder import get_as_library


_library = get_as_library()
_forward_args = (
    "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor? metadata, "
    "Tensor? initial_state=None, Tensor? A_log=None, Tensor? dt_bias=None, "
    "Tensor? cu_seqlens=None, int[]? cu_seqlens_cpu=None, Tensor? seqused=None, *, "
    "float? scale=0.08838834764831845, bool output_final_state=False, "
    "bool return_intermediate_states=False, bool use_qk_l2norm_in_kernel=True, "
    "bool use_gate_in_kernel=True, bool use_beta_sigmoid_in_kernel=True, "
    'bool allow_neg_eigval=False, bool disable_recompute=False, float? lower_bound=None, str layout="TND"'
)
_library.define(f"flash_kda({_forward_args}) -> (Tensor, Tensor?, Tensor?)")
_forward_outputs = ", ".join(["Tensor"] + ["Tensor?"] * 15)
_library.define(f"flash_kda_fwd({_forward_args}) -> ({_forward_outputs})")
_library.define(
    "flash_kda_metadata(Tensor? cu_seqlens=None, Tensor? seqused=None, *, "
    "int batch_size, int nheads_qk, int nheads_v, int seqlens, int headdim_qk=128, "
    'int headdim_v=128, bool enable_grad=False, str layout="TND") -> Tensor'
)


_DSL_PREFIX = ".op_dsl" if (Path(__file__).parent / "op_dsl").is_dir() else "..op_dsl"


def _architecture(tensor=None):
    """Select the tensor's NPU, or the current NPU for scalar metadata calls."""
    requested = os.environ.get("FLASH_KDA_ARCH")
    if requested not in (None, "arch35", "arch92"):
        raise ValueError("FLASH_KDA_ARCH must be arch35 or arch92")
    device = getattr(tensor, "fake_device", getattr(tensor, "device", None))
    if device is not None and device.type not in ("npu", "privateuseone"):
        return requested or "arch35"
    if requested is not None and getattr(tensor, "fake_device", None) is not None:
        return requested
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return requested or "arch35"
    device = npu.current_device() if device is None else device
    name = npu.get_device_properties(device).name.upper()
    if name.startswith("ASCEND950"):
        selected = "arch35"
    elif name.startswith("ASCEND960"):
        selected = "arch92"
    else:
        raise ValueError(f"FlashKDA DSL does not support device {name}")
    if requested is not None and requested != selected:
        raise ValueError(f"FLASH_KDA_ARCH={requested} does not match device {name}")
    return selected


def _dsl_module(name, tensor=None):
    return import_module(f"{_DSL_PREFIX}.{_architecture(tensor)}.{name}", __package__)


def _grad_ops():
    namespace = torch.ops.cann_ops_transformer
    if not hasattr(namespace, "flash_kda_grad"):
        raise RuntimeError(
            "FlashKDA training requires the flash_kda_grad torch extension"
        )
    return namespace


def _split_metadata(v, metadata, cu_seqlens, layout):
    metadata_capacity_upper_bound = _dsl_module(
        "flash_kda_metadata", v
    ).metadata_capacity_upper_bound

    if metadata is None:
        raise ValueError("metadata is required")
    if layout == "TND":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for TND")
        batch, length, heads = cu_seqlens.numel() - 1, v.shape[0], v.shape[1]
    elif layout == "BNSD":
        batch, heads, length, _ = v.shape
    elif layout == "BSND":
        batch, length, heads, _ = v.shape
    else:
        raise ValueError(f"unsupported FlashKDA layout: {layout}")
    capacity = metadata_capacity_upper_bound(
        batch,
        length,
        is_packed=layout == "TND",
        value_heads=heads,
    )
    if (
        metadata.ndim != 1
        or metadata.dtype != torch.int32
        or not metadata.is_contiguous()
    ):
        raise ValueError("metadata must be a contiguous one-dimensional int32 tensor")
    if metadata.numel() == capacity:
        return metadata, None
    if metadata.numel() != capacity + 8 + batch * heads * 7:
        raise ValueError("metadata size does not match FlashKDA input dimensions")
    forward, backward = metadata.split((capacity, metadata.numel() - capacity))
    schedule = getattr(metadata, "_flash_kda_schedule", None)
    if schedule is not None and schedule["owner"] == id(metadata):
        forward._flash_kda_schedule = dict(schedule, owner=id(forward))
    return forward, backward


def _flash_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    **options,
):
    flash_kda_fwd = _dsl_module("flash_kda", q).flash_kda_fwd

    forward, _ = _split_metadata(v, metadata, cu_seqlens, options.get("layout", "TND"))
    return flash_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        forward,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        **options,
    )


def _flash_kda(*args, **kwargs):
    return _flash_kda_fwd(*args, **kwargs)[:3]


@torch.no_grad()
def _metadata(
    cu_seqlens=None,
    seqused=None,
    *,
    batch_size,
    nheads_qk,
    nheads_v,
    seqlens,
    headdim_qk=128,
    headdim_v=128,
    enable_grad=False,
    layout="TND",
):
    # The caller selects training independently of PyTorch grad mode.
    reference = cu_seqlens if cu_seqlens is not None else seqused
    metadata_module = _dsl_module("flash_kda_metadata", reference)

    sizes = dict(
        batch_size=batch_size,
        seqlens=seqlens,
        headdim_qk=headdim_qk,
        headdim_v=headdim_v,
        layout=layout,
    )
    forward = metadata_module.flash_kda_fwd_metadata(
        cu_seqlens,
        seqused,
        nheads_qk=nheads_qk,
        nheads_v=nheads_v,
        enable_grad=enable_grad,
        **sizes,
    )
    if not enable_grad:
        return forward
    if forward.is_meta:
        backward = forward.new_empty(8 + batch_size * nheads_v * 7)
    else:
        backward = _grad_ops().flash_kda_grad_metadata(
            cu_seqlens,
            seqused,
            n_heads_qk=nheads_qk,
            n_heads_v=nheads_v,
            **sizes,
        )
    metadata = torch.cat((forward, backward))
    schedule = getattr(forward, "_flash_kda_schedule", None)
    if schedule is not None:
        metadata._flash_kda_schedule = dict(
            schedule,
            owner=id(metadata),
            versions=metadata_module._schedule_versions(metadata, cu_seqlens, seqused),
        )
    return metadata


class _FlashKDAEntry(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        options,
    ):
        forward_impl = _dsl_module("flash_kda", q).flash_kda_fwd

        layout = options.get("layout", "TND")
        forward, backward = _split_metadata(v, metadata, cu_seqlens, layout)
        if backward is None:
            raise ValueError("training requires flash_kda_metadata(enable_grad=True)")
        outputs = forward_impl(
            q,
            k,
            v,
            g,
            beta,
            forward,
            initial_state,
            A_log,
            dt_bias,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            **options,
        )
        q_norm, q_rstd, k_norm, k_rstd = outputs[11:15]
        if options.get("use_qk_l2norm_in_kernel", True):
            if any(t is None for t in (q_norm, q_rstd, k_norm, k_rstd)):
                raise RuntimeError(
                    "FlashKDA autograd requires forward Q/K L2Norm caches"
                )
            q, k = q_norm, k_norm
        else:
            # With external normalization, q/k are already the operands
            # consumed by forward. The caller's graph owns the chain rule.
            q_rstd, k_rstd = None, None
        heads = v.shape[2 if layout == "BSND" else 1]
        if layout == "TND":
            matrix_shape = (heads, v.shape[0], 64)
        else:
            length = v.shape[2 if layout == "BNSD" else 1]
            matrix_shape = (v.shape[0], heads, length, 64)
        if any(t is None or t.shape != matrix_shape for t in outputs[3:5]):
            raise RuntimeError(
                "FlashKDA autograd requires forward Aqk/Akk in BNSC or NTC layout"
            )
        if options.get("disable_recompute", False):
            states = outputs[2]
            if states is None or states.dtype != torch.bfloat16:
                raise RuntimeError(
                    "saved-cache FlashKDA autograd requires BF16 chunk_states"
                )

        ctx.save_for_backward(
            v,
            g,
            beta,
            backward,
            initial_state,
            A_log,
            dt_bias,
            *outputs[2:11],
            q,
            q_rstd,
            k,
            k_rstd,
            outputs[15],
        )
        ctx.options = options
        ctx.batch = cu_seqlens.numel() - 1 if layout == "TND" else q.shape[0]
        ctx.set_materialize_grads(False)
        if outputs[2] is not None:
            ctx.mark_non_differentiable(outputs[2])
        return outputs[:3]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, dht, _):
        (
            v,
            g,
            beta,
            metadata,
            initial_state,
            A_log,
            dt_bias,
            states,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            gamma,
            q,
            q_rstd,
            k,
            k_rstd,
            beta_activated,
        ) = ctx.saved_tensors
        options = dict(ctx.options)
        options.pop("output_final_state", None)
        options.pop("return_intermediate_states", None)
        options.pop("cu_seqlens_cpu", None)
        layout = options.get("layout", "TND")
        heads = v.shape[2 if layout == "BSND" else 1]
        if do is None:
            do = torch.zeros_like(v.transpose(1, 2) if layout == "BNSD" else v)
        if dht is None:
            dht = v.new_zeros((ctx.batch, heads, 128, 128), dtype=torch.float32)
        g_dtype, beta_dtype = g.dtype, beta.dtype
        if not options.get("use_gate_in_kernel", True):
            g = g.float()
        if not options.get("use_beta_sigmoid_in_kernel", True):
            beta = beta.float()
        dq, dk, dv, dh0, dg, dbeta, da, dbias = _grad_ops().flash_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            do.contiguous(),
            dht.contiguous(),
            Aqk,
            Akk,
            metadata,
            initial_state=initial_state,
            chunk_states=states,
            w=w,
            u=u,
            qg=qg,
            kg=kg,
            v_new=v_new,
            g_cumsum=gamma,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            beta_sigmoid=beta_activated,
            A_log=A_log,
            dt_bias=dt_bias,
            **options,
        )
        gradients = (
            dq,
            dk,
            dv,
            dg.to(g_dtype),
            dbeta.to(beta_dtype),
            None,
            dh0,
            da,
            dbias,
            None,
            None,
            None,
        )
        return tuple(
            grad if needed else None
            for grad, needed in zip(gradients, ctx.needs_input_grad)
        )


def _flash_kda_autograd(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    **options,
):
    inputs = (
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
    )
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in inputs
    ):
        return _FlashKDAEntry.apply(
            *inputs, dict(options, cu_seqlens_cpu=cu_seqlens_cpu)
        )
    with torch._C._AutoDispatchBelowAutograd():
        return torch.ops.cann_ops_transformer.flash_kda.default(
            *inputs[:-1],
            cu_seqlens_cpu,
            seqused,
            **options,
        )


# The DSL entry points implement both eager execution and Meta output inference.
for _key in ("PrivateUse1", "Meta"):
    _library.impl("flash_kda", _flash_kda, _key)
    _library.impl("flash_kda_fwd", _flash_kda_fwd, _key)
    _library.impl("flash_kda_metadata", _metadata, _key)
# Dense layouts also support scalar-only metadata generation, with no tensor dispatch key.
_library.impl("flash_kda_metadata", _metadata, "CompositeExplicitAutograd")
_library.impl("flash_kda", _flash_kda_autograd, "Autograd")

flash_kda = torch.ops.cann_ops_transformer.flash_kda.default
flash_kda_fwd = torch.ops.cann_ops_transformer.flash_kda_fwd.default
flash_kda_metadata = torch.ops.cann_ops_transformer.flash_kda_metadata.default
