# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch contract for the fused recurrent GDN decode operator."""

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library

_SCHEMA = (
    "fused_recurrent_gdn(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor g, "
    "Tensor(a!)? initial_state=None, Tensor? A_log=None, Tensor? dt_bias=None, "
    "Tensor? cu_seqlens=None, Tensor? seqused=None, Tensor? ssm_state_indices=None, "
    "Tensor? num_accepted_tokens=None, *, float? scale=None, bool output_final_state=False, "
    "bool use_qk_l2norm_in_kernel=False, bool use_gate_in_kernel=False, "
    'bool use_beta_sigmoid_in_kernel=False, bool allow_neg_eigval=False, str layout="TND") '
    "-> (Tensor out, Tensor? final_state)"
)
_library = get_as_library()
_library.define(_SCHEMA)


def _check_tensor(name, tensor, device, dtypes, shape):
    if tensor.device != device or tensor.dtype not in dtypes:
        raise ValueError(f"Input {name} must be on {device} with dtype in {dtypes}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"Input {name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def _geometry(q, v, layout):
    if layout not in ("TND", "NTD", "BSND", "BNSD"):
        raise ValueError("Layout must be TND, NTD, BSND or BNSD")
    rank = 3 if layout in ("TND", "NTD") else 4
    if q.ndim != rank or v.ndim != rank:
        raise ValueError(f"Q and v must have rank {rank} for {layout}")
    head_axis = {"TND": 1, "NTD": 0, "BSND": 2, "BNSD": 1}[layout]
    token_axis = {"TND": 0, "NTD": 1, "BSND": 1, "BNSD": 2}[layout]
    nq, nv, dk, dv = q.shape[head_axis], v.shape[head_axis], q.shape[-1], v.shape[-1]
    if nq <= 0 or nv <= 0 or nv % nq:
        raise ValueError("Positive head counts are required, with Nv divisible by Nqk")
    if not 64 <= dk <= 512 or not 64 <= dv <= 512:
        raise ValueError("Dk and Dv must each be in the inclusive range [64, 512]")
    return nq, nv, dk, dv, head_axis, token_axis


def _validate(
    q,
    k,
    v,
    beta,
    g,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    output_final_state,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    layout,
):
    import math

    nq, nv, dk, dv, head_axis, token_axis = _geometry(q, v, layout)
    device = q.device
    value_shape = list(q.shape)
    value_shape[head_axis], value_shape[-1] = nv, dv
    _check_tensor("q", q, device, (torch.bfloat16,), q.shape)
    _check_tensor("k", k, device, (torch.bfloat16,), q.shape)
    _check_tensor("v", v, device, (torch.bfloat16,), value_shape)
    for name, tensor, raw in (
        ("beta", beta, use_beta_sigmoid_in_kernel),
        ("g", g, use_gate_in_kernel),
    ):
        expected = torch.bfloat16 if raw else torch.float32
        _check_tensor(name, tensor, device, (expected,), value_shape[:-1])
    if scale is not None and not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel=True")
    if not use_gate_in_kernel and (A_log is not None or dt_bias is not None):
        raise ValueError("A_log and dt_bias require use_gate_in_kernel=True")
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor is not None:
            _check_tensor(name, tensor, device, (torch.float32,), (nv,))
    for name, flag in (
        ("output_final_state", output_final_state),
        ("use_qk_l2norm_in_kernel", use_qk_l2norm_in_kernel),
        ("use_gate_in_kernel", use_gate_in_kernel),
        ("use_beta_sigmoid_in_kernel", use_beta_sigmoid_in_kernel),
        ("allow_neg_eigval", allow_neg_eigval),
    ):
        if not isinstance(flag, bool):
            raise ValueError(f"Attribute {name} must be bool")
    pool = ssm_state_indices is not None
    if pool and (initial_state is None or output_final_state):
        raise ValueError(
            "State pool requires initial_state and output_final_state=False"
        )
    if not pool and num_accepted_tokens is not None:
        raise ValueError("Input num_accepted_tokens requires ssm_state_indices")
    if pool:
        if ssm_state_indices.ndim not in (1, 2):
            raise ValueError("Input ssm_state_indices must have rank 1 or 2")
        if ssm_state_indices.ndim == 2 and num_accepted_tokens is None:
            raise ValueError("2D ssm_state_indices requires num_accepted_tokens")
    batches = []
    if layout in ("BSND", "BNSD"):
        batches.append(q.shape[0])
    for name, tensor, delta in (
        ("cu_seqlens", cu_seqlens, 1),
        ("seqused", seqused, 0),
        ("num_accepted_tokens", num_accepted_tokens, 0),
    ):
        if tensor is not None:
            if tensor.ndim != 1:
                raise ValueError(f"Input {name} must have rank 1")
            batches.append(tensor.shape[0] - delta)
    if pool:
        batches.append(ssm_state_indices.shape[0])
    if initial_state is not None:
        if initial_state.ndim != 4:
            raise ValueError("Input initial_state must have rank 4")
        if not pool:
            batches.append(initial_state.shape[0])
    batch = batches[0] if batches else 1
    packed = layout in ("TND", "NTD")
    if batch < 0 or any(size != batch for size in batches):
        raise ValueError(
            "Sequence descriptors and state disagree on logical batch size"
        )
    if batch == 0:
        if not packed:
            raise ValueError("Dense layouts require a positive logical batch size")
        if q.shape[token_axis] != 0:
            raise ValueError(
                "Nonempty packed input requires a positive logical batch size"
            )
    if layout in ("TND", "NTD") and batch != 1 and cu_seqlens is None:
        raise ValueError("Packed multi-sequence TND/NTD requires cu_seqlens")
    for name, tensor, length in (
        ("cu_seqlens", cu_seqlens, batch + 1),
        ("seqused", seqused, batch),
        ("num_accepted_tokens", num_accepted_tokens, batch),
    ):
        if tensor is not None:
            _check_tensor(name, tensor, device, (torch.int32,), (length,))
    if pool:
        _check_tensor(
            "ssm_state_indices",
            ssm_state_indices,
            device,
            (torch.int32,),
            ssm_state_indices.shape,
        )
        if ssm_state_indices.ndim == 2 and ssm_state_indices.shape[1] <= 0:
            raise ValueError("Input ssm_state_indices width must be positive")
    if initial_state is not None:
        rows = initial_state.shape[0] if pool else batch
        if pool and rows <= 0:
            raise ValueError("Input initial_state must have positive storage capacity")
        _check_tensor(
            "initial_state",
            initial_state,
            device,
            (torch.bfloat16, torch.float32),
            (rows, nv, dk, dv),
        )
    if pool and initial_state.device.type != "meta":
        if torch._debug_has_internal_overlap(initial_state) == 1:
            raise ValueError(
                "Input initial_state pool must not have overlapping elements"
            )
    return batch, nq, nv, dk, dv


def _infer_outputs(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    *,
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    layout="TND",
):
    batch, _, nv, dk, dv = _validate(
        q,
        k,
        v,
        beta,
        g,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        layout,
    )
    dtype = torch.float32 if initial_state is None else initial_state.dtype
    return ((batch, nv, dk, dv) if output_final_state else None), dtype


def _sequence_plan(
    q,
    batch,
    layout,
    cu_seqlens,
    seqused,
    initial_state,
    ssm_state_indices,
    num_accepted_tokens,
):
    # Preflight descriptor values before launching any mutating kernel.
    # Only small integer descriptors are copied to the host, never Q/K/V or state.
    _, _, _, _, _, token_axis = _geometry(q, q, layout)
    count = q.shape[token_axis]
    packed = layout in ("TND", "NTD")
    total = count if packed else batch * count
    if cu_seqlens is None:
        offsets = [0, total] if packed else [b * count for b in range(batch + 1)]
    else:
        offsets = cu_seqlens.detach().cpu().tolist()
        if (
            offsets[0] != 0
            or offsets[-1] != total
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError(
                "Input cu_seqlens must be nondecreasing, start at 0 and end at total tokens"
            )
        if not packed and offsets != [b * count for b in range(batch + 1)]:
            raise ValueError(
                "Input cu_seqlens must match the dense batch sequence boundaries"
            )
    lengths = [b - a for a, b in zip(offsets, offsets[1:])]
    if seqused is not None:
        used = seqused.detach().cpu().tolist()
        if any(n < 0 or n > capacity for n, capacity in zip(used, lengths)):
            raise ValueError("Input seqused must be between zero and sequence capacity")
        lengths = used
    spans = [[offsets[b], offsets[b] + lengths[b]] for b in range(batch)]
    reads = list(range(batch))
    writes = [0] * total
    if ssm_state_indices is not None:
        indices = ssm_state_indices.detach().cpu().tolist()
        accepted = (
            None
            if num_accepted_tokens is None
            else num_accepted_tokens.detach().cpu().tolist()
        )
        occupied = set()
        for b in range(batch):
            if ssm_state_indices.ndim == 1:
                if accepted is not None and accepted[b] != 1:
                    raise ValueError(
                        "1D state indices require num_accepted_tokens=1 when supplied"
                    )
                source = indices[b]
                destinations = [source] * lengths[b]
            else:
                row = indices[b]
                if not 1 <= accepted[b] <= len(row):
                    raise ValueError(
                        "Input num_accepted_tokens must be within [1, state index width]"
                    )
                if lengths[b] > len(row):
                    raise ValueError(
                        "Input ssm_state_indices width must cover every used token"
                    )
                source = row[accepted[b] - 1]
                destinations = row[: lengths[b]]
                if len(set(destinations)) != len(destinations):
                    raise ValueError(
                        "2D state output slots must be distinct within each sequence"
                    )
            slots = {source, *destinations}
            if any(slot < 0 or slot >= initial_state.shape[0] for slot in slots):
                raise ValueError("State index is outside initial_state pool capacity")
            if occupied.intersection(slots):
                raise ValueError(
                    "State slots from different sequences must not overlap"
                )
            occupied.update(slots)
            reads[b] = source
            writes[offsets[b] : offsets[b] + lengths[b]] = destinations
    return spans, reads, writes


@impl(_library, "fused_recurrent_gdn", "Meta")
def _meta(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    *,
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    layout="TND",
):
    shape, dtype = _infer_outputs(
        q,
        k,
        v,
        beta,
        g,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        ssm_state_indices,
        num_accepted_tokens,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
    )
    return torch.empty_like(v), None if shape is None else v.new_empty(
        shape, dtype=dtype
    )


@impl(_library, "fused_recurrent_gdn", "PrivateUse1")
def _npu(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    *,
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    layout="TND",
):
    from .op_dsl.fused_recurrent_gdn import fused_recurrent_gdn as execute

    return execute(
        q,
        k,
        v,
        beta,
        g,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        ssm_state_indices,
        num_accepted_tokens,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
    )


def fused_recurrent_gdn(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    *,
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    layout="TND",
):
    return torch.ops.cann_ops_transformer.fused_recurrent_gdn(
        q,
        k,
        v,
        beta,
        g,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        ssm_state_indices,
        num_accepted_tokens,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        layout=layout,
    )
