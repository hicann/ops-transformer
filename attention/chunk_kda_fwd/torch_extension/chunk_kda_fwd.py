# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _to_int_tuple(values, name):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional integer sequence") from exc


class _LayoutDimensions(NamedTuple):
    rank3: bool
    batch: int
    seqlen: int
    v_heads: int
    k_dim: int
    v_dim: int


class _ChunkShapeSpec(NamedTuple):
    chunk_size: int
    cu_seqlens: Optional[Sequence[int]]
    chunk_indices: Optional[Sequence[int]]


class _OutputShapes(NamedTuple):
    attn: Tuple[int, ...]
    matrix: Tuple[int, ...]
    k: Tuple[int, ...]
    v: Tuple[int, ...]
    state: Tuple[int, ...]
    h: Tuple[int, ...]


def _layout_dimensions(q, v, layout):
    rank3 = layout in {"TND", "NTD"}
    if layout == "TND":
        batch, seqlen, v_heads = 1, q.shape[0], v.shape[1]
    elif layout == "NTD":
        batch, seqlen, v_heads = 1, q.shape[1], v.shape[0]
    elif layout == "BNSD":
        batch, seqlen, v_heads = q.shape[0], q.shape[2], v.shape[1]
    else:
        batch, seqlen, v_heads = q.shape[0], q.shape[1], v.shape[2]
    k_dim, v_dim = q.shape[-1], v.shape[-1]
    return _LayoutDimensions(rank3, batch, seqlen, v_heads, k_dim, v_dim)


def _total_chunks(seqlen, shape_spec):
    if shape_spec.chunk_indices is not None:
        return len(shape_spec.chunk_indices) // 2
    if shape_spec.cu_seqlens is not None:
        return sum(
            (
                int(shape_spec.cu_seqlens[i + 1])
                - int(shape_spec.cu_seqlens[i])
                + shape_spec.chunk_size
                - 1
            )
            // shape_spec.chunk_size
            for i in range(len(shape_spec.cu_seqlens) - 1)
        )
    return (seqlen + shape_spec.chunk_size - 1) // shape_spec.chunk_size


def _shape_contract(q, v, layout, shape_spec, state_v_first):
    dims = _layout_dimensions(q, v, layout)
    seq_num = (
        len(shape_spec.cu_seqlens) - 1
        if shape_spec.cu_seqlens is not None
        else dims.batch
    )
    total_chunks = _total_chunks(dims.seqlen, shape_spec)
    attn_shape = (
        (dims.seqlen, dims.v_heads, dims.v_dim)
        if dims.rank3
        else (dims.batch, dims.seqlen, dims.v_heads, dims.v_dim)
    )
    matrix_shape = (
        (dims.v_heads, dims.seqlen, shape_spec.chunk_size)
        if dims.rank3
        else (
            dims.batch,
            dims.v_heads,
            dims.seqlen,
            shape_spec.chunk_size,
        )
    )
    k_shape = (
        (dims.v_heads, dims.seqlen, dims.k_dim)
        if dims.rank3
        else (dims.batch, dims.v_heads, dims.seqlen, dims.k_dim)
    )
    v_shape = (
        (dims.v_heads, dims.seqlen, dims.v_dim)
        if dims.rank3
        else (dims.batch, dims.v_heads, dims.seqlen, dims.v_dim)
    )
    state_tail = (dims.v_dim, dims.k_dim) if state_v_first else (dims.k_dim, dims.v_dim)
    state_shape = (seq_num, dims.v_heads, *state_tail)
    h_shape = (
        (total_chunks, dims.v_heads, *state_tail)
        if dims.rank3
        else (dims.batch, total_chunks, dims.v_heads, *state_tail)
    )
    return _OutputShapes(
        attn_shape, matrix_shape, k_shape, v_shape, state_shape, h_shape
    )


class ChunkKdaFwdOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("chunk_kda_fwd", category="attention")

    def sources(self):
        return ["csrc/attention/chunk_kda_fwd.cpp"]

    def schema(self) -> str:
        return (
            "chunk_kda_fwd("
            "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, *, "
            "Tensor? initial_state=None, "
            "bool output_final_state=False, "
            "int[]? cu_seqlens=None, "
            "int[]? chunk_indices=None, "
            "bool safe_gate=False, "
            "float lower_bound=-5.0, "
            "bool use_gate_in_kernel=False, "
            "Tensor? A_log=None, "
            "Tensor? dt_bias=None, "
            "int chunk_size=64, "
            "bool disable_recompute=False, "
            "bool return_intermediate_states=False, "
            "bool state_v_first=False, "
            'str layout="BSND"'
            ") -> "
            "(Tensor, Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor?, "
            "Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
        )

    def register_meta(self):
        # The dispatcher implementation must preserve the published operator schema.
        # pylint: disable=huawei-too-many-arguments,huawei-invalid-name,too-many-return-values
        @impl(get_as_library(), self.name, "Meta")
        def chunk_kda_fwd_meta(
            q,
            k,
            v,
            g,
            beta,
            scale,
            *,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            chunk_indices=None,
            safe_gate=False,
            lower_bound=-5.0,
            use_gate_in_kernel=False,
            A_log=None,
            dt_bias=None,
            chunk_size=64,
            disable_recompute=False,
            return_intermediate_states=False,
            state_v_first=False,
            layout="BSND",
        ):
            del k, g, beta, scale, safe_gate, lower_bound, A_log, dt_bias
            shapes = _shape_contract(
                q,
                v,
                layout,
                _ChunkShapeSpec(chunk_size, cu_seqlens, chunk_indices),
                state_v_first,
            )
            attn_shape, matrix_shape, k_shape, v_shape, state_shape, h_shape = shapes
            attn_out = torch.empty(attn_shape, dtype=v.dtype, device="meta")
            final_state = (
                torch.empty(state_shape, dtype=torch.float32, device="meta")
                if output_final_state
                else None
            )
            gk = (
                torch.empty(k_shape, dtype=torch.float32, device="meta")
                if not use_gate_in_kernel or disable_recompute
                else None
            )
            aqk = torch.empty(matrix_shape, dtype=q.dtype, device="meta")
            akk = torch.empty(matrix_shape, dtype=q.dtype, device="meta")
            w = (
                torch.empty(k_shape, dtype=q.dtype, device="meta")
                if disable_recompute
                else None
            )
            u = (
                torch.empty(v_shape, dtype=v.dtype, device="meta")
                if disable_recompute
                else None
            )
            qg = (
                torch.empty(k_shape, dtype=q.dtype, device="meta")
                if disable_recompute
                else None
            )
            kg = (
                torch.empty(k_shape, dtype=q.dtype, device="meta")
                if disable_recompute
                else None
            )
            v_new = (
                torch.empty(v_shape, dtype=v.dtype, device="meta")
                if disable_recompute
                else None
            )
            h = (
                torch.empty(h_shape, dtype=q.dtype, device="meta")
                if disable_recompute or return_intermediate_states
                else None
            )
            return (
                attn_out,
                final_state,
                gk,
                aqk,
                akk,
                w,
                u,
                qg,
                kg,
                v_new,
                h,
                initial_state,
            )

        # pylint: enable=huawei-too-many-arguments,huawei-invalid-name,too-many-return-values


_chunk_kda_fwd_op_builder = ChunkKdaFwdOpBuilder()
_chunk_kda_fwd_op_builder._ensure_initialized()


# The dispatcher implementation must preserve the published operator schema.
# pylint: disable=huawei-too-many-arguments,huawei-invalid-name
@impl(get_as_library(), _chunk_kda_fwd_op_builder.name, "PrivateUse1")
def _chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=-5.0,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    chunk_size=64,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    layout="BSND",
):
    module = _chunk_kda_fwd_op_builder.load()
    return module.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        chunk_size,
        disable_recompute,
        return_intermediate_states,
        state_v_first,
        layout,
    )


# pylint: enable=huawei-too-many-arguments,huawei-invalid-name


# pylint: disable=huawei-too-many-arguments,huawei-invalid-name
def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[Sequence[int]] = None,
    cu_seqlens_cpu: Optional[Sequence[int]] = None,
    chunk_indices: Optional[Sequence[int]] = None,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    layout: str = "BSND",
) -> Tuple[Optional[torch.Tensor], ...]:
    """Execute chunk KDA forward through ``aclnnChunkKdaFwd``."""
    host_cu_seqlens = _to_int_tuple(cu_seqlens_cpu, "cu_seqlens_cpu")
    if host_cu_seqlens is not None:
        if cu_seqlens is not None and not isinstance(cu_seqlens, torch.Tensor):
            device_cu_seqlens = _to_int_tuple(cu_seqlens, "cu_seqlens")
            if device_cu_seqlens != host_cu_seqlens:
                raise ValueError(
                    "cu_seqlens and cu_seqlens_cpu must contain identical values"
                )
        cu_seqlens = host_cu_seqlens
    else:
        cu_seqlens = _to_int_tuple(cu_seqlens, "cu_seqlens")
    chunk_indices = _to_int_tuple(chunk_indices, "chunk_indices")
    converted_gate_params = []
    if use_gate_in_kernel:
        if A_log is not None and A_log.dtype == torch.bfloat16:
            A_log = A_log.float()
            converted_gate_params.append(A_log)
        if dt_bias is not None and dt_bias.dtype == torch.bfloat16:
            dt_bias = dt_bias.float()
            converted_gate_params.append(dt_bias)
    outputs = torch.ops.cann_ops_transformer.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        lower_bound=-5.0 if lower_bound is None else float(lower_bound),
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
        state_v_first=state_v_first,
        layout=layout,
    )
    if converted_gate_params:
        current_stream = torch.npu.current_stream(q.device)
        for tensor in converted_gate_params:
            tensor.record_stream(current_stream)
    return outputs


# pylint: enable=huawei-too-many-arguments,huawei-invalid-name
