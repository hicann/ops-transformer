# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Out-only Torch operator. Import and Meta/Fake do not import DSL or OpKit."""

import math
from numbers import Integral
from typing import NamedTuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

_OP = "fused_quant_qkv_proj_norm_rope_cache"
_NAMESPACE = "cann_ops_transformer"
_PAYLOAD_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SCALE_DTYPE = (torch.float8_e8m0fnu, torch.int8, torch.uint8)
_BF16 = torch.bfloat16
_F32 = torch.float32
_I64 = torch.int64
_SCHEMA = (
    _OP
    + "(Tensor hidden_quant, Tensor hidden_scale, Tensor weight_quant, Tensor weight_scale, "
    "Tensor k_gamma, Tensor positions, Tensor cos_sin, Tensor slot_mapping, "
    "Tensor(a!) q_out, Tensor(b!) k_out, Tensor(c!) v_out, "
    "Tensor(d!) k_cache, Tensor(e!) v_cache, *, float epsilon=1e-6, "
    "Tensor(f!)? mirror_k_out=None, Tensor(g!)? mirror_v_out=None, "
    "bool positions_contiguous=False, int? tokens_per_rank=None, "
    "int? chunk_tokens_per_rank=None, int? chunk_token_offset=None) -> ()"
)


def _chunk_config(
    input_rows, tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset
):
    values = (tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset)
    if all(value is None for value in values):
        return input_rows, False, (1, 1, 0)
    if any(value is None for value in values):
        raise ValueError("all three chunk parameters must be supplied together")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("chunk parameters must be integers")
    rank_tokens, chunk_rows, chunk_offset = values
    if (
        rank_tokens <= 0
        or chunk_rows <= 0
        or chunk_offset < 0
        or chunk_offset + chunk_rows > rank_tokens
    ):
        raise ValueError("chunk requires L>0, C>0, 0<=offset and offset+C<=L")
    if input_rows % chunk_rows != 0:
        raise ValueError("M must be divisible by chunk_tokens_per_rank")
    return (input_rows // chunk_rows) * rank_tokens, True, values


def _check_tensor(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    allowed = dtype if isinstance(dtype, (tuple, list)) else (dtype,)
    if tensor.dtype not in allowed:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}")
    for actual, expected in zip(tensor.shape, shape):
        torch._check(actual == expected, lambda: f"{name} shape must be {shape}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.requires_grad:
        raise ValueError(
            "this inference-only operator does not support requires_grad tensors"
        )


def _check_no_overlap(tensors):
    """All inputs are contiguous: byte intervals detect overlaps without synchronizing NPU."""
    intervals = []
    for name, tensor, writable in tensors:
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        intervals.append((name, begin, end, writable))
    for i, (name, begin, end, writable) in enumerate(intervals):
        for other, other_begin, other_end, other_writable in intervals[:i]:
            if (writable or other_writable) and max(begin, other_begin) < min(
                end, other_end
            ):
                raise ValueError(f"unsupported storage overlap: {name} and {other}")


class _Contract(NamedTuple):
    """Derived shape relations and specialization keys of one invocation."""

    device: torch.device
    input_rows: int
    hidden_size: int
    weight_rows: int
    q_width: int
    kv_width: int
    output_rows: int
    cos_sin_rows: int
    cache_rows: int
    has_mirror: bool
    has_chunk: bool
    chunk_config: tuple


def _derive_contract(
    hidden_quant,
    weight_quant,
    q_out,
    k_out,
    cos_sin,
    k_cache,
    *,
    epsilon,
    mirror_k_out,
    mirror_v_out,
    positions_contiguous,
    tokens_per_rank,
    chunk_tokens_per_rank,
    chunk_token_offset,
    require_npu,
):
    """Validate the scalar contract and derive every shape the spec table needs.

    The early isinstance/rank checks for the six tensors whose shapes feed
    derived relations keep failures clean TypeErrors before any shape
    dereference below.
    """
    if not isinstance(hidden_quant, torch.Tensor):
        raise TypeError("hidden_quant must be a Tensor")
    if hidden_quant.ndim != 2:
        raise ValueError("hidden_quant must have shape [M,K]")
    if not isinstance(weight_quant, torch.Tensor):
        raise TypeError("weight_quant must be a Tensor")
    if weight_quant.ndim != 2:
        raise ValueError("weight_quant must have shape [N,K]")
    device = hidden_quant.device
    if require_npu and device.type != "npu":
        raise ValueError("FusedQuantQkvProjNormRopeCache requires NPU tensors")
    input_rows, weight_rows = hidden_quant.shape[0], weight_quant.shape[0]
    hidden_size = hidden_quant.shape[1]
    if hidden_size <= 0 or hidden_size % 256:
        raise ValueError("K must be a positive multiple of 256")
    if not isinstance(q_out, torch.Tensor):
        raise TypeError("q_out must be a Tensor")
    if q_out.ndim != 2:
        raise ValueError("q_out must have shape [T,Hq*256]")
    if not isinstance(k_out, torch.Tensor):
        raise TypeError("k_out must be a Tensor")
    if k_out.ndim != 2:
        raise ValueError("k_out must have shape [T,Hkv*256]")
    q_width, kv_width = q_out.shape[1], k_out.shape[1]
    if q_width <= 0 or q_width % 256 or kv_width <= 0 or kv_width % 256:
        raise ValueError("Q and KV widths must be positive multiples of 256")
    if type(positions_contiguous) is not bool:
        raise TypeError("positions_contiguous must be bool")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    output_rows, has_chunk, chunk_config = _chunk_config(
        input_rows, tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset
    )
    has_mirror = mirror_k_out is not None
    if has_mirror != (mirror_v_out is not None):
        raise ValueError("mirror outputs must both be present or both absent")
    if weight_rows != q_width + (4 if has_mirror else 2) * kv_width:
        raise ValueError("weight N must match Q, KV and mirror output widths")
    if not isinstance(cos_sin, torch.Tensor):
        raise TypeError("cos_sin must be a Tensor")
    if cos_sin.ndim != 2:
        raise ValueError("cos_sin must have shape [P,64]")
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("k_cache must be a Tensor")
    if k_cache.ndim != 2:
        raise ValueError("k_cache must have shape [S,Hkv*256]")
    cos_sin_rows, cache_rows = cos_sin.shape[0], k_cache.shape[0]
    torch._check(cos_sin_rows > 0, lambda: "P must be positive")
    torch._check(cache_rows > 0, lambda: "S must be positive")
    return _Contract(
        device=device,
        input_rows=input_rows,
        hidden_size=hidden_size,
        weight_rows=weight_rows,
        q_width=q_width,
        kv_width=kv_width,
        output_rows=output_rows,
        cos_sin_rows=cos_sin_rows,
        cache_rows=cache_rows,
        has_mirror=has_mirror,
        has_chunk=has_chunk,
        chunk_config=chunk_config,
    )


def _check_specs(specs, device, check_overlap):
    for name, tensor, shape, dtype, _ in specs:
        _check_tensor(name, tensor, shape, dtype, device)
    if check_overlap:
        _check_no_overlap(
            [(name, tensor, writable) for name, tensor, _, _, writable in specs]
        )


def _validate_metadata(
    hidden_quant,
    hidden_scale,
    weight_quant,
    weight_scale,
    k_gamma,
    positions,
    cos_sin,
    slot_mapping,
    q_out,
    k_out,
    v_out,
    k_cache,
    v_cache,
    *,
    epsilon=1e-6,
    mirror_k_out=None,
    mirror_v_out=None,
    positions_contiguous=False,
    tokens_per_rank=None,
    chunk_tokens_per_rank=None,
    chunk_token_offset=None,
    require_npu=True,
    check_overlap=True,
):
    c = _derive_contract(
        hidden_quant,
        weight_quant,
        q_out,
        k_out,
        cos_sin,
        k_cache,
        epsilon=epsilon,
        mirror_k_out=mirror_k_out,
        mirror_v_out=mirror_v_out,
        positions_contiguous=positions_contiguous,
        tokens_per_rank=tokens_per_rank,
        chunk_tokens_per_rank=chunk_tokens_per_rank,
        chunk_token_offset=chunk_token_offset,
        require_npu=require_npu,
    )
    scale_cols = (c.hidden_size // 64, 2)
    specs = [
        (
            "hidden_quant",
            hidden_quant,
            (c.input_rows, c.hidden_size),
            _PAYLOAD_DTYPES,
            False,
        ),
        (
            "hidden_scale",
            hidden_scale,
            (c.input_rows, *scale_cols),
            _SCALE_DTYPE,
            False,
        ),
        (
            "weight_quant",
            weight_quant,
            (c.weight_rows, c.hidden_size),
            _PAYLOAD_DTYPES,
            False,
        ),
        (
            "weight_scale",
            weight_scale,
            (c.weight_rows, *scale_cols),
            _SCALE_DTYPE,
            False,
        ),
        ("k_gamma", k_gamma, (256,), _BF16, False),
        ("positions", positions, (c.output_rows,), _I64, False),
        ("cos_sin", cos_sin, (c.cos_sin_rows, 64), _F32, False),
        ("slot_mapping", slot_mapping, (c.output_rows,), _I64, False),
        ("q_out", q_out, (c.output_rows, c.q_width), _BF16, True),
        ("k_out", k_out, (c.output_rows, c.kv_width), _BF16, True),
        ("v_out", v_out, (c.output_rows, c.kv_width), _BF16, True),
        ("k_cache", k_cache, (c.cache_rows, c.kv_width), _BF16, True),
        ("v_cache", v_cache, (c.cache_rows, c.kv_width), _BF16, True),
    ]
    if c.has_mirror:
        specs.extend(
            [
                (
                    "mirror_k_out",
                    mirror_k_out,
                    (c.output_rows, c.kv_width),
                    _BF16,
                    True,
                ),
                (
                    "mirror_v_out",
                    mirror_v_out,
                    (c.output_rows, c.kv_width),
                    _BF16,
                    True,
                ),
            ]
        )
    _check_specs(specs, c.device, check_overlap)
    return c.has_mirror, c.has_chunk, c.chunk_config


def _meta(*args, **kwargs):
    _validate_metadata(*args, **kwargs, require_npu=False, check_overlap=False)
    return None


class FusedQuantQkvProjNormRopeCacheOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__(_OP, category="posembedding")

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        return _SCHEMA

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def _meta_impl(*args, **kwargs):
            return _meta(*args, **kwargs)


_op_builder = FusedQuantQkvProjNormRopeCacheOpBuilder()
_op_was_registered = hasattr(torch.ops.cann_ops_transformer, _OP)
_op_builder._ensure_initialized()


def _execute(*args, **kwargs):
    # Materialize the schema's keyword-only defaults once, so dispatcher calls
    # and direct calls (tests) share one uniform kwargs contract below.
    kwargs = {
        "epsilon": 1e-6,
        "mirror_k_out": None,
        "mirror_v_out": None,
        "positions_contiguous": False,
        "tokens_per_rank": None,
        "chunk_tokens_per_rank": None,
        "chunk_token_offset": None,
        **kwargs,
    }
    has_mirror, has_chunk, chunk_config = _validate_metadata(*args, **kwargs)
    (
        hidden_quant,
        hidden_scale,
        weight_quant,
        weight_scale,
        k_gamma,
        positions,
        cos_sin,
        slot_mapping,
        q_out,
        k_out,
        v_out,
        k_cache,
        v_cache,
    ) = args
    if hidden_quant.shape[0] == 0:
        return None
    # No frontend/DSL dependency is imported until an actual NPU invocation.
    from .op_dsl.fused_quant_qkv_proj_norm_rope_cache import (
        FP8_E4M3,
        FP8_E5M2,
        compile_fused_quant_qkv,
    )

    with torch.npu.device(hidden_quant.device):
        cube_core_count = int(
            torch.npu.get_device_properties(hidden_quant.device).cube_core_num
        )
        if cube_core_count <= 0:
            raise RuntimeError("cannot determine a positive cube core count")
        if has_mirror:
            mirror_k_out, mirror_v_out = kwargs["mirror_k_out"], kwargs["mirror_v_out"]
        else:
            # Never accessed by the plain-layer specialization; keep exact ABI shapes.
            mirror_k_out = torch.empty(
                (1, k_out.shape[1]), dtype=torch.bfloat16, device=hidden_quant.device
            )
            mirror_v_out = torch.empty_like(mirror_k_out)
        hidden_quant_dtype = (
            FP8_E5M2 if hidden_quant.dtype == torch.float8_e5m2 else FP8_E4M3
        )
        weight_quant_dtype = (
            FP8_E5M2 if weight_quant.dtype == torch.float8_e5m2 else FP8_E4M3
        )
        compiled_provider = compile_fused_quant_qkv(
            has_mirror,
            kwargs["positions_contiguous"],
            has_chunk,
            hidden_quant_dtype,
            weight_quant_dtype,
        )
        compiled_provider(
            hidden_quant,
            hidden_scale.view(torch.float8_e8m0fnu),
            weight_quant,
            weight_scale.view(torch.float8_e8m0fnu),
            k_gamma,
            positions,
            cos_sin,
            slot_mapping,
            q_out,
            k_out,
            v_out,
            mirror_k_out,
            mirror_v_out,
            k_cache,
            v_cache,
            float(kwargs["epsilon"]),
            *chunk_config,
            cube_core_count,
        )


if not _op_was_registered:
    _execute = impl(get_as_library(), _op_builder.name, "PrivateUse1")(_execute)


def fused_quant_qkv_proj_norm_rope_cache(
    hidden_quant,
    hidden_scale,
    weight_quant,
    weight_scale,
    k_gamma,
    positions,
    cos_sin,
    slot_mapping,
    q_out,
    k_out,
    v_out,
    k_cache,
    v_cache,
    *,
    epsilon=1e-6,
    mirror_k_out=None,
    mirror_v_out=None,
    positions_contiguous=False,
    tokens_per_rank=None,
    chunk_tokens_per_rank=None,
    chunk_token_offset=None,
):
    """Project MXFP8 QKV, normalize K, apply partial RoPE and update buffers.

    All tensors must be contiguous, on the same NPU, and have requires_grad=False.
    Each head has 256 features. Let QW=Hq*256 and KVW=Hkv*256, with Hq,Hkv>0.
    N=QW+2*KVW for a plain layer, or QW+4*KVW when mirrors are provided.

    Args:
        hidden_quant (Tensor): [M,K] FP8 E4M3FN or E5M2 activation payload.
            M>=0 and K is a positive multiple of 256.
        hidden_scale (Tensor): [M,K//64,2] E8M0 scales, one per 32 payload
            elements, packed as pairs along the last dimension. Accepts
            float8_e8m0fnu or int8/uint8 carriers containing E8M0 encodings.
        weight_quant (Tensor): [N,K] FP8 E4M3FN or E5M2 weight payload.
            Its dtype can differ from hidden_quant.
        weight_scale (Tensor): [N,K//64,2] E8M0 weight scales. Accepts
            float8_e8m0fnu or int8/uint8 carriers containing E8M0 encodings.
        k_gamma (Tensor): [256] BF16 RMSNorm weight shared by all K heads.
        positions (Tensor): [T] int64 position indices. Initialize every used
            row, including the first row of every processing tile and chunk.
        cos_sin (Tensor): [P,64] float32 RoPE table, P>0. Each row contains
            32 cosine values followed by 32 sine values.
        slot_mapping (Tensor): [T] int64 cache slots. Each used slot is -1
            (skip cache write) or in [0,S). Used nonnegative slots are unique.
        q_out (Tensor): [T,QW] BF16 preallocated input/output buffer.
        k_out (Tensor): [T,KVW] BF16 preallocated input/output buffer.
        v_out (Tensor): [T,KVW] BF16 preallocated input/output buffer.
        k_cache (Tensor): [S,KVW] BF16 cache updated in place, S>0.
        v_cache (Tensor): [S,KVW] BF16 cache updated in place.
        epsilon (float): Finite, strictly positive RMSNorm epsilon. Default 1e-6.
        mirror_k_out (Tensor, optional): [T,KVW] BF16 input/output buffer
            receiving raw mirror-K projection, without RMSNorm or RoPE.
        mirror_v_out (Tensor, optional): [T,KVW] BF16 input/output buffer
            receiving raw mirror-V projection. Both mirrors must be present
            together or both absent.
        positions_contiguous (bool): Whether positions of the processed local
            sequence advance by one. Each Q/K tile reads its first position and
            derives the remaining tile positions. Default False.
        tokens_per_rank (int, optional): Chunk output stride L>0.
        chunk_tokens_per_rank (int, optional): Local chunk size C>0, with M%C=0.
        chunk_token_offset (int, optional): Chunk offset a>=0, with a+C<=L.
            All three chunk arguments must be present together or all absent.

    Returns:
        None: Writes Q/K/V, the two caches and any mirror buffers in place.
        Without chunking, T=M. With chunking, T=(M//C)*L and local row i is
        written to g(i)=(i//C)*L+a+i%C. Other dense and mirror rows retain their
        input values; initialize those rows before calling if they will be read.
        Cache rows not selected by a valid slot also retain their input values.

    Raises:
        TypeError: A tensor dtype or scalar argument type is unsupported.
        ValueError: Shapes, devices, contiguity, writable storage overlap,
            epsilon or mirror/chunk argument combinations violate the contract.
        RuntimeError: Compilation or device execution fails.

    Notes:
        The caller must ensure 0<=positions[g(i)]<P, valid and unique cache
        slots, and the declared position continuity. Index values are not
        checked at runtime. Writable buffers must not overlap each other or
        any input. M=0 validates metadata and returns without writing buffers.
    """
    return torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache(
        hidden_quant,
        hidden_scale,
        weight_quant,
        weight_scale,
        k_gamma,
        positions,
        cos_sin,
        slot_mapping,
        q_out,
        k_out,
        v_out,
        k_cache,
        v_cache,
        epsilon=epsilon,
        mirror_k_out=mirror_k_out,
        mirror_v_out=mirror_v_out,
        positions_contiguous=positions_contiguous,
        tokens_per_rank=tokens_per_rank,
        chunk_tokens_per_rank=chunk_tokens_per_rank,
        chunk_token_offset=chunk_token_offset,
    )
