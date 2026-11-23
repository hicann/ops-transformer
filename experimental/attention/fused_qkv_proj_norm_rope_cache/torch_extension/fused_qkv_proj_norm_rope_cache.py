# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library

from .op_dsl.fused_qkv_proj_norm_rope_cache import (
    AIC_NUM,
    HEAD_DIM,
    HIDDEN,
    NUM_Q_HEADS,
    NUM_UNITS,
    PAIR_M,
    ROPE_DIM,
    _get_compiled,
)


def _core_budget(device=None):
    """Return the launchable MIX group count for the current stream.

    A MIX block uses one AIC and two AIV, so the launchable group count is
    ``min(cube_num, vector_num // 2)``.  The live stream quota is queried on
    every call and enters both the launch count and the compile cache key.
    """
    from cannbotdsl.ops.info import get_platform_info

    stream = None
    try:
        stream = torch.npu.current_stream(device)
    except (AttributeError, RuntimeError):
        stream = None
    try:
        info = (
            get_platform_info(stream=stream)
            if stream is not None
            else get_platform_info()
        )
    except (RuntimeError, TypeError):
        info = get_platform_info()
    if (
        info.available
        and int(info.cube_core_num) > 0
        and int(info.vector_core_num) >= 2
    ):
        cube = int(info.cube_core_num)
        vector = int(info.vector_core_num)
        return min(cube, vector // 2)
    return AIC_NUM


def _is_stream_capturing(device):
    """True while the current NPU stream is being captured into an ACL Graph."""
    query = getattr(torch.npu, "is_current_stream_capturing", None)
    if query is None:
        return False
    try:
        with torch.npu.device(device):
            return bool(query())
    except (AttributeError, RuntimeError):
        return False


def _fused_qkv_proj_norm_rope_cache_impl(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    eps=1e-6,
    return_v=False,
    validate=True,
    positions_contiguous=None,
):
    """WeLMv4 QKV 投影 + 后处理全融合前向（对外公开入口）。

    Computes (per TP rank, kv-mirror layer):
        qkv(M,2560) = hidden(M,2048) @ weight(2560,2048)^T   (bf16 MMAD, fp32 acc)
        q   = rope(qkv[:, :1536])                              -> (M,1536) bf16
        k   = rope(rmsnorm(qkv[:, 1536:1792]))                 -> (M,256)  bf16
        v   = bf16(qkv[:, 1792:2048])                          -> v_cache[slot] only
        mk  = bf16(qkv[:, 2048:2304])                          -> (M,256)  bf16
        mv  = bf16(qkv[:, 2304:2560])                          -> (M,256)  bf16
        k_cache[slot] = k, v_cache[slot] = v                   (slot == -1 skipped)

    Args:
        hidden: (M, 2048) bf16 contiguous (QKV projection input)
        weight: (2560, 2048) bf16 contiguous, nn.Linear (N, K) layout
        k_gamma: (256,) bf16, K RMSNorm weight
        positions: (M,) int64, contiguous positions (validated when validate=True)
        cos_sin_cache: (max_pos, 64) fp32, first 32 cos / last 32 sin per row
        slot_mapping: (M,) int64 cache row indices, -1 skips the scatter
        k_cache / v_cache: paged KV cache, last dim 256, flattenable to (slots, 256)
        return_v: also materialize v (M, 256) for callers needing attn(q, k, v);
            the profiled NPU sink-prefill attention reads K/V from the cache,
            so the default (False) matches the production dataflow.
        validate: host-side checks (positions contiguity/range, slot range,
            cache device/shape consistency). Each device-value check is a
            small reduction + stream sync; disable on hot paths where the
            caller guarantees the invariants. Device-value checks are
            skipped automatically while the current stream is being captured
            into an ACL Graph (a sync there is illegal), so graph callers
            must keep positions/slot_mapping valid for both capture and
            replay.

    Returns:
        weight (2560, 2048) — kv-mirror source layer:
            return_v=False: (q, k, mirror_k, mirror_v) — v goes to cache only.
            return_v=True:  (q, k, v, mirror_k, mirror_v).
        weight (2048, 2048) — plain QKV layer (no mirror segments):
            return_v=False: (q, k).
            return_v=True:  (q, k, v).

    Note (scope): bf16 weights only. The production modelslim-mxfp8
    ``_kv_mirror_mxfp8_source_projection`` path (raw (N,K) weight + e8m0
    scales) is NOT covered by this operator.
    """
    assert hidden.dim() == 2 and hidden.shape[1] == HIDDEN
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    assert weight.dim() == 2 and weight.shape[1] == HIDDEN
    # Two profiled widths: 2560 = kv-mirror source layer (5 head-units),
    # 2048 = plain QKV layer (q/k/v only, welmv4.py:1845-1847 branch).
    n_units = weight.shape[0] // HEAD_DIM
    assert n_units * HEAD_DIM == weight.shape[0] and n_units in (8, 10), (
        f"weight must be (2048, {HIDDEN}) or ({QKV_WIDTH}, {HIDDEN}), "
        f"got {tuple(weight.shape)}"
    )
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert k_gamma.dim() == 1 and k_gamma.shape[0] == HEAD_DIM
    assert k_gamma.dtype == torch.bfloat16 and k_gamma.is_contiguous()
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == ROPE_DIM
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()

    num_tokens = hidden.shape[0]
    assert num_tokens > 0
    assert positions.shape[0] == num_tokens
    assert slot_mapping.shape[0] == num_tokens
    assert positions.device == hidden.device and slot_mapping.device == hidden.device

    if positions.dtype == torch.int32:
        positions = positions.to(torch.int64)
    assert positions.dtype == torch.int64
    if slot_mapping.dtype == torch.int32:
        slot_mapping = slot_mapping.to(torch.int64)
    assert slot_mapping.dtype == torch.int64

    def _flatten_cache(cache, name):
        assert cache.dtype == torch.bfloat16
        assert cache.shape[-1] == HEAD_DIM
        assert cache.is_contiguous()
        if cache.dim() > 2:
            cache = cache.view(-1, HEAD_DIM)
        assert cache.dim() == 2, f"{name} must be flattenable to (slots, 256)"
        return cache

    k_cache = _flatten_cache(k_cache, "k_cache")
    v_cache = _flatten_cache(v_cache, "v_cache")
    assert k_cache.device == hidden.device, "k_cache must live on hidden's device"
    assert v_cache.device == hidden.device, "v_cache must live on hidden's device"
    assert k_cache.shape == v_cache.shape, "k_cache/v_cache shape mismatch"

    # Every check below reads a device value back to the host, which
    # synchronizes the stream and is illegal inside ACL Graph capture.
    capturing = _is_stream_capturing(hidden.device)

    # Position mode: None auto-detects (one device reduction; decode
    # callers pass it explicitly to skip the sync). An explicit
    # positions_contiguous=True with validate=True is asserted, matching
    # the historical prefill-only contract. During capture, auto-detection
    # falls back to the decode path so the reduction is never emitted.
    if positions_contiguous is None:
        if capturing:
            positions_contiguous = False
        else:
            positions_contiguous = num_tokens == 1 or bool(
                (positions[1:] - positions[:-1] == 1).all()
            )
    elif validate and not capturing and positions_contiguous and num_tokens > 1:
        assert bool((positions[1:] - positions[:-1] == 1).all()), (
            "positions must be contiguous"
        )
    if validate and not capturing:
        pos_min, pos_max = torch.aminmax(positions)
        assert int(pos_min) >= 0, "negative position"
        assert int(pos_max) < cos_sin_cache.shape[0], (
            "position beyond cos_sin_cache rows"
        )
        slot_min, slot_max = torch.aminmax(slot_mapping)
        assert int(slot_min) >= -1, "slot must be -1 or non-negative"
        if int(slot_max) >= 0:
            assert int(slot_max) < k_cache.shape[0], "slot beyond cache rows"
            valid_slots = slot_mapping[slot_mapping != -1]
            assert valid_slots.numel() == valid_slots.unique().numel(), (
                "duplicate non-negative slots are not supported"
            )

    device = hidden.device
    q = torch.empty(
        (num_tokens, NUM_Q_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k = torch.empty((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
    if return_v:
        v = torch.empty((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
        # real destination; kernel writes it in the return_v branch
        gm_v = v
    else:
        # shape-compatible stand-in; the v-write branch is folded away at
        # trace time so this tensor is never written
        gm_v = k
    if n_units == 10:
        mirror_k = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        mirror_v = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    else:
        # Plain 8-unit layer has no mirror segments; the kernel never
        # touches gm_mk/gm_mv for u < 8, so 1-row stand-ins suffice.
        mirror_k = torch.zeros(1, HEAD_DIM, dtype=torch.bfloat16, device=device)
        mirror_v = mirror_k

    launch_groups = _core_budget(hidden.device)
    m_pairs = (num_tokens + PAIR_M - 1) // PAIR_M
    block_num = min(m_pairs * n_units, launch_groups)
    tensors = (
        hidden,
        weight,
        k_gamma.view(1, HEAD_DIM),
        positions,
        cos_sin_cache,
        slot_mapping,
        q,
        k,
        gm_v,
        mirror_k,
        mirror_v,
        k_cache,
        v_cache,
    )
    _get_compiled(
        num_tokens,
        n_units,
        cos_sin_cache.shape[0],
        k_cache.shape[0],
        return_v,
        positions_contiguous,
        block_num,
        eps,
    )(*tensors)
    if n_units == 10:
        if return_v:
            return q, k, v, mirror_k, mirror_v
        return q, k, mirror_k, mirror_v
    if return_v:
        return q, k, v
    return q, k


get_as_library().define(
    "fused_qkv_proj_norm_rope_cache("
    "Tensor hidden, Tensor weight, Tensor k_gamma, Tensor positions, "
    "Tensor cos_sin_cache, Tensor slot_mapping, Tensor(a!) k_cache, "
    "Tensor(b!) v_cache, *, float eps=1e-06, bool positions_contiguous=True"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


@impl(get_as_library(), "fused_qkv_proj_norm_rope_cache", "Meta")
def _fused_qkv_proj_norm_rope_cache_meta(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    *,
    eps=1e-6,
    positions_contiguous=True,
):
    del k_gamma, positions, cos_sin_cache, slot_mapping, k_cache, v_cache
    del eps, positions_contiguous
    tokens = hidden.shape[0]
    n_units = int(weight.shape[0]) // 256
    q = torch.empty((tokens, 6 * 256), dtype=torch.bfloat16, device="meta")
    k = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    v = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    if n_units == 10:
        mirror_k = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
        mirror_v = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    else:
        mirror_k = torch.empty((0, 256), dtype=torch.bfloat16, device="meta")
        mirror_v = torch.empty((0, 256), dtype=torch.bfloat16, device="meta")
    return q, k, v, mirror_k, mirror_v


@impl(get_as_library(), "fused_qkv_proj_norm_rope_cache", "PrivateUse1")
def fused_qkv_proj_norm_rope_cache(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    k_gamma: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    eps: float = 1e-6,
    positions_contiguous: bool = True,
):
    outputs = _fused_qkv_proj_norm_rope_cache_impl(
        hidden,
        weight,
        k_gamma,
        positions,
        cos_sin_cache,
        slot_mapping,
        k_cache,
        v_cache,
        eps=eps,
        return_v=True,
        positions_contiguous=positions_contiguous,
    )
    if len(outputs) == 5:
        return outputs
    q, k, v = outputs
    # Keep the plain-layer mirror outputs rank-2 so Meta and eager agree.
    empty = torch.empty((0, 256), dtype=q.dtype, device=q.device)
    return q, k, v, empty, empty.clone()
