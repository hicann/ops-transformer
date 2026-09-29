# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch


QUANT_MODE_MXFP4 = 1
GROUP_SIZE = 32
CANDIDATE_BLOCK_SIZE = 8
N1_STATIC = 32
N2_STATIC = 1
LOGICAL_D_STATIC = 128
GOLDEN_QUERY_CHUNK_MIN = 4
GOLDEN_QUERY_CHUNK_MAX = 256
GOLDEN_QK_BUDGET_BYTES = 128 * 1024 * 1024
GOLDEN_BATCH_WORKERS = 2

# MXFP4 E2M1 encoding, low nibble first.  The second half contains the sign.
_FP4_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


# ============================================================================
# 1. Independent CPU reference primitives
# ============================================================================


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def decode_e8m0(raw: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes (255 is reserved/NaN) to FP32."""
    raw_u8 = raw.to(torch.uint8)
    exponent = raw_u8.to(torch.int16).to(torch.float32) - 127.0
    result = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponent)
    return torch.where(raw_u8 == 255, torch.full_like(result, float("nan")), result)


def unpack_mxfp4(
    packed: torch.Tensor, logical_dim: Optional[int] = None
) -> torch.Tensor:
    """Unpack uint8 MXFP4 storage to an FP32 tensor.

    The public D dimension is even; each byte stores two E2M1 values.  A
    torch float4 tensor, when supplied by a caller, is first viewed as bytes.
    """
    data = packed.contiguous()
    if data.dtype != torch.uint8:
        data = data.view(torch.uint8)
    lo = data & 0x0F
    hi = data >> 4
    out = torch.stack((_FP4_VALUES[lo.long()], _FP4_VALUES[hi.long()]), dim=-1)
    out = out.reshape(*data.shape[:-1], data.shape[-1] * 2)
    if logical_dim is not None:
        out = out[..., : int(logical_dim)]
    return out


def pack_mxfp4(values: torch.Tensor) -> torch.Tensor:
    """Deterministically quantize FP32 values to packed MXFP4 bytes for tests."""
    values = values.to(torch.float32)
    levels = _FP4_VALUES.abs().unique(sorted=True)
    signs = values < 0
    distance = (values.abs().unsqueeze(-1) - levels).abs()
    code = distance.argmin(dim=-1).to(torch.uint8)
    code = code + signs.to(torch.uint8) * 8
    if values.shape[-1] % 2:
        raise ValueError("MXFP4 packing requires an even logical D")
    lo = code[..., 0::2]
    hi = code[..., 1::2]
    return lo | (hi << 4)


def _decode_scale(
    scale: Optional[torch.Tensor], shape: tuple[int, ...], d: int
) -> torch.Tensor:
    if scale is None:
        return torch.ones((*shape, d), dtype=torch.float32)
    raw = scale.contiguous().view(torch.uint8)
    groups = ceil_div(d, 32)
    if raw.numel() == math.prod(shape):
        scalar = decode_e8m0(raw.reshape(*shape))
        return scalar.unsqueeze(-1).expand(*shape, d)
    # Public MX scale is [..., ceil(D/64), 2], i.e. two E8M0 bytes per 64 D.
    raw = raw.reshape(*shape, -1)[..., :groups]
    return decode_e8m0(raw).repeat_interleave(32, dim=-1)[..., :d]


def _gt_eq_select(
    values: torch.Tensor,
    sort_keys: torch.Tensor,
    payload_indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU model of QLI ``FindIdxGT`` followed by ``FindIdxEQ``.

    QLI uses radix histograms only to find the kth key.  It does not emit a
    descending sequence: it first compacts every key greater than kth in
    input order, then appends equal keys in input order until ``k`` entries
    are consumed.  This helper intentionally derives only the threshold with
    ``argsort``; the observable output order follows the two compaction scans.
    """
    k = min(int(k), int(values.shape[-1]))
    flat_values = values.reshape(-1, values.shape[-1])
    flat_keys = sort_keys.reshape(-1, sort_keys.shape[-1])
    flat_payload = payload_indices.reshape(-1, payload_indices.shape[-1])
    threshold = torch.topk(
        flat_keys, k, dim=-1, largest=True, sorted=False
    ).values.amin(dim=-1, keepdim=True)
    positions = torch.arange(
        flat_keys.shape[-1], dtype=torch.int64, device=flat_keys.device
    ).reshape(1, -1)
    # The first priority range is FindIdxGT, the second is FindIdxEQ.  Within
    # each range the original position is the tie breaker, so the selected
    # payload order is identical to the two stable compaction scans.
    priority = torch.where(
        flat_keys > threshold,
        positions,
        torch.where(
            flat_keys == threshold,
            flat_keys.shape[-1] + positions,
            3 * flat_keys.shape[-1] + positions,
        ),
    )
    selected_positions = torch.topk(
        priority, k, dim=-1, largest=False, sorted=True
    ).indices
    out_shape = (*values.shape[:-1], k)
    return (
        torch.gather(flat_values, -1, selected_positions).reshape(out_shape),
        torch.gather(flat_payload, -1, selected_positions).reshape(out_shape),
        torch.gather(flat_keys, -1, selected_positions).reshape(out_shape),
    )


def _round_to_bf16_grid_fp32(x: torch.Tensor) -> torch.Tensor:
    """Round an FP32 carrier to the exact BF16 value grid without storing BF16.

    QLI casts QK and weights to BF16 before Vector1 and rounds the destination
    after every weighted G update.  The DSL path intentionally keeps these
    intermediates FP32, so the independent golden models only the observable
    BF16 rounding points with a bit-level round-to-nearest-even conversion.
    """
    bits = x.contiguous().view(torch.int32)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded & torch.tensor(-65536, dtype=torch.int32, device=x.device)).view(
        torch.float32
    )


def _round_to_bf16_grid_fp32_(x: torch.Tensor) -> torch.Tensor:
    """Fast RNE round-trip for disposable contiguous FP32 intermediates."""
    if not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("BF16-grid rounding requires contiguous FP32")
    return x.to(torch.bfloat16).to(torch.float32)


def _fp32_to_sortable_u16(x: torch.Tensor) -> torch.Tensor:
    """Match QLI FloatToSortableKey after BF16 rounding."""
    # Vector1 already rounded exported scores to the BF16 grid; Candidate is
    # a max over those same values, so repeating the conversion is idempotent.
    bits = x.contiguous().view(torch.int32)
    sign = bits >> 31
    sortable = torch.where(
        sign != 0,
        ~bits,
        bits ^ torch.tensor(-2147483648, dtype=torch.int32, device=x.device),
    )
    return (sortable >> 16).to(torch.uint16)


def _mxfp4_weighted_reduce(
    qk_relu: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Match QLI MXFP4 Vector1's BF16 QK/weight/FMA rounding points."""
    qk_bf16 = _round_to_bf16_grid_fp32_(qk_relu)
    weights_bf16 = _round_to_bf16_grid_fp32(weights.to(torch.float32))
    if qk_bf16.shape[1] != GROUP_SIZE or qk_bf16.shape[3] != 1:
        raise ValueError("QLI weighted reduction requires N1=G=32 and N2=1")
    acc = torch.zeros(
        (qk_bf16.shape[0], qk_bf16.shape[2]),
        dtype=torch.float32,
        device=qk_bf16.device,
    )
    for g in range(GROUP_SIZE):
        # QLI's MXFP4 ReduceSum path rounds the BF16 destination after every G.
        update = qk_bf16[:, g, :, 0] * weights_bf16[:, g : g + 1]
        acc = _round_to_bf16_grid_fp32_(acc + update)
    return acc.unsqueeze(1)


def _topk_trunk_len(k: int) -> int:
    k = int(k)
    if k <= 2048:
        return 16384
    if k <= 3072:
        return 12288
    if k <= 4096:
        return 8192
    if k <= 5120:
        return 4096
    if k <= 6144:
        return 2048
    return 12288


def _streaming_topk(
    values: torch.Tensor,
    k: int,
    sort_keys: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror QLI's local-trunk TopK and ping-pong history merge."""
    k = int(k)
    if k <= 0:
        shape = (*values.shape[:-1], 0)
        return (
            torch.empty(shape, dtype=values.dtype, device=values.device),
            torch.empty(shape, dtype=torch.int64, device=values.device),
        )
    s2 = int(values.shape[-1])
    trunk = _topk_trunk_len(k)
    if sort_keys is None:
        # Keep this public helper's historical generic behavior (stable FP32
        # sort).  The QLI path passes QLI's explicit BF16 sortable keys below;
        # candidate-block ranking is intentionally local and not part of QLI's
        # result_compare contract.
        sort_keys = values
    hist_values = None
    hist_indices = None
    hist_keys = None
    for begin in range(0, s2, trunk):
        end = min(begin + trunk, s2)
        local = values[..., begin:end]
        # CPU PyTorch does not implement gather for UInt16; QLI's sortable
        # key is an unsigned 16-bit carrier on device, but its non-negative
        # ordering is identical after widening to int32 for the reference.
        local_keys = sort_keys[..., begin:end]
        if local_keys.dtype == torch.uint16:
            local_keys = local_keys.to(torch.int32)
        base = torch.arange(begin, end, dtype=torch.int64, device=values.device)
        local_idx = base.reshape((1,) * (local.ndim - 1) + (end - begin,)).expand_as(
            local
        )
        if hist_values is None:
            hist_values, hist_indices, hist_keys = _gt_eq_select(
                local, local_keys, local_idx, min(k, end - begin)
            )
            continue
        merged_values = torch.cat((hist_values, local), dim=-1)
        merged_indices = torch.cat((hist_indices, local_idx), dim=-1)
        merged_keys = torch.cat((hist_keys, local_keys), dim=-1)
        hist_values, hist_indices, hist_keys = _gt_eq_select(
            merged_values,
            merged_keys,
            merged_indices,
            min(k, int(merged_keys.shape[-1])),
        )
    return hist_values, hist_indices


def _valid_key_count(
    full_s2: int,
    used_q: int,
    query_in_batch: int,
    mask_mode: int,
    cmp_ratio: int,
    cmp_residual: int,
) -> int:
    """Return the right-down visible K prefix for one packed TND row."""
    if int(mask_mode) != 3:
        return int(full_s2)
    original_k = int(full_s2) * int(cmp_ratio) + int(cmp_residual)
    visible = (original_k - int(used_q) + int(query_in_batch) + 1) // int(cmp_ratio)
    return max(0, min(int(full_s2), visible))


def _query_scores(
    query: torch.Tensor,
    query_scale: Optional[torch.Tensor],
    weights: torch.Tensor,
    logical_key: torch.Tensor,
    logical_d: int,
) -> torch.Tensor:
    """Compute one small query chunk without materializing an S1×S2 matrix."""
    query_fp32 = (
        unpack_mxfp4(query.contiguous(), logical_d)
        if query.dtype == torch.uint8
        else query.to(torch.float32)
    )
    rows, heads = query_fp32.shape[:2]
    scale = _decode_scale(query_scale, (rows, heads), logical_d).to(query_fp32.device)
    query_fp32 = query_fp32 * scale
    qk = (
        torch.mm(
            query_fp32.reshape(rows * heads, logical_d),
            logical_key[:, 0, :].transpose(0, 1),
        )
        .reshape(rows, heads, logical_key.shape[0], 1)
        .clamp_min(0.0)
    )
    return _mxfp4_weighted_reduce(qk, weights.to(torch.float32))


def _candidate_scores(values: torch.Tensor) -> torch.Tensor:
    """Reduce eight token scores into Candidate block scores."""
    valid_s2 = int(values.shape[-1])
    block_count = ceil_div(valid_s2, CANDIDATE_BLOCK_SIZE)
    padded_s2 = block_count * CANDIDATE_BLOCK_SIZE
    block_input = values
    if padded_s2 != valid_s2:
        block_input = torch.full(
            (*values.shape[:-1], padded_s2),
            float("-inf"),
            dtype=torch.float32,
            device=values.device,
        )
        block_input[..., :valid_s2] = values
    result = block_input.reshape(
        *values.shape[:-1], block_count, CANDIDATE_BLOCK_SIZE
    ).amax(dim=-1)
    if padded_s2 != valid_s2:
        result[..., -1] = float("inf")
    return result


class _ReferenceScoreProvider:
    """Lazily recompute one score row only for ResultCompare fallback."""

    def __init__(
        self,
        query,
        query_scale,
        weights,
        key_physical,
        page_ids,
        cu_seqlens,
        used_q,
        used_k,
        residual,
        logical_d,
        mask_mode,
        cmp_ratio,
    ):
        self.query = query
        self.query_scale = query_scale
        self.weights = weights
        self.key_physical = key_physical
        self.page_ids = page_ids
        self.cu_seqlens = cu_seqlens
        self.used_q = used_q
        self.used_k = used_k
        self.residual = residual
        self.logical_d = int(logical_d)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)

    def _batch_and_local_row(self, row: int) -> tuple[int, int]:
        for batch in range(len(self.used_q)):
            begin = self.cu_seqlens[batch]
            if begin <= row < self.cu_seqlens[batch + 1]:
                return batch, row - begin
        raise IndexError(f"query row {row} is outside cu_seqlens_q")

    def token_scores(self, row: int) -> torch.Tensor:
        batch, local = self._batch_and_local_row(int(row))
        if local >= self.used_q[batch]:
            return torch.empty((0,), dtype=torch.float32)
        ids = torch.tensor(self.page_ids[batch], dtype=torch.int64)
        logical_key = self.key_physical.index_select(0, ids).reshape(
            -1, N2_STATIC, self.logical_d
        )[: self.used_k[batch]]
        scale = None
        if self.query_scale is not None:
            scale = self.query_scale[row : row + 1]
        scores = _query_scores(
            self.query[row : row + 1],
            scale,
            self.weights[row : row + 1],
            logical_key,
            self.logical_d,
        )[0, 0]
        visible = _valid_key_count(
            self.used_k[batch],
            self.used_q[batch],
            local,
            self.mask_mode,
            self.cmp_ratio,
            self.residual[batch],
        )
        return scores[:visible]

    def candidate_scores(self, row: int) -> torch.Tensor:
        scores = self.token_scores(row)
        if scores.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)
        return _candidate_scores(scores.reshape(1, 1, -1))[0, 0]


class _CombinedReferenceScoreProvider:
    """Route ResultCompare fallback rows to independent per-batch providers."""

    def __init__(self, providers, row_offsets):
        self.providers = tuple(providers)
        self.row_offsets = tuple(int(value) for value in row_offsets)

    def _provider_and_row(self, row: int):
        row = int(row)
        for batch, begin in enumerate(self.row_offsets[:-1]):
            end = self.row_offsets[batch + 1]
            if begin <= row < end:
                return self.providers[batch], row - begin
        raise IndexError(f"query row {row} is outside combined batches")

    def token_scores(self, row: int) -> torch.Tensor:
        provider, local_row = self._provider_and_row(row)
        return provider.token_scores(local_row)

    def candidate_scores(self, row: int) -> torch.Tensor:
        provider, local_row = self._provider_and_row(row)
        return provider.candidate_scores(local_row)


# ============================================================================
# 2. Public-contract validation and TND/PA planning
# ============================================================================


def validate_quant_lightning_indexer_contract(
    query: torch.Tensor,
    key: torch.Tensor,
    query_dequant_scale: Optional[torch.Tensor],
    key_dequant_scale: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
) -> None:
    """Validate the public qli shape contract from ``算子接口.xlsx``.

    MXFP4 is physically packed two values per uint8 byte, so the logical D
    check multiplies a uint8 last dimension by two while preserving the
    public TND/PA-BBND ranks.
    """
    if query.ndim != 3 or key.ndim != 4:
        raise ValueError(
            "qli MX4 expects q=(T1,N1,D) and k=(block_num,block_size,N2,D)"
        )
    if int(query.shape[1]) != N1_STATIC or int(key.shape[2]) != N2_STATIC:
        raise ValueError("qli MX4 requires N1=32 and N2=1")
    q_logical_d = (
        int(query.shape[-1]) * 2 if query.dtype == torch.uint8 else int(query.shape[-1])
    )
    k_logical_d = (
        int(key.shape[-1]) * 2 if key.dtype == torch.uint8 else int(key.shape[-1])
    )
    if q_logical_d != LOGICAL_D_STATIC or k_logical_d != LOGICAL_D_STATIC:
        raise ValueError("qli MX4 requires logical D=128")
    if query_dequant_scale is not None:
        expected = (
            *query.shape[:2],
            ceil_div(LOGICAL_D_STATIC, 64),
            2,
        )
        if tuple(query_dequant_scale.shape) != expected:
            raise ValueError(
                "q_descale must follow the corrected xlsx shape (T1,N1,D/64,2)"
            )
    if key_dequant_scale is not None:
        expected = tuple(key.shape[:3]) + (LOGICAL_D_STATIC // 64, 2)
        if tuple(key_dequant_scale.shape) != expected:
            raise ValueError(
                "k_descale must follow the xlsx shape (block_num,block_size,N2,ceil(D/64),2)"
            )
    if layout_q != "TND":
        raise ValueError("layout_q must be TND under the current xlsx contract")
    if layout_k != "PA_BBND":
        raise ValueError("layout_k must be PA_BBND under the current xlsx contract")
    _cpu_page_plan(
        query,
        key,
        cu_seqlens_q,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )
    if metadata is not None:
        raise ValueError(
            "metadata is reserved by the current xlsx constraint and must be None"
        )


def _cpu_int_list(value: torch.Tensor, name: str) -> list[int]:
    if value.dtype != torch.int32:
        raise TypeError(f"{name} must be int32")
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _cpu_page_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cmp_residual_k: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    *,
    max_seqlen_q: int,
    mask_mode: int,
    cmp_ratio: int,
) -> tuple[list[int], list[int], list[int], list[list[int]], list[int]]:
    """CPU equivalent of the xlsx TND/PA argument semantics."""
    t = int(query.shape[0])
    physical_blocks = int(key.shape[0])
    if cu_seqlens_q is None:
        cu = [0, t]
    else:
        if cu_seqlens_q.ndim != 1 or int(cu_seqlens_q.numel()) < 2:
            raise ValueError("cu_seqlens_q must have shape (B+1,)")
        cu = _cpu_int_list(cu_seqlens_q, "cu_seqlens_q")
        if cu[0] != 0 or cu[-1] != t or any(a > b for a, b in zip(cu, cu[1:])):
            raise ValueError(
                "cu_seqlens_q must start at 0, be nondecreasing, and end at T1"
            )
    batch = len(cu) - 1
    q_lengths = [cu[b + 1] - cu[b] for b in range(batch)]
    if seqused_q is None:
        used = list(q_lengths)
    else:
        if tuple(seqused_q.shape) != (batch,):
            raise ValueError("seqused_q must have shape (B,)")
        used = _cpu_int_list(seqused_q, "seqused_q")
        if any(v < 0 or v > q_lengths[b] for b, v in enumerate(used)):
            raise ValueError(
                "seqused_q values must be within each cu_seqlens_q interval"
            )
    if int(max_seqlen_q) != -1 and (
        int(max_seqlen_q) < 0 or max(q_lengths, default=0) > int(max_seqlen_q)
    ):
        raise ValueError("max_seqlen_q must be -1 or cover every query sequence")

    if block_table is None:
        page_ids = [list(range(physical_blocks)) for _ in range(batch)]
    else:
        if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
            raise ValueError("block_table must have shape (B,max_num_blocks_per_seq)")
        flat = _cpu_int_list(block_table, "block_table")
        width = int(block_table.shape[1])
        page_ids = []
        for b in range(batch):
            row = flat[b * width : (b + 1) * width]
            valid = []
            saw_padding = False
            for page in row:
                if page == -1:
                    saw_padding = True
                    continue
                if saw_padding:
                    raise ValueError(
                        "block_table valid page ids must precede -1 padding"
                    )
                if page < 0 or page >= physical_blocks:
                    raise ValueError(
                        "block_table contains an invalid physical block id"
                    )
                valid.append(page)
            if not valid:
                raise ValueError(
                    "every batch must reference at least one physical K block"
                )
            page_ids.append(valid)

    capacities = [len(ids) * int(key.shape[1]) for ids in page_ids]
    if seqused_k is None:
        used_k = capacities
    else:
        if tuple(seqused_k.shape) != (batch,):
            raise ValueError("seqused_k must have shape (B,)")
        used_k = _cpu_int_list(seqused_k, "seqused_k")
        if any(v < 0 or v > capacities[b] for b, v in enumerate(used_k)):
            raise ValueError("seqused_k values must be within each batch PA capacity")

    if int(mask_mode) not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("cmp_ratio must be in [1,128]")
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None or tuple(cmp_residual_k.shape) != (batch,):
            raise ValueError(
                "cmp_residual_k with shape (B,) is required for mask_mode=3 and cmp_ratio!=1"
            )
        residual = _cpu_int_list(cmp_residual_k, "cmp_residual_k")
        if any(v < 0 or v >= int(cmp_ratio) for v in residual):
            raise ValueError("cmp_residual_k values must be in [0,cmp_ratio)")
    else:
        if cmp_residual_k is not None:
            raise ValueError(
                "cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
            )
        residual = [0] * batch
    return cu, used, used_k, page_ids, residual


# ============================================================================
# 3. Independent CPU Golden
# ============================================================================


def quant_lightning_indexer_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_k=None,
    output_idx_offset=None,
    topk: int,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
    quant_mode: int = QUANT_MODE_MXFP4,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
    return_value: bool = False,
) -> dict[str, torch.Tensor]:
    """Independent CPU golden for xlsx TND query plus PA_BBND key.

    Physical K/K-scale pages are gathered through ``block_table`` for every
    batch; TND rows are assigned through ``cu_seqlens_q``.  Consequently this
    reference does not rely on a shared flattened-K shortcut.
    """
    if output_idx_offset is not None:
        parameters = locals().copy()
        parameters["output_idx_offset"] = None
        result = quant_lightning_indexer_golden(**parameters)
        indices = result["sparse_indices"]
        result["sparse_indices"] = torch.where(
            indices >= 0, indices + output_idx_offset.unsqueeze(-1), indices
        )
        result["_output_idx_offset"] = output_idx_offset
        return result
    query, key, weights = q, k, w
    query_dequant_scale, key_dequant_scale = q_descale, k_descale
    if int(quant_mode) != QUANT_MODE_MXFP4:
        raise ValueError("DSL QLI MX4 requires quant_mode=1")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError(
                "candidate_block_size must be -1 when candidate is disabled"
            )
    elif candidate_topk_blocks > 0 and candidate_block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("candidate_block_size must be 8 when candidate is enabled")
    elif candidate_topk_blocks <= 0:
        raise ValueError("candidate_topk_blocks must be -1 or positive")
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("topk must be in QLI-supported range [1,8192]")
    if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != torch.float32:
        raise ValueError("w must be FP32 with shape (T1,N1)")

    if layout_k == "TND":
        if tuple(query.shape[1:]) != (32, 64) or tuple(key.shape[1:]) != (1, 64):
            raise ValueError("TND golden requires N1=32, N2=1, logical D=128")
        cu = [0, len(query)] if cu_seqlens_q is None else cu_seqlens_q.tolist()
        bounds = [0, len(key)] if cu_seqlens_k is None else cu_seqlens_k.tolist()
        if (
            len(cu) != len(bounds)
            or cu[0] != 0
            or bounds[0] != 0
            or cu[-1] != len(query)
            or bounds[-1] != len(key)
        ):
            raise ValueError("invalid TND boundaries")
        used = (
            [end - begin for begin, end in zip(cu, cu[1:])]
            if seqused_q is None
            else seqused_q.tolist()
        )
        used_k = (
            [end - begin for begin, end in zip(bounds, bounds[1:])]
            if seqused_k is None
            else seqused_k.tolist()
        )
        residual = (
            [0] * len(used) if cmp_residual_k is None else cmp_residual_k.tolist()
        )
        page_ids = [range(begin, begin + count) for begin, count in zip(bounds, used_k)]
    else:
        validate_quant_lightning_indexer_contract(
            query,
            key,
            query_dequant_scale,
            key_dequant_scale,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            metadata,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            layout_k=layout_k,
        )
        cu, used, used_k, page_ids, residual = _cpu_page_plan(
            query,
            key,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )

    if len(used) > 1 and layout_k != "TND":
        total_rows = int(query.shape[0])
        candidate_width = max(0, int(candidate_topk_blocks))
        combined = {
            "sparse_indices": torch.empty(
                (total_rows, N2_STATIC, int(topk)), dtype=torch.int32
            ),
            "sparse_values": torch.empty(
                (total_rows, N2_STATIC, int(topk)), dtype=torch.bfloat16
            ),
            "candidate_block_indices": torch.empty(
                (total_rows, N2_STATIC, candidate_width), dtype=torch.int32
            ),
            "candidate_block_length": torch.empty(
                (total_rows, N2_STATIC), dtype=torch.int32
            ),
        }
        providers = [None] * len(used)

        def compute_batch(batch: int):
            row_begin = cu[batch]
            row_end = cu[batch + 1]
            batch_table = torch.tensor(page_ids[batch], dtype=torch.int32).reshape(
                1, -1
            )
            batch_residual = None
            if int(mask_mode) == 3 and int(cmp_ratio) != 1:
                batch_residual = torch.tensor([residual[batch]], dtype=torch.int32)
            result = quant_lightning_indexer_golden(
                query[row_begin:row_end],
                key,
                weights[row_begin:row_end],
                None
                if query_dequant_scale is None
                else query_dequant_scale[row_begin:row_end],
                key_dequant_scale,
                cu_seqlens_q=torch.tensor([0, row_end - row_begin], dtype=torch.int32),
                seqused_q=torch.tensor([used[batch]], dtype=torch.int32),
                seqused_k=torch.tensor([used_k[batch]], dtype=torch.int32),
                cmp_residual_k=batch_residual,
                block_table=batch_table,
                metadata=metadata,
                topk=topk,
                candidate_topk_blocks=candidate_topk_blocks,
                candidate_block_size=candidate_block_size,
                quant_mode=quant_mode,
                max_seqlen_q=max_seqlen_q,
                mask_mode=mask_mode,
                cmp_ratio=cmp_ratio,
                layout_q=layout_q,
                layout_k=layout_k,
                return_value=return_value,
            )
            return batch, row_begin, row_end, result

        worker_count = min(GOLDEN_BATCH_WORKERS, len(used))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, previous_threads // worker_count))
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for batch, row_begin, row_end, result in executor.map(
                    compute_batch, range(len(used))
                ):
                    for name in combined:
                        combined[name][row_begin:row_end].copy_(result[name])
                    providers[batch] = result["_reference_scores"]
        finally:
            torch.set_num_threads(previous_threads)
        reference_provider = _CombinedReferenceScoreProvider(providers, cu)
        combined["_reference_scores"] = reference_provider
        combined["_candidate_block_scores"] = reference_provider
        combined["_return_value"] = bool(return_value)
        return combined

    qd = (
        int(query.shape[-1]) * 2 if query.dtype == torch.uint8 else int(query.shape[-1])
    )
    kd = int(key.shape[-1]) * 2 if key.dtype == torch.uint8 else int(key.shape[-1])
    if qd != kd:
        raise ValueError("query/key logical D must match")
    k_phys = (
        unpack_mxfp4(key.contiguous(), kd)
        if key.dtype == torch.uint8
        else key.to(torch.float32)
    )
    t, n1 = query.shape[:2]
    d = qd
    if key_dequant_scale is None:
        k_scale_phys = torch.ones(
            (*key.shape[:3], d), dtype=torch.float32, device=k_phys.device
        )
    else:
        k_scale_phys = _decode_scale(
            key_dequant_scale.contiguous().view(torch.uint8),
            tuple(key.shape[:2] if layout_k == "TND" else key.shape[:3]),
            d,
        ).to(k_phys.device)
    k_phys = k_phys * k_scale_phys
    w = weights.to(torch.float32)

    sparse_indices = torch.full(
        (t, N2_STATIC, int(topk)), -1, dtype=torch.int32, device=query.device
    )
    sparse_value_fp32 = torch.full(
        (t, N2_STATIC, int(topk)),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    if candidate_topk_blocks > 0:
        c = int(candidate_topk_blocks)
        candidate_indices = torch.full(
            (t, N2_STATIC, c), -1, dtype=torch.int32, device=query.device
        )
        candidate_length = torch.zeros(
            (t, N2_STATIC), dtype=torch.int32, device=query.device
        )
    else:
        candidate_indices = torch.empty(
            (t, N2_STATIC, 0), dtype=torch.int32, device=query.device
        )
        candidate_length = torch.zeros(
            (t, N2_STATIC), dtype=torch.int32, device=query.device
        )

    for b, ids in enumerate(page_ids):
        full_s2 = used_k[b]
        logical_k = k_phys[torch.tensor(ids, dtype=torch.int64)].reshape(
            -1, N2_STATIC, d
        )[:full_s2]
        used_q = used[b]
        if used_q == 0 or full_s2 == 0:
            continue
        q_begin = cu[b]
        query_chunk = max(
            GOLDEN_QUERY_CHUNK_MIN,
            min(
                GOLDEN_QUERY_CHUNK_MAX,
                GOLDEN_QK_BUDGET_BYTES // (N1_STATIC * max(1, full_s2) * 4),
            ),
        )
        for chunk_begin in range(0, used_q, query_chunk):
            chunk_end = min(chunk_begin + query_chunk, used_q)
            row_begin = q_begin + chunk_begin
            row_end = q_begin + chunk_end
            visible_counts = [
                _valid_key_count(
                    full_s2,
                    used_q,
                    local,
                    mask_mode,
                    cmp_ratio,
                    residual[b],
                )
                for local in range(chunk_begin, chunk_end)
            ]
            max_visible = max(visible_counts)
            if max_visible == 0:
                continue
            chunk_scale = None
            if query_dequant_scale is not None:
                chunk_scale = query_dequant_scale[row_begin:row_end]
            weighted = _query_scores(
                query[row_begin:row_end],
                chunk_scale,
                w[row_begin:row_end],
                logical_k[:max_visible],
                d,
            )
            visible = torch.tensor(
                visible_counts, dtype=torch.int64, device=weighted.device
            ).reshape(-1, 1, 1)
            token_position = torch.arange(
                max_visible, dtype=torch.int64, device=weighted.device
            ).reshape(1, 1, -1)
            row_values = weighted.masked_fill(token_position >= visible, float("-inf"))
            sparse_count = min(int(topk), max_visible)
            values, indices = _streaming_topk(
                row_values,
                sparse_count,
                _fp32_to_sortable_u16(row_values),
            )
            sparse_rank = torch.arange(
                sparse_count, dtype=torch.int64, device=weighted.device
            ).reshape(1, 1, -1)
            sparse_valid = sparse_rank < torch.minimum(
                visible,
                torch.tensor(sparse_count, dtype=torch.int64, device=weighted.device),
            )
            sparse_indices[row_begin:row_end, :, :sparse_count] = torch.where(
                sparse_valid,
                indices.to(torch.int32),
                torch.tensor(-1, dtype=torch.int32, device=weighted.device),
            )
            sparse_value_fp32[row_begin:row_end, :, :sparse_count] = torch.where(
                sparse_valid,
                values,
                torch.tensor(0.0, dtype=torch.float32, device=weighted.device),
            )

            if candidate_topk_blocks > 0:
                block_count = ceil_div(max_visible, CANDIDATE_BLOCK_SIZE)
                padded_tokens = block_count * CANDIDATE_BLOCK_SIZE
                if padded_tokens != max_visible:
                    padded_values = torch.full(
                        (row_values.shape[0], N2_STATIC, padded_tokens),
                        float("-inf"),
                        dtype=torch.float32,
                        device=weighted.device,
                    )
                    padded_values[..., :max_visible] = row_values
                else:
                    padded_values = row_values
                block_scores = padded_values.reshape(
                    row_values.shape[0],
                    N2_STATIC,
                    block_count,
                    CANDIDATE_BLOCK_SIZE,
                ).amax(dim=-1)
                visible_flat = visible.reshape(-1)
                partial_rows = torch.nonzero(
                    visible_flat % CANDIDATE_BLOCK_SIZE != 0,
                    as_tuple=False,
                ).flatten()
                if partial_rows.numel() > 0:
                    partial_blocks = (
                        visible_flat.index_select(0, partial_rows)
                        // CANDIDATE_BLOCK_SIZE
                    )
                    block_scores[partial_rows, 0, partial_blocks] = float("inf")
                candidate_count = min(int(candidate_topk_blocks), block_count)
                _, block_indices = _streaming_topk(
                    block_scores,
                    candidate_count,
                    _fp32_to_sortable_u16(block_scores),
                )
                visible_blocks = (
                    visible + CANDIDATE_BLOCK_SIZE - 1
                ) // CANDIDATE_BLOCK_SIZE
                candidate_rank = torch.arange(
                    candidate_count,
                    dtype=torch.int64,
                    device=weighted.device,
                ).reshape(1, 1, -1)
                candidate_valid = candidate_rank < torch.minimum(
                    visible_blocks,
                    torch.tensor(
                        candidate_count,
                        dtype=torch.int64,
                        device=weighted.device,
                    ),
                )
                candidate_indices[row_begin:row_end, :, :candidate_count] = torch.where(
                    candidate_valid,
                    block_indices.to(torch.int32),
                    torch.tensor(-1, dtype=torch.int32, device=weighted.device),
                )
                candidate_length[row_begin:row_end, :] = torch.minimum(
                    visible_blocks.reshape(-1, 1),
                    torch.tensor(
                        candidate_count,
                        dtype=torch.int64,
                        device=weighted.device,
                    ),
                ).to(torch.int32)

    sparse_values = (
        sparse_value_fp32.to(torch.bfloat16)
        if return_value
        else torch.zeros(
            (t, N2_STATIC, int(topk)), dtype=torch.bfloat16, device=query.device
        )
    )
    reference_provider = _ReferenceScoreProvider(
        query,
        query_dequant_scale,
        w,
        k_phys,
        page_ids,
        cu,
        used,
        used_k,
        residual,
        d,
        mask_mode,
        cmp_ratio,
    )
    return {
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "candidate_block_indices": candidate_indices,
        "candidate_block_length": candidate_length,
        # CPU-only lazy score source consumed only when sets differ at TopK.
        "_reference_scores": reference_provider,
        "_candidate_block_scores": reference_provider,
        "_return_value": bool(return_value),
    }


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "GROUP_SIZE",
    "pack_mxfp4",
    "quant_lightning_indexer_golden",
    "unpack_mxfp4",
    "validate_quant_lightning_indexer_contract",
]
